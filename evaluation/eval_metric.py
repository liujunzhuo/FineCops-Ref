"""
Usage:

python eval/eval_metric.py \
    --prediction_file xxx \
    --annotation_file xxx \
    --save_dir xxx
"""
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.utils.data

import re
import json
from torchvision.ops import box_iou
from pycocotools.coco import COCO
import sklearn.metrics as sk


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def save_scores_by_cate(data, cate):
    classified = defaultdict(list)
    classified_id_list = defaultdict(list)
    for item in data:
        try:
            score = max(item['score'])
        except Exception as e:
            print(e)
            print('no score > 0.3')
            continue
        key = tuple((item[name] for name in cate))
        classified[key].append(score)
        classified_id_list[key].append(item['pos_id'])
    sorted_keys = sorted(classified.keys(), key=lambda x: tuple(x[i] for i in range(len(key))))
    sorted_final_dict = {key: classified[key] for key in sorted_keys}
    return sorted_final_dict, classified_id_list


def save_scores_by_cate_and_pos_id(data, cate):
    classified = defaultdict(dict)
    for item in data:
        pos_id = item['pos_id']
        key = tuple((item[name] for name in cate))
        classified[key].setdefault(pos_id, []).extend(item['score'])
        # classified_id_list[key].append(item[cate_id_map['pos_id']])
    sorted_keys = sorted(classified.keys(), key=lambda x: tuple(x[i] for i in range(len(key))))
    sorted_final_dict = {key: classified[key] for key in sorted_keys}
    return sorted_final_dict


VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000


def resize_bbox(box, image_w=None, image_h=None):
    ratio_w = image_w * 1.0 / VOCAB_IMAGE_W
    ratio_h = image_h * 1.0 / VOCAB_IMAGE_H

    new_box = [int(box[0] * ratio_w), int(box[1] * ratio_h),
               int(box[2] * ratio_w), int(box[3] * ratio_h)]
    return new_box


def decode_bbox_from_caption(text, img_w, img_h, verbose=False):
    entities = []
    boxes = []

    start = 0
    in_brackets = False
    entity = ""
    box = ""

    for i, char in enumerate(text):
        if char == '[':
            in_brackets = True
            entity = text[start:i].strip()
            start = i + 1
        elif char == ']':
            in_brackets = False
            box = text[start:i].strip()
            start = i + 1

            # Convert box string to list of integers
            box_list = list(map(int, box.split(',')))
            try:
                resized_box_list = resize_bbox(box_list, img_w, img_h)
            except Exception as e:
                print(f'unexpected box: {box_list}')
                continue
            entities.append(entity)
            boxes.append(resized_box_list)

            # Skip until the next entity (ignoring periods or other delimiters)
            while start < len(text) and text[start] not in ['.', ',', ';', '!', '?']:
                start += 1
            start += 1  # Skip the delimiter

    return entities, boxes


def find_top_k_indices(pos_score_list, negative_score_list, k):
    combined_scores = negative_score_list + pos_score_list
    combined_scores.sort(reverse=True)

    top_k_scores = combined_scores[:k]

    indices = [idx for idx, score in enumerate(pos_score_list) if score in top_k_scores]

    return indices


def are_phrases_similar(phrase1, phrase2):
    # Step 1: Convert to lower case
    phrase1 = phrase1.lower()
    phrase2 = phrase2.lower()

    # Step 2: Standardize spacing around punctuation
    phrase1 = re.sub(r'\s*([\'",.;!?|:])\s*', r'\1 ', phrase1).strip()
    phrase2 = re.sub(r'\s*([\'",.;!?|:])\s*', r'\1 ', phrase2).strip()

    # Step 3: Remove all punctuation
    phrase1 = re.sub(r'[^\w\s]', '', phrase1)
    phrase2 = re.sub(r'[^\w\s]', '', phrase2)

    # Step 4: Remove extra white spaces
    phrase1 = ' '.join(phrase1.split())
    phrase2 = ' '.join(phrase2.split())

    return phrase1 == phrase2


class RefExpEvaluatorFromJsonl(object):
    def __init__(self, refexp_gt_path, k=(1, -1), iou_thrs=0.5, score_thrs=0, save_dir=None, save_failure=False,
                 score_type='sum_log_prob'):
        assert isinstance(k, (list, tuple))
        # with open(refexp_gt_path, 'r') as f:
        #     self.refexp_gt = json.load(f)
        # self.img_ids = [item['id'] for item in self.refexp_gt['images']]
        # print(f"Load {len(self.img_ids)} images")
        # print(f"Load {len(self.refexp_gt['annotations'])} annotations")
        self.k = k
        self.topk = [1, 5, 10]
        self.iou_thrs = iou_thrs
        self.score_thrs = score_thrs
        self.coco = COCO(refexp_gt_path)
        self.save_dir = save_dir
        self.save_failure = save_failure
        self.score_type = score_type
        os.makedirs(save_dir, exist_ok=True)

    def summarize(self,
                  prediction_file: str,
                  verbose: bool = False, ):
        # get the predictions
        if os.path.isfile(prediction_file):
            predictions = [json.loads(line) for line in open(prediction_file)]
        elif os.path.isdir(prediction_file):
            predictions = [json.loads(line) for pred_file in os.listdir(prediction_file) for line in
                           open(os.path.join(prediction_file, pred_file))]
        else:
            raise NotImplementedError('Not supported file format.')

        dataset2score = {
            1: {k: 0.0
                for k in self.k},
            2: {k: 0.0
                for k in self.k},
            3: {k: 0.0
                for k in self.k},
        }
        dataset2count = {1: 0.0, 2: 0.0, 3: 0.0}

        in_score = []
        out_score = []
        in_score_level = dict()
        negative_data_list = []
        results_id_to_iou = dict()
        results_id_to_score = dict()
        failure_list = []  # id, box, score
        for prediction in predictions:
            img_id = prediction['image_id']
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            assert len(ann_ids) == 1
            img_info = self.coco.loadImgs(img_id)[0]
            target = self.coco.loadAnns(ann_ids[0])
            img_height = img_info['height']
            img_width = img_info['width']
            caption = img_info['caption']
            target_bbox = target[0]['bbox']
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            target_bbox = torch.as_tensor(converted_bbox).view(-1, 4)
            prediction_text = prediction["text"]
            try:
                entities, boxes = decode_bbox_from_caption(prediction_text, img_width, img_height, verbose=verbose)
            except ValueError as e:
                print(f"Can't find any bbox for the given phrase {caption}")
                entities, boxes = [], []
            predict_boxes = []
            for (entity, box) in zip(entities, boxes):
                if not are_phrases_similar(entity, caption):
                    if len(box) > 0:
                        predict_boxes.append(box)
                else:
                    predict_boxes.append(box)
            try:
                pred_scores = np.array(prediction[self.score_type])
            except KeyError as e:
                print(e)

            if len(predict_boxes) == 0:
                print(f"Can't find valid bbox for the given phrase {caption}, \n{entities, boxes}")
                print(f"We set a 0-area box to calculate result")
                predict_boxes = [[0., 0., 0., 0.]]
                pred_scores = np.array([0])
            # sort by score
            sorted_indices = np.argsort(pred_scores)[::-1]
            predict_boxes = [predict_boxes[i] for i in sorted_indices]
            pred_scores = pred_scores[sorted_indices]

            predict_boxes = torch.as_tensor(predict_boxes).view(-1, 4).to(dtype=torch.float32)
            score_thr = pred_scores[pred_scores >= self.score_thrs]
            score = max(pred_scores)
            if target[0].get('negative_type', None) is None:
                iou = box_iou(predict_boxes, target_bbox)
                mean_iou = box_iou(predict_boxes.mean(0).view(-1, 4), target_bbox)
                for k in self.k:
                    if max(iou[:k]) >= self.iou_thrs:
                        dataset2score[target[0]['level']][k] += 1.0
                if max(iou[:1]) < self.iou_thrs:
                    failure_list.append([img_id, predict_boxes[0].tolist(), float(score)])
                dataset2count[target[0]['level']] += 1.0
                in_score_level.setdefault(target[0]['level'], []).append(score)
                in_score.append(score)
                results_id_to_score[str(img_id)] = score_thr
                results_id_to_iou[str(img_id)] = iou
            else:
                out_score.append(score)
                negative_level = target[0]['negative_level']

                negative_data = {"negative_level": negative_level, "positive_level": target[0]['level'],
                                 "negative_type": target[0]['negative_type'], "tuple_type": target[0]['tuple_type'],
                                 "score": score_thr, "pos_id": str(target[0]['positive_id']),
                                 "negative_cate": target[0]['negative_cate']}
                negative_data_list.append(negative_data)

        print(f"count positive: {dataset2count}")
        if self.save_failure:
            json.dump(failure_list, open(f'{self.save_dir}/failure_list', 'w'))
        for key, value in dataset2score.items():
            for k in self.k:
                try:
                    value[k] /= dataset2count[key]
                except Exception as e:
                    print(e)
        save_results = {}
        for key, value in dataset2score.items():
            save_results[key] = sorted([round(v * 100, 2) for k, v in value.items()])
            print(f" Dataset: {key} - Precision @ 1, mean, all: {save_results[key]} \n")

        precision_df = pd.DataFrame(save_results).T
        precision_df.columns = ["Precision @ 1", " Precision @ 5", "Precision @ 10"]
        precision_df.insert(loc=0, column='level', value=[1, 2, 3])
        precision_df.to_csv(os.path.join(self.save_dir, f'precision.csv'), index=False)
        recall_dict = {}
        # ["negative_level", "negative_type", "tuple_type", "negative_cate"]
        cate = ["negative_type", "negative_level", "negative_cate"]
        negative_score_dict = save_scores_by_cate_and_pos_id(negative_data_list, cate)
        for neg_type, neg_score_dict in negative_score_dict.items():
            print(f"type: {dict(zip(cate, neg_type))} count:{len(neg_score_dict)} ---------")
            recall_count = 0
            recall = {k: 0.0 for k in self.topk}
            for pos_id, neg_score_list in neg_score_dict.items():
                pos_score_list = results_id_to_score[str(pos_id)]
                iou = results_id_to_iou[str(pos_id)]
                for k in self.topk:
                    indices = find_top_k_indices(pos_score_list.tolist(), neg_score_list, k)
                    max_index = min(len(indices), k)
                    if max_index > 0 and max(iou[:max_index]) >= self.iou_thrs:
                        recall[k] += 1.0
                recall_count += 1.0
            save_recall = {}
            mean_recall = 0.0
            for key, value in recall.items():
                save_recall[key] = round(value / recall_count * 100, 2)
                mean_recall += save_recall[key]
            recall_dict[neg_type] = list(save_recall.values()) + [len(neg_score_dict)]
        recall_df = pd.DataFrame(recall_dict).T
        recall_df.reset_index(inplace=True)
        recall_df.columns = ['type', 'level', 'cate', "Recall @ 1", " Recall @ 5", "Recall @ 10", "count"]
        recall_df.sort_values(by=['cate', 'Recall @ 1'], ascending=[False, False], inplace=True)
        recall_df['count'] = recall_df['count'].astype(int)
        recall_df.to_csv(os.path.join(self.save_dir, f'recall_{cate}.csv'), index=False)

        auroc, aupr, fpr = get_measures(in_score_level[1], out_score)
        print(f"Overall AUROC--------------")
        print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
        print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * 0.95), 100 * fpr))
        print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
        count_negative = dict()
        overall_dict = {tuple(['overall', 0, 0]): [round(auroc * 100, 2), round(fpr * 100, 2), round(100 * aupr, 2),
                                                   len(out_score)]}
        result_dict = dict()
        final_dict, pos_id_dict = save_scores_by_cate(negative_data_list, cate)
        for types, score in final_dict.items():
            print(f"type: {dict(zip(cate, types))} count:{len(score)} ---------")
            count_negative[str(dict(zip(cate, types)))] = len(score)
            pos_id_list = pos_id_dict[types]
            in_score_tmp = [max(results_id_to_score[str(idx)]) for idx in pos_id_list]
            auroc, aupr, fpr = get_measures(in_score_tmp, score)
            # auroc, aupr, fpr = get_measures(in_score_level[1], score)
            result_dict[types] = [round(auroc * 100, 2), round(fpr * 100, 2), round(100 * aupr, 2), len(score)]
            # plot_distribution(round(auroc * 100, 2), in_score_level[1], score, dict(zip(cate, types)))
            print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
            print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * 0.95), 100 * fpr))
            print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
        result_dict.update(overall_dict)
        auroc_df = pd.DataFrame(result_dict).T
        auroc_df.reset_index(inplace=True)
        auroc_df.columns = ['type', 'level', 'cate', "AUROC", "FPR95", "AUPR", "count"]
        auroc_df['count'] = auroc_df['count'].astype(int)
        auroc_df.sort_values(by=['cate', 'AUROC'], ascending=[False, False], inplace=True)
        auroc_df.to_csv(os.path.join(self.save_dir, f'auroc_{cate}.csv'), index=False)

        return precision_df, recall_df, auroc_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', help='prediction_file')
    parser.add_argument('--annotation_file', default='/path/to/json_annotations', help='annotation_file')
    parser.add_argument("--save_dir", default='/path/save_result', type=str)
    parser.add_argument("--score_type", default='sum_log_prob', type=str, choices=['scores_prob_average', 'scores_logits_average', 'sum_log_prob'])
    args = parser.parse_args()

    evaluator = RefExpEvaluatorFromJsonl(
        refexp_gt_path=args.annotation_file,
        k=(1, 5, 10),
        iou_thrs=0.5,
        save_dir=args.save_dir,
        score_type=args.score_type
    )

    results = evaluator.summarize(args.prediction_file, verbose=False)
