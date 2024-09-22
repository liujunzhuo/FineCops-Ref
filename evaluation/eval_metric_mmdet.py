# Copyright (c) OpenMMLab. All rights reserved.
import collections
import json
import os
import random
from collections import defaultdict
from typing import Dict, Optional, Sequence

import argparse
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import get_local_path
from mmengine.logging import MMLogger
from scipy.stats import gaussian_kde

from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import METRICS
from mmdet.evaluation import bbox_overlaps
import sklearn.metrics as sk
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


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


def plot_distribution(auroc, id_scores, ood_scores, data_type):
    # min_length = min(len(id_scores), len(ood_scores))
    # if len(id_scores) > min_length:
    #     id_scores = random.sample(id_scores, min_length)
    # elif len(ood_scores) > min_length:
    #     ood_scores = random.sample(ood_scores, min_length)
    kde_id = gaussian_kde(id_scores)
    kde_ood = gaussian_kde(ood_scores)
    x_values = np.linspace(0, 1, 100)
    # plt.figure(figsize=(1, 3))
    plt.plot(x_values, kde_id(x_values), label="ID", color='blue')
    plt.plot(x_values, kde_ood(x_values), label="OOD", color='red')

    # plt.fill_between(x_values, kde_id(x_values), alpha=0.8, color='#A8BAE3')
    # plt.fill_between(x_values, kde_ood(x_values), alpha=0.8, color='#55AB83')
    plt.legend()
    plt.text(0.5, -0.1, str(data_type) + ' AUROC: ' + str(auroc), horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 3)
    plt.show()


def generalized_entropy(softmax_id_val, gamma=0.1, M=10):
    probs = softmax_id_val
    probs_sorted = np.sort(probs)[-M:]
    scores = np.sum(probs_sorted ** gamma * (1 - probs_sorted) ** (gamma))

    return -scores


def find_top_k_indices(pos_score_list, negative_score_list, k):
    combined_scores = negative_score_list + pos_score_list
    combined_scores.sort(reverse=True)

    top_k_scores = combined_scores[:k]

    indices = [idx for idx, score in enumerate(pos_score_list) if score in top_k_scores]

    return indices


@METRICS.register_module()
class RefRecallAUROC(BaseMetric):
    default_prefix: Optional[str] = 'refrecall_auroc'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: str = 'bbox',
                 topk=(1, 5, 10),
                 iou_thrs: float = 0.5,
                 save_dir: str = None,
                 score_thrs: float = 0,
                 save_failure: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.topk = topk
        self.iou_thrs = iou_thrs
        self.save_dir = save_dir
        self.score_thrs = score_thrs
        self.save_failure = save_failure
        os.makedirs(save_dir, exist_ok=True)
        self.coco = COCO(ann_file)

    def save_results(self, results, default_path):
        if self.save_dir is not None:
            file_path = os.path.join(self.save_dir, 'results_cache.npy')
        else:
            file_path = default_path

        # Check if the file already exists
        if os.path.exists(file_path):
            # If the file exists, create a new filename
            base_name, extension = os.path.splitext(file_path)
            index = 1
            new_file_path = f"{base_name}_{index}{extension}"

            while os.path.exists(new_file_path):
                index += 1
                new_file_path = f"{base_name}_{index}{extension}"
            if index > 2:
                print(f"File already exists more than {index - 1}, don't save")
                return

            file_path = new_file_path
            print(f"File already exists. Saving as {file_path}")

        np.save(file_path, results)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: list) -> int:
        logger: MMLogger = MMLogger.get_current_instance()
        self.save_results(results, default_path="mm_results_cache.npy")

        dataset2score = {
            1: {k: 0.0
                for k in self.topk},
            2: {k: 0.0
                for k in self.topk},
            3: {k: 0.0
                for k in self.topk},
        }
        dataset2count = {1: 0.0, 2: 0.0, 3: 0.0}

        in_score = []
        out_score = []
        in_score_level = dict()
        negative_data_list = []
        results_id_to_iou = dict()
        results_id_to_score = dict()
        failure_list = []  # id, box, score
        for result in results:
            img_id = result['img_id']
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            assert len(ann_ids) == 1
            img_info = self.coco.loadImgs(img_id)[0]
            target = self.coco.loadAnns(ann_ids[0])
            target_bbox = target[0]['bbox']
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            score = max(result['scores'])
            score_thr = result['scores'][result['scores'] > self.score_thrs]
            if target[0].get('negative_type', None) is None:
                iou = bbox_overlaps(result['bboxes'],
                                    np.array(converted_bbox).reshape(-1, 4))
                for k in self.topk:
                    if max(iou[:k]) >= self.iou_thrs:
                        dataset2score[target[0]['level']][k] += 1.0
                if max(iou[:1]) < self.iou_thrs:
                    failure_list.append([img_id, result['bboxes'][0].tolist(), float(score)])
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
            for k in self.topk:
                try:
                    value[k] /= dataset2count[key]
                except Exception as e:
                    print(e)
        save_results = {}
        mean_precision = 0.0
        for key, value in dataset2score.items():
            save_results[key] = sorted([round(v * 100, 2) for k, v in value.items()])
            mean_precision += sum(save_results[key])
            logger.info(
                f' Level: {key} - Precision @ 1, 5, 10: {save_results[key]}')
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
            logger.info(
                f' Recall: {save_recall}')
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
        print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
        print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * 0.95), 100 * fpr))
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

        return result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default='/mm_results_cache.npy', type=str)
    parser.add_argument('--annotation_file', default='/path/to/json_annotations', help='annotation_file')
    parser.add_argument("--save_dir", default='/path/save_result', type=str)
    args = parser.parse_args()

    result = np.load(args.result_dir, allow_pickle=True)
    evaluator = RefRecallAUROC(
        args.anno_file,
        save_dir=args.save_dir)
    evaluator.compute_metrics(result)
