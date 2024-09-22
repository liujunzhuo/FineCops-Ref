"""
This is a demo for using CogAgent and CogVLM in CLI
Make sure you have installed vicuna-7b-v1.5 tokenizer model (https://huggingface.co/lmsys/vicuna-7b-v1.5), full checkpoint of vicuna-7b-v1.5 LLM is not required.
In this demo, We us chat template, you can use others to replace such as 'vqa'.
Strongly suggest to use GPU with bfloat16 support, otherwise, it will be slow.
Mention that only one picture can be processed at one conversation, which means you can not replace or insert another picture during the conversation.
"""

import argparse
import json
import os
from copy import deepcopy

import numpy as np
import torch
import torchvision

from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000


def resize_bbox(box, image_w=None, image_h=None):
    ratio_w = image_w * 1.0 / VOCAB_IMAGE_W
    ratio_h = image_h * 1.0 / VOCAB_IMAGE_H

    new_box = [int(box[0] * ratio_w), int(box[1] * ratio_h), \
               int(box[2] * ratio_w), int(box[3] * ratio_h)]
    return new_box


def remove_punctuation(text: str) -> str:
    punct = [',', ]
    for p in punct:
        text = text.replace(p, '')
    return text.strip()


class RefExpGrounding(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(RefExpGrounding, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.question_prompt = "Where is <expr>? answer in [[x0,y0,x1,y1]] format."

    def __getitem__(self, idx):
        img, target = super(RefExpGrounding, self).__getitem__(idx)
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]
        file_name = coco_img["file_name"]
        caption = coco_img["caption"]
        dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None
        assert len(target) == 1
        bbox_xywh = target[0]["bbox"]
        bbox_xyxy = np.array([bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]])
        w, h = img.size
        bbox_xyxy[0::2].clip(min=0, max=w)
        bbox_xyxy[1::2].clip(min=0, max=h)

        assert "<expr>" in self.question_prompt
        question = self.question_prompt.replace("<expr>", remove_punctuation(caption))

        target = {"image_id": image_id, "file_name": file_name, "annotations": target, "caption": caption,
                  "img_w": w, "img_h": h, "question": question, "bboxes": bbox_xyxy.tolist(), "entities": [caption]}
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        target["dataset_name"] = dataset_name
        for extra_key in ["sentence_id", "original_img_id", "original_id", "task_id"]:
            if extra_key in coco_img:
                target[extra_key] = coco_img[extra_key]
        return img, target


def get_score(tokens, transition_scores, transition_scores_norm, tokenizer):
    entities = []
    boxes = []
    scores_logits_average = []
    scores_prob_multi = []
    scores_prob_average = []
    tokens_length = []

    start = 0
    entity = ""

    for i, char in enumerate(tokens):
        if tokenizer.decode(char) == '[[':  # [
            entity = tokens[start:i]
            start = i + 1
        elif tokenizer.decode(char) == ']]':  # ]
            box_token = tokens[start:i]
            try:
                score_logits_average = sum(transition_scores[start - 1:i + 1]) / len(transition_scores[start - 1:i + 1])
                score_log_sum = sum(transition_scores_norm[start - 1:i + 1])
                token_length = len(tokens[start - 1:i + 1])
                score_prob_average = sum(torch.exp(transition_scores_norm[start - 1:i + 1])) / len(
                    transition_scores_norm[start - 1:i + 1])
            except Exception as e:
                print(e, box_token)
                start = i + 1
                continue
            start = i + 1
            box = tokenizer.decode(box_token, skip_special_tokens=True).strip()

            # valid = torch.nonzero(torch.eq(box_token, 29892)).size(0) == 3

            # Convert box string to list of integers
            try:
                box_list = list(map(int, box.split(',')))
            except Exception as e:
                print(e, box)
                continue
            try:
                resized_box_list = resize_bbox(box_list, 100, 100)
            except Exception as e:
                print(f'unexpected box: {box_list}')
                continue
            entities.append(entity)
            boxes.append(resized_box_list)
            scores_logits_average.append(float(score_logits_average))
            scores_prob_multi.append(float(torch.exp(score_log_sum)))
            scores_prob_average.append(float(score_prob_average))
            tokens_length.append(token_length)

            # Skip until the next entity (ignoring periods or other delimiters)
            while start < len(tokens) and tokens[start] not in tokenizer(['.', ',', ';', '!', '?']):
                start += 1
            start += 1  # Skip the delimiter

    return scores_logits_average, scores_prob_multi, scores_prob_average, tokens_length


def eval_model(args):
    MODEL_PATH = args.from_pretrained
    TOKENIZER_PATH = args.local_tokenizer
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


    if args.bf16:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16

    print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        # low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(DEVICE).eval()

    dataset = RefExpGrounding(img_folder=args.image_path,
                              ann_file=args.data_path,
                              transforms=None)

    data_ids = range(len(dataset))
    chunk_data_ids = get_chunk(data_ids, args.num_chunks, args.chunk_idx)

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = args.answers_file
    # answers_file = os.path.join(answers_file, f'{args.chunk_idx}_of_{args.num_chunks}.jsonl')
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for i, id in enumerate(tqdm(chunk_data_ids)):
        image, label = dataset[id]
        query = label["question"]
        history = []
        # query = '' + "USER: {} ASSISTANT:".format(query)

        if image is None:
            input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history,
                                                                template_version='base')
        else:
            input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history,
                                                                images=[image])

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image is not None else None,
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

        # add any transformers params here.
        gen_kwargs = {"max_length": 2048,
                      "do_sample": False,
                      "return_dict_in_generate": True,
                      "output_scores": True
                      }  # "temperature": 0.9
        with torch.no_grad():
            output_dict = model.generate(**inputs, **gen_kwargs)

            outputs, output_score = output_dict.sequences, output_dict.scores
            transition_scores_norm = model.compute_transition_scores(
                outputs, output_score, normalize_logits=True)
            transition_scores = model.compute_transition_scores(
                outputs, output_score, normalize_logits=False)
            output_tokens = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(output_tokens[0])
            response = response.split("</s>")[0]
            scores_logits_average, scores_prob_multi, scores_prob_average, tokens_length = get_score(
                output_tokens[0], transition_scores[0], transition_scores_norm[0], tokenizer)

        ans_file.write(json.dumps({"image_id": label['image_id'],
                                   "file_name": label["file_name"],
                                   "prompt": query,
                                   "text": response,
                                   "width": label['img_w'],
                                   "height": label['img_h'],
                                   'scores_logits_average': scores_logits_average,
                                   'scores_prob_average': scores_prob_average,
                                   'sum_log_prob': scores_prob_multi,
                                   'tokens_length': tokens_length
                                   }) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_pretrained", type=str, default="cogvlm-grounding-generalist-hf",
                        help='pretrained ckpt')
    parser.add_argument("--image_path", type=str, default="data/gqa")
    parser.add_argument("--data_path", type=str, default="data/annotations/test_coco_format.json")
    parser.add_argument("--local_tokenizer", type=str, default="vicuna-7b-v1.5",
                        help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    args = parser.parse_args()
    eval_model(args)
