## Datasets
- GQA Image: you can download for there [website](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- Annotations and negative images: [Dataset](https://figshare.com/s/e323fe078924c8b36043?file=47091109). We provide both vanilla and coco format annotations.

The vanilla annotations consist of positive and negative data samples with specific keys and structures as described below:

### Positive Data Format

```json
{
  "id": "Unique identifier for the data sample.",
  "image_id": "Stores the GQA image ID.",
  "objects_id": "Stores the corresponding original object IDs from the GQA scene graph.",
  "tuple": "Stores the generated path in the format of subject-relation-object pairs.",
  "tuple_type": "Indicates the type of tuple, such as '2_hop'.",
  "level": "Specifies the difficulty level of the sample.",
  "spatial": "Reserved for spatial information for the target object.",
  "attribute": "Stores attributes of the objects in the tuple.",
  "expression": "Referring expression."
}
```

### Negative Data Format

```json
{
  "id": "Unique identifier for the data sample.",
  "image_id": "Stores the GQA image ID or neg_image_id.",
  "objects_id": "Stores the corresponding original object IDs from the GQA scene graph.",
  "tuple": "Stores the generated path in the format of subject-relation-object pairs.",
  "tuple_type": "Indicates the type of tuple, such as 'and'.",
  "level": "Specifies the difficulty level of the sample.",
  "spatial": "Reserved for spatial information.",
  "attribute": "Stores attributes of the objects in the tuple.",
  "expression": "Referring expression",
  "positive_id": "Stores the ID of the corresponding positive data sample.",
  "negative_level": "Specifies the difficulty level of the negative sample.",
  "negative_type": "Indicates the type of negative sample, such as 'attribute'.",
  "shuffle_index": "Stores the shuffle indices used to create the negative sample.",
  "ori_type": "Indicates the original type of the tuple, such as '2_hop'.",
  "negative_cate": "Can be 'text' or 'image', indicating the category of negativity."
}
```

### Usage
The coco format annotations is consistent with [Refcoco/+/g](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1).

You can load the annotations simply with:

```python
# if you want to load the image
torchvision.datasets.CocoDetection(image_root, annfile)
# if you want to load the annotations only
pycocotools.coco.COCO(annfile)
```


## Benchmarking

To perform benchmark evaluation, you first need to run the model inference on the benchmark dataset and save its prediction results, including image_id, bounding box/response, and the corresponding score for each box. Note that the score is only used to calculate recall and AUROC. Then, calculate the metrics. 

### Inference

Here is an example for [Hugging Face model inference](evaluation/run_hf_model.py) for CogVLM

```sh
python evaluation/run_hf_model.py
--image_path <image_root path> \
--data_path <dataset>_coco_format.json \
--answers-file <path_to_prediction_file>.jsonl \
--chunk-idx 0 \
--num-chunks 1 \
--bf16 \
--from_pretrained THUDM/cogvlm-grounding-generalist-hf \
# --local_tokenizer <tokenizer path>
```

For other models, you should modify the following parts:
- [get_score](./evaluation/run_hf_model.py#L88)
- [instruction template](evaluation/run_hf_model.py#L58)

Additionally, adjust the chat template or other parts as specified by the different models.

### Evaluation

Below is an introduction to the result saving format and evaluation script.
#### Specialist
To evaluate the [MM-GDINO](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino), you can use the [eval_metric_mmdet.py](./evaluation/eval_metric_mmdet.py)

`eval_metric_mmdet.py` can be directly served as the metric in mmdetection framework.

**Usage:**

```sh
python eval/eval_metric_mmdet.py \
    --result_dir <path_to_result_dir> \
    --annotation_file <path_to_annotation_file> \
    --save_dir <path_to_save_dir>
```

- `result_dir` should be in `.npy` format and contain a list of dictionaries. Each dictionary should be formatted as follows:
  ```python
  {
    'img_id': 1,
    'bboxes': array([[200.604, 92.932144, 499.06314, 332.1396]], dtype=float32),
    'scores': array([0.63060474], dtype=float32)
  }
  ```


#### MLLMs

Example of evaluation for [ferret](https://github.com/apple/ml-ferret) at [eval_metric.py](./evaluation/eval_metric.py)

**Usage:**

```sh
python eval/eval_metric.py \
    --prediction_file <path_to_prediction_file> \
    --annotation_file <path_to_annotation_file> \
    --save_dir <path_to_save_dir>
```

- `prediction_file` should be in `.jsonl` format, with each line formatted as follows:

  ```json
  {
    "image_id": 1,
    "text": "The boy riding the brown horse positioned ahead of the white truck. in the image [526, 15, 798, 757].",
    "scores_logits_average": [51.3125],
    "scores_prob_average": [0.85205078125],
    "sum_log_prob": [0.0029430389404296875]
  }
  ```
For evaluating other models, you may need to modify the following parts:

1. [VOCAB SIZE](./evaluation/eval_metric.py#L133). Indicates the coordinate scaling factor for bounding boxes in the model's output. Adjust as needed. For example, LLAVA uses a scale of 1 (normalized coordinates like [0.1, 0.2, 0.45, 0.22]), while Ferret/CogVLM uses a scale of 1000 (coordinates like [100, 200, 450, 220]).
2. [decode the box form caption](./evaluation/eval_metric.py#L146). Customize the `decode_bbox_from_caption` function based on your model's output format.

   - Example function for `cogvlm/cogcom`:
   ```python
   def decode_bbox_from_caption(text, img_w, img_h, verbose=False):
       pattern = r'\[\[(\d+),(\d+),(\d+),(\d+)\]\]'  # [[x1,y1,x2,y2]] 
       match = re.findall(pattern, text)
       if match:
           bounding_box = list(map(list, match))
           return '', bounding_box
       else:
           return '', None
   ```

## Training
