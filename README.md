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

example of evaluation at [eval_metric.py](./evaluation/eval_metric.py)
"""
Usage:

python eval/eval_metric.py \
    --prediction_file xxx \
    --annotation_file xxx \
    --save_dir xxx
"""

## Training
