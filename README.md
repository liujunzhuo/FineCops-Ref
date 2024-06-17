## Datasets

- [Dataset](https://figshare.com/s/e323fe078924c8b36043?file=47091109)
consists of positive and negative data samples with specific keys and structures as described below:

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

## Benchmarking

## Training
