import json

json_path = r"C:\Users\umakirk\nnUNet_raw\Dataset100_KidneyUS\dataset.json"
dataset = {
  "channel_names": {
    "0": "US"
  },
  "labels": {
    "background": 0,
    "kidney": 1
  },
  "numTraining": 59,
  "file_ending": ".nii.gz",
  "overwrite_image_reader_writer": "SimpleITKIO"
}

with open(json_path, "w") as f:
    json.dump(dataset, f, indent=4)
print(f"Saved: {json_path}")
