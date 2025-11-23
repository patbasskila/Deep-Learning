# Open Images Object Detection with Faster R-CNN (PyTorch + FiftyOne)

This project trains a **Faster R-CNN ResNet50 FPN** object detection model in PyTorch on a subset of **Open Images V6**, using **FiftyOne** to download and export data in Pascal VOC format.

The goal is to demonstrate an end-to-end object detection pipeline:
- dataset creation (multi-class, multi-source),
- a flexible custom `Dataset` that supports VOC-style XML/JSON/TXT annotations,
- robust training with error handling,
- and inference + visualization of detections.

---

## 🧾 Classes

The project focuses on the following classes:

```text
Dog, Cat, Deer, Bear, Bird, Person, Car, Truck, Airplane

These are mapped to integer labels starting at 1 (0 is reserved for background).

✨ Features

Data pipeline with FiftyOne

Uses fiftyone.zoo.open-images-v6 to download images for each class
Exports to Pascal VOC format (images/ + annotations/) per class

Flexible custom dataset

CustomDataset can read:
VOC-style XML
JSON annotations with objects[]
Simple text format: xmin ymin xmax ymax label
Combines multiple dataset roots into a single PyTorch Dataset
Returns image, target dict (boxes, labels), and meta info (paths) for debugging

Transforms and DataLoader
Basic transforms with ToTensor and optional RandomHorizontalFlip
Custom collate_fn to handle variable numbers of boxes and meta info

Model

torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
Replaced box predictor head with FastRCNNPredictor for custom num_classes

Training loop with error handling

Standard Faster R-CNN training (sum of loss components)
If a batch fails, it:
falls back to per-sample processing,
logs problematic samples (image + annotation paths + error),
continues training on the valid subset

Inference + visualization

Helper function predict_and_plot(...) to:
run inference on a single image
draw bounding boxes with class label + confidence score
display results with Matplotlib


📦 Requirements

Typical dependencies:

pip install torch torchvision fiftyone matplotlib


You may also need:

pip install opencv-python  # optional, if you add OpenCV-based utilities


💡 The notebooks are written with Google Colab-like paths in mind (/content/...).
