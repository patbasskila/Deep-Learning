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


🗂 Data Preparation (FiftyOne + Open Images)

Select classes
The notebook defines:

classes = ["Dog", "Cat", "Deer", "Bear", "Bird", "Person", "Car", "Truck", "Airplane"]


Download with FiftyOne Zoo

For each class, the notebook:

dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    classes=[cls],
    max_samples=500,  # tune as needed
    shuffle=True,
)


Export to Pascal VOC

Export each class to a folder like /content/dataset/<ClassName> with:

dataset.export(
    export_dir=export_dir,
    dataset_type=fot.VOCDetectionDataset,
    label_field="detections",
)


You end up with directories like:

/content/dataset/Dog/
    images/
    annotations/
...


🧰 Custom Dataset & DataLoader

The CustomDataset:

Accepts one or many dataset roots (e.g. all class folders):

dataset_roots = [
    "/content/dataset/Dog",
    "/content/dataset/Cat",
    ...
]
dataset = CustomDataset(dataset_roots, transforms=get_transform(train=True), classes_map=classes_map)


Handles XML, JSON, and TXT formats in parse_annotation(...).

Returns (image, target, meta) where:

target["boxes"] → tensor of [xmin, ymin, xmax, ymax]

target["labels"] → class indices

meta holds img_path and ann_path for debugging

Custom collate_fn for DataLoader:

def collate_fn(batch):
    images, targets, metas = zip(*batch)
    return list(images), list(targets), list(metas)

data_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)

🏋️ Training

Model setup:

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

num_classes = 1 + len(classes_map)  # background + object classes
model = get_model(num_classes).to(device)


Optimizer & scheduler:

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

🔍 Inference & Visualization

The notebook provides a helper like:

def predict_and_plot(image_path, model, device, threshold=0.5):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])

    # plot image and draw bounding boxes + labels above threshold


Draws red rectangles for detections

Labels each box with class_name and confidence score

Uses reverse_classes_map to convert integer labels back to strings
