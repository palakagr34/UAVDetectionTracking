import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import ToTensor
import torchvision
from xml.etree import ElementTree as ET
from PIL import Image

class DroneDatasetXML(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.xml_folder = os.path.join(root, 'dataset_xml_format', 'dataset_xml_format')
        self.image_folder = os.path.join(root, 'dataset_xml_format', 'dataset_xml_format')

        self.image_files = [f for f in os.listdir(self.image_folder) if f.endswith('.png')]

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        xml_name = os.path.join(self.xml_folder, self.image_files[idx].replace('.png', '.xml'))

        image = Image.open(img_name).convert("RGB")
        boxes, labels = self.parse_xml_annotation(xml_name)

        target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                  "labels": torch.tensor(labels, dtype=torch.int64)}

        image = ToTensor()(image)

        return image, target

    def __len__(self):
        return len(self.image_files)

    def parse_xml_annotation(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('.//object'):
            label = obj.find('name').text
            if label == 'drone':
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)

        return boxes, labels

backbone = torchvision.models.resnet50(pretrained=True)
in_features = backbone.fc.in_features
backbone.fc = torch.nn.Identity()
backbone.out_channels = in_features

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

num_classes = 2
model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

root_dir = "./archive"
transform = ToTensor()
train_dataset = DroneDatasetXML(root_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

num_epochs = 10
if __name__ == "__main__":

    for epoch in range(num_epochs):
        for images, targets in train_dataloader:
            images = list(image.to(device) for image in images)

            if isinstance(targets, dict):
                targets = {k: v.to(device) for k, v in targets.items()}
            elif isinstance(targets, list):
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            else:
                raise ValueError(f"Unsupported targets format: {type(targets)}")

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()

    torch.save(model.state_dict(), "./model.pth")
