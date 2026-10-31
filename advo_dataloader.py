import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


# -----------------------------------
# 1. Configuration
# -----------------------------------

SPLIT_FILE = Path(
    "/home/rag/datasets/FaceForensicsPP/frames_advo/splits_advo.json"
)

IMAGE_SIZE = 224
BATCH_SIZE = 8

LABELS = {
    "real": 0,
    "fake": 1
}


# -----------------------------------
# 2. Custom PyTorch Dataset
# -----------------------------------

class DeepfakeDataset(Dataset):

    def __init__(self, split_file, split_name):

        with open(split_file, "r") as f:
            splits = json.load(f)

        if split_name not in splits:
            raise ValueError(f"Unknown split: {split_name}")

        self.samples = []

        # Convert video-level entries into image-level samples
        for video in splits[split_name]:

            label = LABELS[video["label"]]

            for frame_path in video["frames"]:
                self.samples.append(
                    (frame_path, label)
                )

        print(
            f"{split_name.upper()} dataset loaded: "
            f"{len(self.samples)} images"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        # Load image and convert to grayscale
        image = Image.open(image_path).convert("L")

        # Resize image
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # Convert to NumPy array
        image = np.array(image, dtype=np.float32)

        # Normalize to [0, 1]
        image = image / 255.0

        # Convert to tensor: [1, H, W]
        image = torch.from_numpy(image).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


# -----------------------------------
# 3. Create datasets
# -----------------------------------

train_dataset = DeepfakeDataset(
    SPLIT_FILE, "train"
)

validation_dataset = DeepfakeDataset(
    SPLIT_FILE, "validation"
)

test_dataset = DeepfakeDataset(
    SPLIT_FILE, "test"
)


# -----------------------------------
# 4. Create DataLoaders
# -----------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# -----------------------------------
# 5. Test one batch
# -----------------------------------

images, labels = next(iter(train_loader))

print("\n========== DATALOADER TEST ==========")

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("Image data type:", images.dtype)
print("Label data type:", labels.dtype)

print("Minimum pixel value:", images.min().item())
print("Maximum pixel value:", images.max().item())

print("Labels in first batch:", labels.tolist())

print("\nNumber of training batches:", len(train_loader))
print("Number of validation batches:", len(validation_loader))
print("Number of testing batches:", len(test_loader))

print("\nDataLoader test completed successfully.")