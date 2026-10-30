from pathlib import Path
import random
import json

random.seed(42)

root = Path("/home/rag/datasets/FaceForensicsPP/frames")
output = root / "splits.json"

splits = {"train": [], "validation": [], "test": []}

for label in ("real", "fake"):
    label_dir = root / label
    videos = sorted(p for p in label_dir.iterdir() if p.is_dir())
    random.shuffle(videos)

    n = len(videos)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    groups = {
        "train": videos[:train_end],
        "validation": videos[train_end:val_end],
        "test": videos[val_end:],
    }

    for split, folders in groups.items():
        for folder in folders:
            frames = sorted(folder.glob("*.jpg"))
            if frames:
                splits[split].append({
                    "video": folder.name,
                    "label": label,
                    "frames": [str(f) for f in frames],
                })

for split in splits:
    print(f"{split}: {len(splits[split])} videos")

with output.open("w") as f:
    json.dump(splits, f, indent=2)

print(f"\nSaved split file: {output}")