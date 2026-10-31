import json
from pathlib import Path
from collections import Counter

original_split = Path(
    "/home/rag/datasets/FaceForensicsPP/frames/splits.json"
)

advo_root = Path(
    "/home/rag/datasets/FaceForensicsPP/frames_advo"
)

output_file = advo_root / "splits_advo.json"

# Load original splits
with open(original_split, "r") as f:
    splits = json.load(f)

new_splits = {}
all_video_ids = set()
total_frames = 0
missing_files = []

for split_name, videos in splits.items():

    new_splits[split_name] = []

    labels = Counter()

    for video in videos:

        video_id = video["video"]
        label = video["label"]

        all_video_ids.add((label, video_id))
        labels[label] += 1

        new_video = {
            "video": video_id,
            "label": label,
            "frames": []
        }

        for old_path in video["frames"]:

            old_path = Path(old_path)

            # Replace original frames directory with frames_advo
            relative_path = old_path.relative_to(
                "/home/rag/datasets/FaceForensicsPP/frames"
            )

            new_path = advo_root / relative_path

            new_video["frames"].append(str(new_path))
            total_frames += 1

            if not new_path.exists():
                missing_files.append(str(new_path))

        new_splits[split_name].append(new_video)

    print(f"\n{split_name.upper()}")
    print("Videos:", len(videos))
    print("Labels:", dict(labels))

# Check for missing images
print("\n========== VALIDATION REPORT ==========")
print("Total videos:", len(all_video_ids))
print("Total frame references:", total_frames)
print("Missing processed images:", len(missing_files))

if missing_files:
    print("\nFirst 10 missing paths:")
    for path in missing_files[:10]:
        print(path)

    raise RuntimeError("Some ADVO image paths are missing.")

# Save new split file
with open(output_file, "w") as f:
    json.dump(new_splits, f, indent=4)

print("\nNew split file saved to:")
print(output_file)

print("\nADVO split preparation completed successfully.")