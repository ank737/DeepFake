import cv2
from pathlib import Path

fake_path = Path(
    "/home/rag/datasets/FaceForensicsPP/manipulated_sequences/Deepfakes/c23/videos"
)
real_path = Path(
    "/home/rag/datasets/FaceForensicsPP/original_sequences/youtube/c23/videos"
)

output = Path("/home/rag/datasets/FaceForensicsPP/frames")
frames_per_video = 10

for label, folder in [("fake", fake_path), ("real", real_path)]:
    videos = sorted(folder.glob("*.mp4"))
    print(f"\nProcessing {label}: {len(videos)} videos")

    for index, video in enumerate(videos, start=1):
        cap = cv2.VideoCapture(str(video))

        if not cap.isOpened():
            print(f"Cannot open: {video.name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            continue

        save_dir = output / label / video.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        positions = [
            round(i * (total_frames - 1) / (frames_per_video - 1))
            for i in range(frames_per_video)
        ]

        saved = 0

        for frame_number in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = cap.read()

            if success:
                filename = save_dir / f"frame_{frame_number:06d}.jpg"
                if cv2.imwrite(str(filename), frame):
                    saved += 1

        cap.release()

        if index % 50 == 0 or index == len(videos):
            print(f"Progress: {index}/{len(videos)} videos")

print("\nFrame extraction completed!")
print(f"Output: {output}")