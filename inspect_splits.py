import json
from pathlib import Path

split_file = Path(
    "/home/rag/datasets/FaceForensicsPP/frames_advo/splits.json"
)

with open(split_file, "r") as f:
    data = json.load(f)

print("JSON data type:", type(data).__name__)

if isinstance(data, dict):
    print("\nAvailable split names:", list(data.keys()))

    for split_name, videos in data.items():
        print(f"\n{split_name}:")
        print("Data type:", type(videos).__name__)
        print("Number of entries:", len(videos))

        print("First 5 entries:")
        print(videos[:5] if isinstance(videos, list)
              else list(videos.items())[:5])

elif isinstance(data, list):
    print("\nTotal entries:", len(data))
    print("First 10 entries:", data[:10])

else:
    print("\nUnexpected JSON structure:")
    print(data)