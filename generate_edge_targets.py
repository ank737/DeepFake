import cv2
from pathlib import Path

# ---------------------------------------
# 1. Configuration
# ---------------------------------------

INPUT_ROOT = Path(
    "/home/rag/datasets/FaceForensicsPP/frames_advo"
)

OUTPUT_ROOT = Path(
    "/home/rag/datasets/FaceForensicsPP/edge_targets"
)

IMAGE_SIZE = 224

CANNY_LOW = 100
CANNY_HIGH = 200

EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------
# 2. Generate edge targets
# ---------------------------------------

total = 0
successful = 0
failed = 0

for category in ["real", "fake"]:

    input_category = INPUT_ROOT / category
    output_category = OUTPUT_ROOT / category

    images = [
        p for p in input_category.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    print(f"\nProcessing: {category}")
    print("Images found:", len(images))

    for index, image_path in enumerate(images, start=1):

        total += 1

        try:

            # Read grayscale image
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                raise ValueError("Image could not be read")

            # Resize to U-Net input dimensions
            image = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            # Generate Canny edge map
            edges = cv2.Canny(
                image,
                CANNY_LOW,
                CANNY_HIGH
            )

            # Preserve directory structure
            relative_path = image_path.relative_to(INPUT_ROOT)

            output_path = OUTPUT_ROOT / relative_path

            # Save target as PNG
            output_path = output_path.with_suffix(".png")

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            if not cv2.imwrite(str(output_path), edges):
                raise IOError("Could not save edge target")

            successful += 1

        except Exception as error:

            failed += 1
            print("Failed:", image_path)
            print("Reason:", error)

        if index % 500 == 0 or index == len(images):
            print(
                f"{category}: {index}/{len(images)} | "
                f"Successful: {successful} | Failed: {failed}"
            )


# ---------------------------------------
# 3. Final report
# ---------------------------------------

print("\n========== EDGE TARGET REPORT ==========")

print("Total images:", total)
print("Successfully generated:", successful)
print("Failed:", failed)

print("Output directory:", OUTPUT_ROOT)

print("\nEdge target generation completed.")