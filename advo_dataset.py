import cv2
import numpy as np
import shutil
from pathlib import Path

# ==========================================
# 1. Dataset paths
# ==========================================

input_dir = Path(
    "/home/rag/datasets/FaceForensicsPP/frames"
)

output_dir = Path(
    "/home/rag/datasets/FaceForensicsPP/frames_advo"
)

# ==========================================
# 2. ADVO experimental parameters
# ==========================================

c = 0.1
lam = 0.02
beta = 0.5
dt = 0.001
iterations = 20

# Supported image formats
image_extensions = {".jpg", ".jpeg", ".png"}

# ==========================================
# 3. Validate input dataset
# ==========================================

if not input_dir.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {input_dir}"
    )

for category in ["real", "fake"]:
    if not (input_dir / category).is_dir():
        raise FileNotFoundError(
            f"Missing dataset category: {category}"
        )

output_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# 4. ADVO filtering function
# ==========================================

def apply_advo(image):

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Normalize to [0, 1]
    x = gray.astype(np.float32) / 255.0

    # ADVO iterations
    for _ in range(iterations):

        uy, ux = np.gradient(x)

        magnitude = np.sqrt(
            ux**2 + uy**2 + 1e-8
        )

        # Normalized gradients
        nx = ux / magnitude
        ny = uy / magnitude

        # Curvature term
        curvature = (
            np.gradient(nx, axis=1)
            + np.gradient(ny, axis=0)
        )

        # Diffusion coefficient
        g = 1.0 / (
            1.0 + magnitude**2 / c**2
        )

        # Anisotropic diffusion
        diffusion = (
            np.gradient(g * ux, axis=1)
            + np.gradient(g * uy, axis=0)
        )

        # Equation (35)
        x = x + dt * (
            lam * curvature
            + (1 - beta) * diffusion
        )

        # Numerical safeguard
        x = np.clip(x, 0.0, 1.0)

    # Convert back to 8-bit grayscale
    filtered = np.clip(
        x * 255.0,
        0,
        255
    ).astype(np.uint8)

    return filtered


# ==========================================
# 5. Process dataset
# ==========================================

total_images = 0
successful_images = 0
failed_images = 0

for category in ["real", "fake"]:

    category_dir = input_dir / category

    image_paths = [
        p for p in category_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in image_extensions
    ]

    print(f"\nProcessing category: {category}")
    print(f"Images found: {len(image_paths)}")

    for index, image_path in enumerate(image_paths, start=1):

        total_images += 1

        # Preserve the relative folder structure
        relative_path = image_path.relative_to(input_dir)

        output_path = output_dir / relative_path

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            image = cv2.imread(str(image_path))

            if image is None:
                raise ValueError("Image could not be read")

            filtered = apply_advo(image)

            success = cv2.imwrite(
                str(output_path),
                filtered
            )

            if not success:
                raise IOError("Could not save filtered image")

            successful_images += 1

        except Exception as error:

            failed_images += 1

            print(
                f"\nFailed: {image_path}"
            )
            print(f"Reason: {error}")

        if index % 100 == 0 or index == len(image_paths):

            print(
                f"{category}: {index}/{len(image_paths)} "
                f"| Successful: {successful_images} "
                f"| Failed: {failed_images}"
            )


# ==========================================
# 6. Copy dataset split information
# ==========================================

split_file = input_dir / "splits.json"

if split_file.exists():

    shutil.copy2(
        split_file,
        output_dir / "splits.json"
    )

    print("\nDataset split file copied.")

else:

    print("\nWARNING: splits.json was not found.")


# ==========================================
# 7. Final report
# ==========================================

print("\n========== ADVO DATASET PROCESSING REPORT ==========")

print("Input directory:", input_dir)
print("Output directory:", output_dir)

print("Total images:", total_images)
print("Successfully processed:", successful_images)
print("Failed images:", failed_images)

if total_images > 0:

    print(
        "Success percentage:",
        round(successful_images / total_images * 100, 2),
        "%"
    )

print("\nDataset preprocessing completed.")