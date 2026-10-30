import cv2
import numpy as np
from pathlib import Path

image_path = Path(
    "/home/rag/datasets/FaceForensicsPP/frames/real/585/frame_000159.jpg"
)
output_path = Path.home() / "Code" / "advo_test_color.jpg"

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Cannot read {image_path}")

# Lab color space: process brightness while retaining color information.
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
x = lab[:, :, 0].astype(np.float32) / 255.0

# Conservative provisional parameters.
beta = 0.5
lam = 0.02
c = 0.1
dt = 0.02
iterations = 5

for _ in range(iterations):
    uy, ux = np.gradient(x)
    magnitude = np.sqrt(ux**2 + uy**2 + 1e-8)

    nx = ux / magnitude
    ny = uy / magnitude
    curvature = (
        np.gradient(nx, axis=1)
        + np.gradient(ny, axis=0)
    )

    g = 1.0 / (1.0 + (magnitude / c) ** 2)
    diffusion = (
        np.gradient(g * ux, axis=1)
        + np.gradient(g * uy, axis=0)
    )

    x = np.clip(
        x + dt * (lam * curvature + (1 - beta) * diffusion),
        0.0,
        1.0,
    )

lab[:, :, 0] = np.clip(x * 255, 0, 255).astype(np.uint8)
result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

cv2.imwrite(str(output_path), result)
print(f"Saved color-preserving result: {output_path}")