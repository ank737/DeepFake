import cv2
import numpy as np
from pathlib import Path

# Input image
image_path = Path(
    "/home/rag/datasets/FaceForensicsPP/frames/real/585/frame_000159.jpg"
)

# Output directory
output_dir = Path.home() / "Code" / "advo_results"
output_dir.mkdir(parents=True, exist_ok=True)

# Load image
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Cannot read image: {image_path}")

# Step 1: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Normalize pixel values to [0, 1]
x = gray.astype(np.float32) / 255.0

# Save grayscale image for inspection
gray_output = output_dir / "grayscale.jpg"
cv2.imwrite(str(gray_output), gray)

print("Image loaded successfully.")
print("Original dimensions:", image.shape)
print("Grayscale dimensions:", gray.shape)
print("Normalized minimum:", x.min())
print("Normalized maximum:", x.max())
print("Grayscale image saved at:", gray_output)

# Step 3: Calculate image gradients
uy, ux = np.gradient(x)

# Calculate gradient magnitude
magnitude = np.sqrt(ux**2 + uy**2)

print("\nGradient calculation completed.")
print("Gradient magnitude minimum:", magnitude.min())
print("Gradient magnitude maximum:", magnitude.max())

# Save gradient magnitude for visualization
gradient_image = np.clip(
    magnitude / (magnitude.max() + 1e-8) * 255,
    0,
    255
).astype(np.uint8)

gradient_path = output_dir / "gradient.jpg"
cv2.imwrite(str(gradient_path), gradient_image)

print("Gradient image saved at:", gradient_path)

# Step 4: Calculate the diffusion coefficient
# Based on Equation (2) of the research paper

c = 0.1  # Provisional value; the paper does not specify a numerical value here.

g = 1.0 / (1.0 + (magnitude**2 / c**2))

print("\nDiffusion coefficient calculation completed.")
print("Coefficient minimum:", g.min())
print("Coefficient maximum:", g.max())

# Save coefficient image for visualization
coefficient_image = np.clip(
    g * 255,
    0,
    255
).astype(np.uint8)

coefficient_path = output_dir / "diffusion_coefficient.jpg"
cv2.imwrite(str(coefficient_path), coefficient_image)

print("Diffusion coefficient image saved at:", coefficient_path)

# Step 5: Experiment with different beta values
# Experimental parameters
c = 0.1
lam = 0.02
dt = 0.001
iterations = 20

beta_values = [0.25, 0.50, 0.75]

from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

# Keep the original normalized image unchanged.
original = gray.astype(np.float32) / 255.0

results = []

for beta in beta_values:

    print(f"\nRunning experiment with beta = {beta}")

    # Start every experiment from the same original image.
    x = original.copy()

    for i in range(iterations):

        # Calculate image gradients
        uy, ux = np.gradient(x)

        # Calculate gradient magnitude
        magnitude = np.sqrt(ux**2 + uy**2 + 1e-8)

        # Calculate normalized gradients
        nx = ux / magnitude
        ny = uy / magnitude

        # Curvature term
        curvature = (
            np.gradient(nx, axis=1)
            + np.gradient(ny, axis=0)
        )

        # Diffusion coefficient
        g = 1.0 / (1.0 + (magnitude**2 / c**2))

        # Anisotropic diffusion term
        diffusion = (
            np.gradient(g * ux, axis=1)
            + np.gradient(g * uy, axis=0)
        )

        # Update image using Equation (35)
        x = x + dt * (
            lam * curvature
            + (1 - beta) * diffusion
        )

        # Numerical safeguard
        x = np.clip(x, 0.0, 1.0)

    # Convert filtered image to uint8
    filtered_image = np.clip(
        x * 255, 0, 255
    ).astype(np.uint8)

    # Save each experiment separately
    filtered_path = output_dir / f"filtered_beta_{beta:.2f}.jpg"
    cv2.imwrite(str(filtered_path), filtered_image)

    # Calculate image comparison metrics
    mse = mean_squared_error(original, x)

    psnr = peak_signal_noise_ratio(
        original,
        x,
        data_range=1.0
    )

    ssim = structural_similarity(
        original,
        x,
        data_range=1.0
    )

    difference = np.abs(original - x)

    average_change = difference.mean()

    results.append({
        "beta": beta,
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim,
        "Average_change": average_change,
        "Saved_image": str(filtered_path)
    })

    print(f"Completed beta = {beta}")
    print("MSE:", mse)
    print("PSNR:", psnr)
    print("SSIM:", ssim)
    print("Average absolute change:", average_change)
    print("Saved at:", filtered_path)


# Step 6: Display final comparison

print("\n========== BETA EXPERIMENT RESULTS ==========")

for result in results:

    print(f"\nBeta: {result['beta']}")
    print("MSE:", result["MSE"])
    print("PSNR:", result["PSNR"])
    print("SSIM:", result["SSIM"])
    print("Average change:", result["Average_change"])
    print("Image:", result["Saved_image"])

print("\nAll experiments completed.")