import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load grayscale frames (as 2D arrays)
frame1 = cv2.imread('frame325.png', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('frame326.png', cv2.IMREAD_GRAYSCALE)

# Ensure same shape
assert frame1.shape == frame2.shape, "Frames must have same dimensions"

# Compute absolute pixel differences
diff = np.abs(frame1.astype(np.int16) - frame2.astype(np.int16))

# Optional: Set low differences below threshold to 0 to ignore noise
threshold = 0
diff_filtered = np.where(diff >= threshold, diff, 0)

# Display heatmap
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Frame 1")
plt.imshow(frame1, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Frame 2")
plt.imshow(frame2, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f"Pixel Difference (≥{threshold})")
plt.imshow(diff_filtered, cmap='hot')
plt.colorbar(label='Intensity Difference')
plt.axis('off')

plt.tight_layout()
plt.show()

