import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load poses
scan4_poses = np.load(r"scan4_givens\scan4_poses_centered(3, 4, 2375).npy")
scan1_poses = np.load(r"multipass_givens\multipass_poses_centered(3, 4, 2375).npy")
poses = np.concatenate((scan4_poses, scan1_poses), axis=2)

translations = poses[:, 3, :]
x = translations[0]

# Detect major peaks and valleys
peaks, _ = find_peaks(x, prominence=0.001, distance=250)  
valleys, _ = find_peaks(-x, prominence=0.001, distance=250)

# Combine and sort
turning_points = np.sort(np.concatenate([peaks, valleys]))

# Pass start/end indices
pass_boundaries = [0] + turning_points.tolist() + [len(x)-1]
all_passes = [(pass_boundaries[i], pass_boundaries[i+1]) for i in range(len(pass_boundaries)-1)]

# Filter passes by minimum number of frames
min_frames = 150  # adjust this threshold
passes = [(start, end) for start, end in all_passes if (end - start) >= min_frames]

print("Detected passes (start_idx, end_idx) with at least", min_frames, "frames:")
for p in passes:
    print(p)
print(f"Detected {len(passes)} total passes.")
print(passes)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, label="X translation", color="r", alpha=0.5)
plt.plot(peaks, x[peaks], "go", label="Maxima")
plt.plot(valleys, x[valleys], "bo", label="Minima")

for cp in turning_points:
    plt.axvline(cp, color="k", linestyle="--", alpha=0.5)

for start, end in passes:
    plt.axvspan(start, end, color="yellow", alpha=0.2)

plt.xlabel("Frame index")
plt.ylabel("X translation")
plt.title("Major pass detection from X turning points")
plt.legend()
plt.grid(True)
plt.show()
