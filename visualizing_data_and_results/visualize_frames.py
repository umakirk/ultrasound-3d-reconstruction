import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from filter_segmentations import filter_small_segments

# Load frames to visualize
array = np.load(r"multipass_givens\multipass_phantom(2375, 649, 850).npy")[1183:1382]

# Verify dimensions
num_frames = array.shape[0]

# Create figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # Space for the slider

# Display the first frame
frame_display = ax.imshow(array[0], cmap='gray', vmin=np.min(array), vmax=np.max(array))
ax.set_title("Frame 0")
ax.axis('off')

# Slider axis: [left, bottom, width, height]
slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
frame_slider = Slider(slider_ax, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

# Update function
def update(val):
    frame = int(frame_slider.val)
    frame_display.set_data(array[frame])
    ax.set_title(f"Frame {frame}")
    fig.canvas.draw_idle()

frame_slider.on_changed(update)

plt.show()
