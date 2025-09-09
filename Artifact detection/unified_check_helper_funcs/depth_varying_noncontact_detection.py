import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

### Baseline tracker class (real-time updating)
class TopPercentileBaselineTracker:
    def __init__(self, top_percent=5, min_valid_ratio=0.2):
        self.areas = []
        self.top_percent = top_percent
        self.min_valid_ratio = min_valid_ratio

    def is_valid_frame(self, frame):
        nonzero_ratio = np.count_nonzero(frame) / frame.size
        return nonzero_ratio >= self.min_valid_ratio

    def update(self, frame):
        if not self.is_valid_frame(frame):
            return self.get_baseline()

        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self.get_baseline()

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        self.areas.append(area)

        return self.get_baseline()

    def get_baseline(self):
        if len(self.areas) == 0:
            return None

        sorted_areas = sorted(self.areas, reverse=True)
        num_top = max(1, int(len(sorted_areas) * self.top_percent / 100))
        return np.mean(sorted_areas[:num_top])

    def reset(self):
        self.areas.clear()

### Helper: Get top sector width (distance between top-left and top-right points of sector)
def get_top_sector_width(frame, row_ignore=30, col_ignore_right=20):
    # Ignore top metadata rows and right-most columns
    cropped = frame[row_ignore:, :-col_ignore_right]

    # Find all contours in the cropped image
    contours, _ = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    # Find the contour with the top-most point (i.e., lowest row index in cropped image)
    def min_row(contour):
        return np.min(contour[:, 0, 1])  # row coordinate (y)

    top_contour = min(contours, key=min_row)
    contour_points = top_contour[:, 0, :]

    # Find the minimum y (row) value in the contour (in cropped coords)
    top_row_cropped = np.min(contour_points[:, 1])
    top_row = top_row_cropped + row_ignore  # Adjust back to full frame coordinates

    # Get all points at that top row
    top_row_points = contour_points[contour_points[:, 1] == top_row_cropped]

    if len(top_row_points) < 2:
        return None, None, None

    # Determine top-left and top-right based on x (in cropped image coords)
    top_left_x = np.min(top_row_points[:, 0])
    top_right_x = np.max(top_row_points[:, 0])

    # Adjust x coordinates back to full image (no need here since we cropped columns from right)
    top_left = (top_left_x, top_row)
    top_right = (top_right_x, top_row)
    top_width = top_right_x - top_left_x

    return top_width, top_left, top_right




### Real-time noncontact detection with depth change handling
def detect_noncontact_realtime(
    dicom_array,
    threshold_factor=0.85,
    top_percent=5,
    min_valid_ratio=0.2,
    width_change_thresh=10
):
    num_frames = dicom_array.shape[0]
    noncontact_flags = np.full((num_frames,), False)
    corner_points = [None] * num_frames
    baseline_tracker = TopPercentileBaselineTracker(top_percent=top_percent, min_valid_ratio=min_valid_ratio)

    prev_top_width = None

    for i in range(num_frames):
        frame = dicom_array[i]

        # Get top width and corners
        top_width, top_left, top_right = get_top_sector_width(frame)
        if top_width is None:
            noncontact_flags[i] = True
            continue

        corner_points[i] = (top_left, top_right)

        # Detect significant depth change
        if prev_top_width is not None and abs(top_width - prev_top_width) > width_change_thresh:
            print(f"Depth change detected at frame {i}: width changed from {prev_top_width} to {top_width}")
            baseline_tracker.reset()

        prev_top_width = top_width

        # Update baseline with current frame
        baseline = baseline_tracker.update(frame)
        if baseline is None:
            noncontact_flags[i] = True
            continue

        # Compute current frame area
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            noncontact_flags[i] = True
            continue

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Compare to baseline
        if area < threshold_factor * baseline:
            noncontact_flags[i] = True
        else:
            noncontact_flags[i] = False

    return noncontact_flags, corner_points


### Visualization
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize(dicom_array, noncontact_flags, corner_points):
    num_frames = dicom_array.shape[0]
    current_frame_idx = [0]
    suppress_slider_callback = [False]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    img_disp = ax.imshow(dicom_array[0], cmap='gray')
    text_disp = ax.text(0.5, -0.1, '', transform=ax.transAxes,
                        ha='center', va='top', fontsize=12, weight='bold')

    # Initialize red corner markers (top-left and top-right)
    corner_scatter = ax.plot([], [], 'ro')[0]

    # Slider setup
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    frame_slider = Slider(ax_slider, 'Frame', 0, num_frames - 1,
                          valinit=0, valfmt='%0.0f', valstep=1)

    def update_display(frame_idx):
        frame_idx = max(0, min(num_frames - 1, frame_idx))
        current_frame_idx[0] = frame_idx

        # Update image
        img_disp.set_data(dicom_array[frame_idx])

        # Update contact status text
        if noncontact_flags[frame_idx]:
            text_disp.set_text("Noncontact detected")
            text_disp.set_color('red')
        else:
            text_disp.set_text("No noncontact")
            text_disp.set_color('black')

        # Update red corner markers
        corners = corner_points[frame_idx]
        if corners:
            top_left, top_right = corners
            xs = [top_left[0], top_right[0]]
            ys = [top_left[1], top_right[1]]
            corner_scatter.set_data(xs, ys)
        else:
            corner_scatter.set_data([], [])

        # Sync slider (safely)
        suppress_slider_callback[0] = True
        frame_slider.set_val(frame_idx)
        suppress_slider_callback[0] = False

        fig.canvas.draw_idle()

    def on_slider_change(val):
        if not suppress_slider_callback[0]:
            update_display(int(val))

    def on_key(event):
        if event.key == 'right':
            update_display(current_frame_idx[0] + 1)
        elif event.key == 'left':
            update_display(current_frame_idx[0] - 1)

    frame_slider.on_changed(on_slider_change)
    fig.canvas.mpl_connect('key_press_event', on_key)

    update_display(0)
    plt.show()


### Main execution
if __name__ == "__main__":
    from pydicom import dcmread
    from unified_check_helper_funcs.load_dicom import get_dicom_array
    
    dicom = dcmread("linear probe dicoms/P7HC7OO6")
    dicom_array = get_dicom_array(dicom)

    noncontact_flags, corner_points = detect_noncontact_realtime(
        dicom_array,
        threshold_factor=0.75,
        top_percent=2,
        min_valid_ratio=0.15,
        width_change_thresh=10  # Adjust as needed
    )

    visualize(dicom_array, noncontact_flags, corner_points)

