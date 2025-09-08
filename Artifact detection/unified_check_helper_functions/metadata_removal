import cv2
import numpy as np

template = np.load("unified_check_helper_funcs/LOGIC_E9.npy")

def remove_metadata(original_frame, template, row_start = 67, row_end = 94, col_start = 0, col_end = 426):
    """
    Blacks out metadata in given 2D ultrasound numpy array, including:
    - "LOGIC E9", scans along a defined region for the matching text
        Optional: user can alter region bounds in the function inputs
    - Text along the top edge and right edge 
    - Intensity bar on the left edge
    
    Returns the numpy array with metadata pixels set to intensity 0.
    """
    frame = original_frame.copy()


    ### "LOGIC E9" removal
    
    # Define the region to search for text template
    search_region = frame[row_start:row_end, col_start:col_end]
    
    # Run template matching only in this cropped region
    match_scores = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, top_left_in_region = cv2.minMaxLoc(match_scores) # (col, row) of top left corner of matched region

    # Adjust coordinates to full frame
    top_left = (top_left_in_region[0] + col_start, top_left_in_region[1] + row_start) # (col, row)
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0]) # (col, row)

    # Mask the detected text
    frame[top_left[1]-3:bottom_right[1]+3, top_left[0]-3:bottom_right[0]+3] = 0 # (row, col)

    ### Border metadata and intensity bar removal
    clean_frame = np.zeros_like(frame)  # start with all black
    clean_frame[45:, 40:820] = frame[45:, 40:820]

    return clean_frame




if __name__ == "__main__":
    from pydicom import dcmread
    import time
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    
    #dicom = dcmread(r"start noncon dicoms\arm\P7FCMG80")
    dicom = dcmread(r"dicoms/linear probe dicoms/P7HC7OO6")


    pixel_array = dicom.pixel_array
    greyscale = np.mean(pixel_array, axis=-1).astype(np.uint8)  # Shape: #scans, #rows, #cols

    greyscale = np.load(r"multipass_phantom\multipass_phantom(2375, 649, 850).npy")


    # Apply metadata removal, and time the process
    clean_frames = [remove_metadata(frame, template) for frame in greyscale]
   
    # Visualization
    def visualize_frames(original_frames, cleaned_frames):
        num_frames = original_frames.shape[0]
        current_frame = [0]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(bottom=0.25)

        ax_orig, ax_clean = axes
        img_orig = ax_orig.imshow(original_frames[0], cmap='gray', vmin=0, vmax=255)
        img_clean = ax_clean.imshow(cleaned_frames[0], cmap='gray', vmin=0, vmax=255)

        ax_orig.set_title("Original")
        ax_clean.set_title("Cleaned")
        ax_orig.axis('off')
        ax_clean.axis('off')

        # Slider setup
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

        def update(val):
            idx = int(slider.val)
            img_orig.set_data(original_frames[idx])
            img_clean.set_data(cleaned_frames[idx])
            fig.canvas.draw_idle()

        slider.on_changed(update)

        def on_key(event):
            if event.key == 'right':
                current_frame[0] = min(current_frame[0] + 1, num_frames - 1)
            elif event.key == 'left':
                current_frame[0] = max(current_frame[0] - 1, 0)
            else:
                return
            slider.set_val(current_frame[0])

        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

    visualize_frames(greyscale, np.array(clean_frames))
