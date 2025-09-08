import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


### Blur detection methods (masked to non-black region)

def detect_blur_tenengrad(frame, threshold):
    gx = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
    grad_mag = gx**2 + gy**2
    score = np.mean(grad_mag[frame > 0])
    return score < threshold, score

def detect_blur_wavelet(frame, threshold):
    coeffs = pywt.dwt2(frame, 'haar')
    _, (cH, cV, cD) = coeffs
    mask = frame[::2, ::2] > 0  # approximate mask for wavelet subbands
    energy = np.sum((cH**2 + cV**2 + cD**2)[mask])
    score = energy / np.count_nonzero(mask)
    return score < threshold, score


### Main blur detection loop

def detect_blur_in_frame(cleaned_frame, tenengrad_threshold = 11000, wavelet_threshold = 300):
       
    blurred1, score1 = detect_blur_tenengrad(cleaned_frame, threshold = tenengrad_threshold)
    blurred2, score2 = detect_blur_wavelet(cleaned_frame, threshold = wavelet_threshold)

    blur_flag = blurred1 and blurred2
    scores = (score1, score2)

    return blur_flag, scores



### Visualization (frame display + method scores + blur status)
def visualize(dicom_file, tenengrad_threshold=11000, wavelet_threshold=300):
    from pydicom import dcmread
    from unified_check_helper_funcs.load_dicom import get_dicom_array

    dicom_array = get_dicom_array(dcmread(dicom_file))

    num_frames = dicom_array.shape[0]
    current_frame_idx = [0]
    suppress_slider_callback = [False]

    fig, ax_img = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.3)

    img_disp = ax_img.imshow(dicom_array[0], cmap='gray')
    ax_img.axis('off')

    score_text = ax_img.text(0.5, -0.08, '', transform=ax_img.transAxes,
                             fontsize=12, ha='center', va='top', color='black', family='monospace')

    flag_text = ax_img.text(0.5, -0.14, '', transform=ax_img.transAxes,
                            fontsize=14, ha='center', va='top', weight='bold')

    ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
    frame_slider = Slider(ax_slider, 'Frame', 0, num_frames - 1,
                          valinit=0, valfmt='%0.0f', valstep=1)

    def update_display(frame_idx):
        frame_idx = max(0, min(num_frames - 1, frame_idx))
        current_frame_idx[0] = frame_idx

        frame = dicom_array[frame_idx].astype(np.uint8)
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img_disp.set_data(frame)

        blur_flag, (score1, score2) = detect_blur_in_frame(
            frame, tenengrad_threshold=tenengrad_threshold, wavelet_threshold=wavelet_threshold
        )
        score_text.set_text(f"Tenengrad: {score1:.1f}   |   Wavelet: {score2:.1f}")

        if blur_flag:
            flag_text.set_text("Blur Detected")
            flag_text.set_color('red')
        else:
            flag_text.set_text("No Blur")
            flag_text.set_color('black')

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



### Main

if __name__ == "__main__":

    
    # (Noncontact scans contain some blur before/after noncontact)

    ### Noncontact, abdominal transverse
    #dicom_file = "noncontact dicoms/P7EE0E88"

    ### Noncontact, arm
    #dicom_file =  "noncontact dicoms/P7EE0E84"
    #dicom_file =  "noncontact dicoms/P7EE0E86"

    ### Noncontact, kidney sagittal-coronal
    #dicom_file =  "noncontact dicoms/P7EAN7G0"
    #dicom_file =  "noncontact dicoms/P7EARBO2" ###
    #dicom_file =  "noncontact dicoms/P7EARBO6" ###

    
    ### Blur, transverse abdominal
    #dicom_file =  "blur dicoms/P7EE0E8A" #doesn't have much blur
    #dicom_file =  "blur dicoms/P7EE0E80" ###
    #dicom_file =  "blur dicoms/P7EE0E82" ###

    ### Blur, kidney sagittal/coronal (collected earlier)
    dicom_file =  "blur dicoms/P77CHQOI" ###

  
    visualize(dicom_file, tenengrad_threshold=10500, wavelet_threshold=325)
