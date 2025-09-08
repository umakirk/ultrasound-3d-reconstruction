import numpy as np
from pydicom import dcmread

# Importing my helper functions
from unified_check_helper_funcs.load_dicom import get_dicom_array
from unified_check_helper_funcs.metadata_removal import remove_metadata, template
from unified_check_helper_funcs.shadow_detection import (
    find_sector_corners, generate_rays,
    ray_intensity_profiles, shadow_segments, ray_shadow_mask,
    filter_standalone_shadow_rays, smooth_and_fill_shadow_mask, extract_largest_shadows, filter_artifact_segments_by_length, two_threshold_tail_segments)
from unified_check_helper_funcs.blur_detection_per_frame import detect_blur_in_frame
from unified_check_helper_funcs.depth_varying_noncontact_detection import detect_noncontact_realtime

from add_artifacts import blur_frames



def run_unified_detection(dicom_array, noncontact_detect = True, blur_detect = True, dark_shadow_detect = True, 
                          medium_shadow_detect = True, PAE_detect = True, phantom_PAE_detect = True,
                          shadow_params=None, blur_params=None, noncontact_params=None):
    
    if noncontact_detect and noncontact_params is None:
        noncontact_params = dict(threshold_factor=0.75, top_percent=2, min_valid_ratio=0.15, width_change_thresh=10)

    if blur_detect and blur_params is None:
            blur_params = dict(tenengrad_threshold=11000, wavelet_threshold=300)
    
    ### Setting default parameters if none are given
    if shadow_params is None:
        shadow_params = {
            "dark": dict(shadow_threshold=20, min_shadow_length=10, shadow_min_threshold=0),
            "medium": dict(shadow_threshold=60, min_shadow_length=33, shadow_min_threshold=30),
            "PAE": dict(shadow_threshold=255, min_shadow_length=10, shadow_min_threshold=200),
            "phantom_PAE" : dict(presence_thresh=185, tail_thresh=70, max_artifact_length=45)
        }
    
   
    ### Create list of ray detections that are toggled on, to loop through easily
    simple_ray_detections = []
    if dark_shadow_detect:
        simple_ray_detections.append("dark")
    if medium_shadow_detect:
        simple_ray_detections.append("medium")
    if PAE_detect:
        simple_ray_detections.append("PAE")


    ### Read in DICOM and identify transducer type
    #dicom = dcmread(dicom_file)
    #transducer = dicom.TransducerType

    ### Run noncontact detection
    #frames = get_dicom_array(dicom)

    frames = dicom_array

    num_frames = frames.shape[0]
    if noncontact_detect:
        noncontact_flags, _ = detect_noncontact_realtime(frames, **noncontact_params)
    #blur_flags, _ = detect_blur_in_frames(frames, **blur_params)

    overlays = []
    statuses = []

    for i in range(num_frames):
        
        # Remove metadata from frame
        uncleaned_frame = frames[i]
        frame = remove_metadata(uncleaned_frame, template)
        #avg_intensity = np.average(frame[frame != 0])
        #print(avg_intensity)

        # Create empty rgba array for visualization later
        rgba = np.stack([frame]*3 + [np.full_like(frame, 255)], axis=-1)


        # Noncontact and blur detection
        status = ""
        if noncontact_detect and noncontact_flags[i]:
            status = "Rejected: Noncontact"
        
        elif blur_detect:
            blur_flag, scores = detect_blur_in_frame(uncleaned_frame, **blur_params)
            #print(f"Frame {i}: Tenengrad={scores[0]:.2f}, Wavelet={scores[1]:.2f}, Flagged={blur_flag}")

            if blur_flag == True:
                status = "Rejected: Blur"

        # Shadow (dark and medium) and posterior acoustic enhancement detection
        if status == "" and (simple_ray_detections or phantom_PAE_detect):

            #sector_mask = sector(frame)
            
            corners = find_sector_corners(frame, transducer = "CURVED LINEAR")
            rays = generate_rays(corners, num_rays=200, points_per_ray=100, transducer = "CURVED LINEAR")
            profiles = ray_intensity_profiles(frame, rays)

            masks = {}
            for ray_detection in simple_ray_detections:
                params = shadow_params[ray_detection]
                segs = shadow_segments(profiles, **params)
                raw_mask = ray_shadow_mask(rays, segs, frame.shape)
                filtered = filter_standalone_shadow_rays(raw_mask, rays, segs, neighbor_range=1)
                filled = smooth_and_fill_shadow_mask(filtered, dilation_iter=4, closing_iter=3)
                final_mask = extract_largest_shadows(filled, min_area=500)
                masks[ray_detection] = final_mask

            if phantom_PAE_detect:
                segs = two_threshold_tail_segments(
                    profiles,
                    shadow_params["phantom_PAE"]["presence_thresh"],
                    shadow_params["phantom_PAE"]["tail_thresh"]
                )
                segs = filter_artifact_segments_by_length(segs, shadow_params["phantom_PAE"]["max_artifact_length"])

                tail_mask = ray_shadow_mask(rays, segs, frame.shape)
                tail_mask = smooth_and_fill_shadow_mask(tail_mask, dilation_iter=5, closing_iter=3)
                masks["phantom_PAE"] = tail_mask


            # Build colored overlay
            #rgba = np.stack([frame]*3 + [np.full_like(frame, 255)], axis=-1)

            rgba = np.stack([uncleaned_frame]*3 + [np.full_like(uncleaned_frame, 255)], axis=-1)
            mask_colors = {
                "dark": (200, 0, 255),       # Red
                "medium": (0, 0, 255),     # Blue
                "PAE": (255, 0, 200), # Green
                "phantom_PAE": (255, 105, 180)  # Pink
            }
            for ray_detection, mask in masks.items():
                color = mask_colors[ray_detection]
                mask_bool = mask.astype(bool)
                for c in range(3):
                    alpha = 0.35
                    rgba[mask_bool, c] = np.clip((1 - alpha) * rgba[mask_bool, c] + alpha * color[c], 0, 255)

        overlays.append(rgba)
        statuses.append(status)

    overlays = np.stack(overlays)
    return overlays, statuses


# Visualization
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
def visualize_unified(dicom_array, overlays, statuses):

    #dicom = dcmread(dicom_file)

    #frames = get_dicom_array(dicom)

    frames = dicom_array

    num_frames = frames.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)

    img1 = ax1.imshow(frames[0], cmap='gray')
    ax1.set_title("Original")
    ax1.axis('off')

    img2 = ax2.imshow(overlays[0])
    ax2.set_title("Overlay")
    ax2.axis('off')

    status_text = ax2.text(0.5, -0.08, '', transform=ax2.transAxes,
                           fontsize=12, ha='center', va='top', color='black', family='monospace')

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

    def update(val):
        i = int(slider.val)
        img1.set_data(frames[i])
        img2.set_data(overlays[i])
        status_text.set_text(statuses[i])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":   

    ### Setting all artifact detections toggled on
    noncontact_detect = True
    blur_detect = True
    dark_shadow_detect = True
    medium_shadow_detect = True
    PAE_detect = True
    phantom_PAE_detect = True
    
    ### Setting default parameters
    shadow_params = {
        "dark": dict(shadow_threshold=20, min_shadow_length=10, shadow_min_threshold=0),
        "medium": dict(shadow_threshold=60, min_shadow_length=33, shadow_min_threshold=30),
        "PAE": dict(shadow_threshold=255, min_shadow_length=10, shadow_min_threshold=200),
        "phantom_PAE" : dict(presence_thresh=185, tail_thresh=70, max_artifact_length=45)
    }
    blur_params = dict(tenengrad_threshold=11000, wavelet_threshold=300)
    noncontact_params = dict(threshold_factor=0.75, top_percent=2, min_valid_ratio=0.15, width_change_thresh=10)


    ### LL1 testing
    # dicom_file = r"mini proj 3 (recon) files\recon LL1\LL1" # changed metadata removal to crop out more of the top rows, might have to change back for other input scans
    # dicom_array = get_dicom_array(dcmread(dicom_file))
    # shadow_params["dark"] = dict(shadow_threshold=30, min_shadow_length=10, shadow_min_threshold=0)
    # shadow_params["PAE"] = dict(shadow_threshold=255, min_shadow_length=5, shadow_min_threshold=150)

    ### Multipass phantom testing
    # dicom_array = np.load(r"multipass_phantom\multipass_phantom(2375, 649, 850).npy")
    # noncontact_detect = False
    # dark_shadow_detect = False
    # medium_shadow_detect = False
    # PAE_detect = False

  
    ### My kidney scan testing
    dicom_array = get_dicom_array(dcmread(r"dicoms\shadow dicoms\uma kidney\P77CHQOI"))
    noncontact_detect = False
    blur_detect = False
    medium_shadow_detect = False
    phantom_PAE_detect = False
    shadow_params = {
        "dark": dict(shadow_threshold=55, min_shadow_length=30, shadow_min_threshold=0),
        "medium": dict(shadow_threshold=100, min_shadow_length=45, shadow_min_threshold=10),
        "PAE": dict(shadow_threshold=255, min_shadow_length=10, shadow_min_threshold=140),
        "phantom_PAE" : dict(presence_thresh=185, tail_thresh=70, max_artifact_length=45)
    }



    ### ADDING ARTIFICIAL ARTIFACTS (optional)
    #dicom_array = blur_dicom_array(dicom_array, range(180, 210), sigma_value = 5)


    ### Running and visualizing unified detection
    overlays, statuses = run_unified_detection(dicom_array, noncontact_detect = noncontact_detect, blur_detect = blur_detect, dark_shadow_detect = dark_shadow_detect, 
                          medium_shadow_detect = medium_shadow_detect, PAE_detect = PAE_detect, phantom_PAE_detect = phantom_PAE_detect, shadow_params = shadow_params, blur_params=blur_params, noncontact_params=noncontact_params)
    visualize_unified(dicom_array, overlays, statuses)
