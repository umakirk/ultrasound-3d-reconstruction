import cv2
import numpy as np
from PIL import Image

### Generating radial rays

# def sector(frame_image):
  
#     # find contours to detect the sector
#     contours, _ = cv2.findContours(frame_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     largest_contour = max(contours, key=cv2.contourArea)

#     # create mask from the largest contour
#     sector_mask = np.zeros_like(frame_image)
#     cv2.drawContours(sector_mask, [largest_contour], -1, 255, -1)  # draw filled contour

#     return sector_mask

def find_sector_corners(cleaned_frame, transducer):
    rows, cols = np.where(cleaned_frame != 0)

    if transducer == "CURVED LINEAR":

        # bottom corners
        min_col = cols.min()
        max_col = cols.max()
        
        # find bottom-most row of the left-most and right-most cols
        bottom_left_row = rows[cols == min_col].max()
        bottom_right_row = rows[cols == max_col].max()

        bottom_left = (bottom_left_row, min_col)
        bottom_right = (bottom_right_row, max_col)

        # middle column (average of bottom left and right x-coords)
        middle_col = (min_col + max_col) // 2

        # min row of entire sector region
        min_row = rows.min()

        # find the rightmost white pixel in the top row
        cols_in_top_row = cols[rows == min_row]
        max_col_top = cols_in_top_row.max()

        # offset from middle column
        offset = abs(max_col_top - middle_col)

        # top corners: both have y-coord of min_row, x-coords are middle_col +- offset
        top_left = (min_row, middle_col - offset)
        top_right = (min_row, middle_col + offset)
    
    elif transducer == "LINEAR":

        min_row = rows.min()
        max_row = rows.max()

        min_col = cols.min()
        max_col = cols.max()

        # (row, col)
        top_left = (min_row, min_col)
        top_right = (min_row, max_col)
        bottom_left = (max_row, min_col)
        bottom_right = (max_row, max_col)
        

    return {'top_left': top_left,
            'top_right': top_right,
            'bottom_left': bottom_left,
            'bottom_right': bottom_right}

def line_intersection(p1, p2, p3, p4):
    """
    Find intersection line from p1 to p2 and line from p3 to p4
    Each p is (row, col) or (y, x)
    Returns intersection point as (row, col)
    """
    x1, y1 = p1[1], p1[0]
    x2, y2 = p2[1], p2[0]
    x3, y3 = p3[1], p3[0]
    x4, y4 = p4[1], p4[0]

    fraction_along_line1  = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / ((y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1))

    x = x1 + fraction_along_line1 *(x2 - x1)
    y = y1 + fraction_along_line1 *(y2 - y1)

    return (y, x)

def generate_rays(corners, num_rays = 200, points_per_ray = 100, transducer = 'CURVED LINEAR'):
    
    if transducer == 'CURVED LINEAR':
        
        tl = corners['top_left']
        tr = corners['top_right']
        bl = corners['bottom_left']
        br = corners['bottom_right']

        # find origin -- point where ultrasound waves were emitted from
        origin = line_intersection(bl, tl, br, tr)
        oy, ox = origin

        # radii of curvature for top and bottom curves
        top_radius = np.linalg.norm(np.array(tl) - np.array(origin))
        bottom_radius = np.linalg.norm(np.array(bl) - np.array(origin))

        # angles of corners relative to origin
        def angle_from_origin(point):
            y, x = point
            return np.arctan2(y - oy, x - ox)

        angle_tl = angle_from_origin(tl)
        angle_tr = angle_from_origin(tr)
        angle_bl = angle_from_origin(bl)
        angle_br = angle_from_origin(br)

        # Make sure angles are in increasing order for interpolation
        angles_top = np.unwrap([angle_tl, angle_tr])
        angle_start_top, angle_end_top = angles_top.min(), angles_top.max()

        angles_bottom = np.unwrap([angle_bl, angle_br])
        angle_start_bottom, angle_end_bottom = angles_bottom.min(), angles_bottom.max()

        # Assume top and bottom arcs span the same angular range
        angle_start = min(angle_start_top, angle_start_bottom)
        angle_end = max(angle_end_top, angle_end_bottom)

        ray_angles = np.linspace(angle_start, angle_end, num_rays)

        rays = [] #list that will contain numpy arrays of all point coordinates in a ray

        for angle in ray_angles:
            # compute top and bottom points on arcs
            top_pt = (oy + top_radius * np.sin(angle), ox + top_radius * np.cos(angle)) # (row, col) or (y, x)
            bottom_pt = (oy + bottom_radius * np.sin(angle), ox + bottom_radius * np.cos(angle))

            # generate points along ray from top_pt to bottom_pt
            y_vals = np.linspace(top_pt[0], bottom_pt[0], points_per_ray) #rows
            x_vals = np.linspace(top_pt[1], bottom_pt[1], points_per_ray) #cols

            ray_coords = np.vstack((y_vals, x_vals)).T  # shape: (num_points, 2)
            rays.append(ray_coords)
    
    elif transducer == "LINEAR":
        tl = corners['top_left']
        tr = corners['top_right']
        bl = corners['bottom_left']
        br = corners['bottom_right']

        # x (col) coordinates of rays, linearly spaced between left and right edge
        start_x_vals = np.linspace(tl[1], tr[1], num_rays)
        start_y = tl[0]
        end_y = bl[0]

        rays = []

        for x in start_x_vals:
            y_vals = np.linspace(start_y, end_y, points_per_ray)
            x_vals = np.full(points_per_ray, x)
            ray_coords = np.vstack((y_vals, x_vals)).T
            rays.append(ray_coords)
        

    return rays

def render_rays(rays, shape):
    """
    Converts output of generate_rays (a list of rays, where each ray is a numpy array with the ray's point coordinates)
    into 2D numpy array (of inputted shape) of all rays.
    """

    ray_mask = np.zeros(shape, dtype=np.uint8)

    for ray in rays:
        for y, x in ray:
            # round to nearest integer pixel location
            row = int(round(y))
            col = int(round(x))
            ray_mask[row, col] = 255

    return ray_mask


### Shadow detection along ray profiles

def ray_intensity_profiles(frame_image, rays):
    profiles = []
    for ray in rays: # ray is (points_per_ray, 2) with (row, col)
        coords = ray.astype(int) #need integer pixel values to grab intensities from frame_image along ray
        intensities = frame_image[coords[:,0], coords[:,1]] # sample the intesity value for each profile point
        profiles.append(intensities)
    return profiles # list of 1D numpy arrays, each array contains the intensity values along one ray

def shadow_segments_per_profile(profile, shadow_threshold, min_shadow_length, shadow_min_threshold):
    """
    Takes in a single ray profile (1D numpy array containing the intensity values along one ray).
    Outputs a list of tuples (a,b), where each tuple contains start and end index of shadow region.

    Shadows must be <= shadow_threshold (0-255), and must we at least as long as min_shadow_length.
    ***min_shadow_length is in terms of profile points, not pixels. Thus the physical min length
        depends on points_per_ray as inputted by user in generate_rays().

    """
    profile_dark_points = (profile >= shadow_min_threshold) & (profile <= shadow_threshold)
    shadow_segments = []
    
    start_index = None
    for index, boolean_dark_enough in enumerate(profile_dark_points): #profile_dark_points has boolean value for each profile point
        if boolean_dark_enough: #if the profile point we're on is dark enough to be a shadow
            if start_index is None:
                start_index = index # start tracking this potential shadow region
        
        else: #profile point we're on is not dark enough to be a shadow
            if start_index is not None: #if already tracking a shadow region
                length = index - start_index
                if length >= min_shadow_length:
                    shadow_segments.append((start_index, index-1)) #add segment location to list of segments
                start_index = None
    # Case where shadow segment doesn't end, assign the end of shadow as end of profile
    if start_index is not None:
        length = len(profile) - start_index
        if length >= min_shadow_length:
            shadow_segments.append((start_index, len(profile)-1))
    return shadow_segments

def shadow_segments(profiles, shadow_threshold = 20, min_shadow_length = 10, shadow_min_threshold = 0):
    """
    Run shadow_segments_per_profile() for each profile, compiles results in all_rays_shadow_segments list.
    """
    all_rays_shadow_segments = []
    for profile in profiles:
        shadow_segments = shadow_segments_per_profile(profile, shadow_threshold, min_shadow_length, shadow_min_threshold)
        all_rays_shadow_segments.append(shadow_segments)
    return all_rays_shadow_segments

def ray_shadow_mask(rays, all_rays_shadow_segments, shape):
    ray_shadow_mask = np.zeros(shape, dtype=np.uint8)

    # rays is a list containing numpy arrays of all point coordinates in a ray
    for ray_point_coords, shadow_segments in zip(rays, all_rays_shadow_segments):
        for start_index, end_index in shadow_segments: # for each shadow segment
            for i in range(start_index, end_index+1): #iterate through profile points of segment
                y, x = ray_point_coords[i] # grab 2D pixel location of this profile point
                #round to integer pixel locations
                row = int(round(y))
                col = int(round(x))
                ray_shadow_mask[row, col] = 255
    return ray_shadow_mask


### Final smooth shadow mask

def filter_standalone_shadow_rays(ray_shadows, rays, all_rays_shadow_segments, neighbor_range=1):
    """
    Remove shadow pixels for rays that have no neighboring rays with shadows.
    neighbor_range is num of neighboring rays on each side to check
    
    Returns shadow mask with standalone rays removed.
    """
    filtered_mask = ray_shadows.copy()

    num_rays = len(rays)

    # identify rays that have shadow segments
    rays_with_shadows = [ray_index for ray_index, shadow_segments in enumerate(all_rays_shadow_segments) if len(shadow_segments) > 0]
    rays_with_shadows_set = set(rays_with_shadows)

    for ray_index in rays_with_shadows:
        # check ray's neighbors within neighbor_range, and within bounds of 0 to num_rays
        neighbors = range(max(0, ray_index - neighbor_range), min(num_rays, ray_index + neighbor_range + 1))
        neighbors_with_shadow = [n for n in neighbors if n != ray_index and n in rays_with_shadows_set]

        if len(neighbors_with_shadow) == 0: # if ray is standalone, remove its shadow pixels from mask
            shadow_segments = all_rays_shadow_segments[ray_index]
            ray_point_locations = rays[ray_index]
            for start_index, end_index in shadow_segments:
                for index in range(start_index, end_index + 1):
                    y, x = ray_point_locations[index]
                    row = int(round(y))
                    col = int(round(x))
                    filtered_mask[row, col] = 0

    return filtered_mask

def smooth_and_fill_shadow_mask(shadow_mask, dilation_iter=5, closing_iter=3):
    """
    Expand and smooth initial shadow mask using morphological operaions
        - dilation_iter: num of dilation iterations to expand shadows
        - closing_iter: num of closing iterations to smooth and fill gaps within shadows

    """
    binary_mask = (shadow_mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) #elliptical kernel
    
    dilated = cv2.dilate(binary_mask, kernel, iterations=dilation_iter) #expands shadow regions outward
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=closing_iter) #fills small holes and smooths edges

    # invert image and use flood fill to isolate holes
    inv_img = 1 - closed
    height, width = inv_img.shape
    mask_floodfill = np.zeros((height+2, width+2), np.uint8) # the +2 adds a black 1-pixel border to start flood fill from
    cv2.floodFill(inv_img, mask_floodfill, (0,0), 255) # start floodfill from (0,0) which is black background
    holes = cv2.bitwise_not(inv_img) # invert flood filled image to get holes

    filled_mask = closed | (holes // 255) # combine holes with closed image to fill them
    filled_mask = (filled_mask * 255).astype(np.uint8) #convert back to 0/255 values

    return filled_mask

def extract_largest_shadows(filled_mask, min_area=500):

    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(filled_mask)

    # extract only the shadows that are >= min_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)
                # -1 means draw all contours (in this case just one contour in the list)

    return final_mask



def filter_artifact_segments_by_length(all_segments, max_length):
    """
    all_segments: list of lists of (start_idx, end_idx) tuples, one list per ray
    max_length: maximum allowed total artifact length (points) per ray
    
    Returns filtered list where rays with total artifact length > max_length have empty segments
    """
    filtered_segments = []
    for segments in all_segments:
        total_length = sum(end - start + 1 for start, end in segments)
        if total_length <= max_length:
            filtered_segments.append(segments)
        else:
            filtered_segments.append([])  # discard all segments in this ray
    return filtered_segments


def two_threshold_tail_segments_per_profile(profile, presence_threshold, tail_threshold):
    """
    For a single ray intensity profile:
    - First check if the profile contains any point >= presence_threshold.
      If no points meet this, return empty list (no artifact).
    - If yes, find the furthest down (max index) point >= tail_threshold.
    - Return segment from that point to the end of profile as artifact.

    Args:
        profile (np.ndarray): 1D array of intensities along the ray.
        presence_threshold (float): Threshold to decide if ray should be considered.
        tail_threshold (float): Threshold to find the furthest down point.

    Returns:
        List of tuples [(start_idx, end_idx)] representing artifact segments.
    """
    if np.all(profile < presence_threshold):
        # No points >= presence_threshold, so no artifact
        return []
    
    # Find indices where intensity >= tail_threshold
    tail_indices = np.where(profile >= tail_threshold)[0]
    if len(tail_indices) == 0:
        # No points meet tail threshold, no artifact
        return []
    
    start_index = tail_indices.max()
    return [(start_index, len(profile) - 1)]


def two_threshold_tail_segments(profiles, presence_threshold, tail_threshold):
    all_segments = []
    for profile in profiles:
        segments = two_threshold_tail_segments_per_profile(profile, presence_threshold, tail_threshold)
        all_segments.append(segments)
    return all_segments





if __name__ == "__main__":

    from pydicom import dcmread
    from unified_check_helper_funcs.load_dicom import get_dicom_array
    from unified_check_helper_funcs.metadata_removal import remove_metadata

    r"linear probe dicoms\P7HC7OO6"



    # frame_num = 475
    # dicom_file = "shadow dicoms\starting_US_test_c"

    # dicom = dcmread(dicom_file)
    # dicom_array = get_dicom_array(dicom)
    # uncleaned_frame = dicom_array[frame_num]
    # transducer = dicom.TransducerType

    # # sector_mask = sector(frame_image)
    # # sector_image = Image.fromarray(sector_mask)
    # # sector_image.save(f"sector.png")


    # ### Clean frame

    # template = np.load("LOGIC_E9.npy")
    # cleaned_frame = remove_metadata(uncleaned_frame, template)

    # ### Ray generation
    
    # corners = find_sector_corners(cleaned_frame, transducer = transducer)
    # rays = generate_rays(corners, num_rays=200, points_per_ray = 100, transducer = transducer)

    # ray_mask = render_rays(rays, cleaned_frame.shape)
    # ray_image = Image.fromarray(ray_mask)
    # ray_image.save(f"all_rays.png")


    # ### Shadow detection along ray profiles
        
    # profiles = ray_intensity_profiles(uncleaned_frame, rays)
    # all_rays_shadow_segments = shadow_segments(profiles, shadow_threshold=20, min_shadow_length=10)

    # # Dark shadows
    # #all_rays_shadow_segments = shadow_segments(profiles, shadow_threshold=avg, min_shadow_length=points_per_ray/10, shadow_min_threshold = 0)
    
    # # Medium, long shadows
    # #all_rays_shadow_segments = shadow_segments(profiles, shadow_threshold=avg*3, min_shadow_length=points_per_ray/3, shadow_min_threshold = avg)

    # # Posterior acoustic enhancement
    # #all_rays_shadow_segments = shadow_segments(profiles, shadow_threshold=255, min_shadow_length=points_per_ray/10, shadow_min_threshold = avg*4.5)


    # ### Final smooth shadow mask
        
    # filtered_shadow_mask = filter_standalone_shadow_rays(ray_shadows, rays, all_rays_shadow_segments, neighbor_range=1)
    # filled_shadow_mask = smooth_and_fill_shadow_mask(filtered_shadow_mask, dilation_iter=4, closing_iter=3)
    # final_shadow_mask = extract_largest_shadows(filled_shadow_mask, min_area=500)
    # Image.fromarray(final_shadow_mask).save(f"shadow_mask.png")


