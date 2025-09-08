import math
import os
import numpy as np
from PIL import Image

### Read in single PNG image and display###

def open_greyscale_image(filename):
    img = Image.open(filename)
    img.show()

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image("test_images/cat.png")
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}
    
sample_image = load_greyscale_image("Sample_US_img.png")
# open_greyscale_image("Sample_US_img.png")


### Read in full DICOM scan and output 3 pieces of DICOM info,
# the number of images in the scan, and the size of the images ###

#from pydicom import examples
#path = examples.get_path("ct")
#reading in from path
#ds1 = dcmread(path)
#reading in from file itself
#with open(path, 'rb') as infile:
#    ds2 = dcmread(infile)
#grabbing FileDataset directly
#ds3 = examples.ct

from pydicom import dcmread # dcmread() will read any DICOM dataset

# view the contents of the entire dataset by using print()
starting_US_test_c = dcmread("starting_US_test_c")

#num_images = starting_US_test_c['NumberOfFrames'].value

print("Study Date:", starting_US_test_c.StudyDate)
print("Study Time:", starting_US_test_c.StudyTime)
print("Image Type:", starting_US_test_c.ImageType)

print("Number of images in scan:", starting_US_test_c.NumberOfFrames)
rows = starting_US_test_c.Rows
cols = starting_US_test_c.Columns
print("Size of images:", rows, "by", cols)

# starting_US_test_c.pixel_array.shape -> outputs (num scans, height, width, num channels)



### Slice up full scan into individual images
# display an image, save another image in PNG and NIfTI formats ###

# Get the pixel data as a NumPy array
# 4D array: Number of images/scans, rows, cols, num channels
pixel_array = starting_US_test_c.pixel_array
print("Shape:", pixel_array.shape)  # (577, 649, 850, 3)

# averaging RBG values at each pixel:
grayscale_array = np.mean(pixel_array, axis=-1).astype(np.uint8)  # shape: (577, 649, 850)

# Displaying a slice
frame464_array = grayscale_array[464] # shape: (649, 850)
frame464 = Image.fromarray(frame464_array) # from array converts a NumPy array into a PIL Image object.
#frame464.show()

# Saving a slice to PNG and NIfTI
frame327_array = grayscale_array[327]
frame327 = Image.fromarray(frame327_array)
frame327.save("frame327.png")

import nibabel as nib
# NIfTI expects at least 3D data, so add a singleton dimension
frame327_array3D = frame327_array[..., np.newaxis]  # shape: (rows, cols, 1)

# Create a NIfTI image object
nifti_img = nib.Nifti1Image(frame327_array3D, affine=np.eye(4)) # just identity matrix, 
                                                                # no transformations applied to pixels

# Save to file
nib.save(nifti_img, 'frame327.nii')

# Tried displaying NIfTI image in ITK-SNAP but was oriented weirdly
# Used matplotlib to make sure NIfTI image is correct

#import matplotlib.pyplot as plt
#img_data = nifti_img.get_fdata()
#plt.imshow(img_data[:, :, 0], cmap="gray")
#plt.axis('off')
#plt.show()



### Find the minimum and maximum pixel intensity in the scan ###

print(f'Minimum pixel intensity: {np.min(grayscale_array)}')
print(f'Maximum pixel intensity: {np.max(grayscale_array)}')


### Transform an image ###

# Rotate upside down, print the location and value of all pixels greater than 200, 
# invert all pixels

rotated_array = np.rot90(frame327_array, 2)
rotated = Image.fromarray(rotated_array)
rotated.save("rotated_frame327.png")

indices = np.argwhere(rotated_array > 200) #gives array of [row col] indices
#for (row, col) in indices:
    #print(f"Pixel at ({row}, {col}) has value {rotated_array[row, col]}")

inverted_array = 255 - rotated_array
inverted = Image.fromarray(inverted_array)
inverted.save("rotated_and_inverted.png")

# Crop the image such that only the US sector is showing (not the metadata), 
# estimate the size (height, width) of the kidney, 
# choose two images and display the differences between them

import crop
import transparent_crop

crop.sector_crop('Sample_US_img.png')
transparent_crop.transparent_sector_crop('Sample_US_img.png')

import image_difference
frame325_array = grayscale_array[325]
frame325 = Image.fromarray(frame325_array)
frame325.save("frame325.png")

frame326_array = grayscale_array[326]
frame326 = Image.fromarray(frame326_array)
frame326.save("frame326.png")
image_difference.difference('frame325.png', 'frame326.png', 20)
#image_difference.frame_difference('starting_US_test_c', 325, 326, 5)

### Display images as video ###

import cv2 #using OpenCV library

#creating a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # MPEG-4 video format
height = int(rows)
width = int(cols)
video = cv2.VideoWriter('video.mp4', fourcc, 24, (width, height), isColor = False)

# Write each frame to the video
for frame in grayscale_array:
    video.write(frame)

video.release()
