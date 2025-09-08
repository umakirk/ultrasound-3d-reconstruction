import numpy as np
from pydicom import dcmread

starting_US_test_c = dcmread("test_files/starting_US_test_c")
pixel_array = starting_US_test_c.pixel_array
greyscale = np.mean(pixel_array, axis=-1).astype(np.uint8)  # Shape: (X, Y, Z)

# reorder as necessary
print(greyscale.shape)
reordered = np.transpose(greyscale, (2,1,0))
print(reordered.shape)

np.save('starting_US_test_c', reordered)
