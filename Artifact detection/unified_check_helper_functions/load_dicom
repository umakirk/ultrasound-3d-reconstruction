import numpy as np

def get_dicom_array(dicom):
    pixel_array = dicom.pixel_array
    greyscale = np.mean(pixel_array, axis=-1).astype(np.uint8)
    return greyscale


if __name__ == "__main__":
    from pydicom import dcmread

    dicom_file = "shadow dicoms\starting_US_test_c"
    dicom = dcmread(dicom_file)
    greyscale = get_dicom_array(dicom)

    # reorder as necessary
    print(greyscale.shape)
    reordered = np.transpose(greyscale, (2,1,0))
    print(reordered.shape)
    np.save('starting_US_test_c', reordered)

    print(dicom.TransducerType) #"CURVED LINEAR" or "LINEAR"
 
