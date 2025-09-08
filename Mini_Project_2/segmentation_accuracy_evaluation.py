import numpy as np
from scipy.spatial.distance import cdist
import cv2
from medpy.metric.binary import hd95


def evaluations(ground_truth_filename, mask_filename):
    mask_pred = cv2.imread(mask_filename)
    mask_true = cv2.imread(ground_truth_filename)

    mask_pred = (mask_pred > 0).astype(np.uint8)
    mask_true = (mask_true > 0).astype(np.uint8)


    def dice_score(mask_pred, mask_true):
    
        intersection = np.logical_and(mask_pred, mask_true).sum()
        total = mask_pred.sum() + mask_true.sum()
        
        return 2 * intersection / total


    def iou_score(mask_pred, mask_true):
        intersection = np.logical_and(mask_pred, mask_true).sum()
        union = np.logical_or(mask_pred, mask_true).sum()

        return intersection / union


    def hausdorff_95(mask_pred, mask_true):
        hd = hd95(mask_pred > 0, mask_true > 0)
        return hd


    def pixel_accuracy(mask_pred, mask_true):
        # num correct pixels in pred mask, divided by total number of pixels in true mask
        num_correct = (mask_pred==mask_true).sum()
        total_num = mask_true.size

        return num_correct/total_num


    for method in [dice_score, iou_score, hausdorff_95, pixel_accuracy]:
        print(f"{method.__name__}: {method(mask_pred, mask_true)}")


print("PHANTOM")
evaluations("test_files/ground_truth_phantom_scan.png", "mask_simple_phantom_scan.png")

print("\nCAROTID")
evaluations("test_files/ground_truth_carotid.png", "mask_simple_inverted_carotid.png")


print("\nIJ_CAROTID")
evaluations("test_files/ground_truth_IJ_carotid.png", "mask_simple_inverted_IJcarotid.png")


print("\nDICOM CAROTID")
evaluations("test_files/ground_truth_carotid_dicom.png", "mask_simple_inverted_carotid_dicom.png")
