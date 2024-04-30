# import SimpleITK as sitk
# import numpy as np
# import os
# import glob
#
#
# def calculate_dice_score(mask1, mask2):
#     """Calculate the Dice score between two binary masks."""
#     intersection = np.sum(mask1[mask2 == 1])
#     return 2.0 * intersection / (np.sum(mask1) + np.sum(mask2))
#
#
# def main():
#     base_dir = "/home/navid/Desktop/Papers/paper2/pancreas_segmentaion/Duct/three_inputs/swin_output"
#     num_folds = 4
#     num_steps = 37
#
#     dice_scores = []
#
#     for fold in range(1, num_folds + 1):
#         for step in range(0, num_steps):
#             sg_path = os.path.join(base_dir, f"sg_{step}_{fold}.nii.gz")
#             pred_path = os.path.join(base_dir, f"pred_{step}_{fold}.nii.gz")
#
#             if os.path.exists(sg_path) and os.path.exists(pred_path):
#                 sg_mask_sitk = sitk.ReadImage(sg_path)
#                 pred_mask_sitk = sitk.ReadImage(pred_path)
#
#                 sg_mask_np = sitk.GetArrayFromImage(sg_mask_sitk)
#                 pred_mask_np = sitk.GetArrayFromImage(pred_mask_sitk)
#
#                 dice_score = calculate_dice_score(sg_mask_np, pred_mask_np)
#                 dice_scores.append(dice_score)
#                 print(f"Dice score for sg_{step}_{fold} and pred_{step}_{fold}: {dice_score}")
#             else:
#                 print(f"Files for step {step} fold {fold} do not exist.")
#
#     # Calculate average Dice score
#     average_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
#     print(f"Average Dice Score: {average_dice}")
#
#
# if __name__ == "__main__":
#     main()

import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import label


# def dice_score_3d(vol1, vol2):
#     # Ensure the volumes have the same shape
#     if vol1.shape != vol2.shape:
#         return 0
#     else:
#         return np.sum(vol2[vol1 == 1]) * 2.0 / (np.sum(vol1) + np.sum(vol2))

def dice_score_per_slice(slice1, slice2):
    """Calculates the Dice score for a single slice if both slices have annotations."""
    if np.any(slice1) and np.any(slice2):  # Only calculate if both slices have annotations
        intersection = np.sum(slice1 & slice2)
        total = np.sum(slice1) + np.sum(slice2)
        return 2.0 * intersection / total
    return None  # Return None if one or both slices have no annotations


def dice_score_3d(vol1, vol2):
    """Calculates the average Dice score only for slices with annotations in both volumes."""
    dice_scores = []
    num_slices = vol1.shape[0]  # Assuming the first dimension is the slice dimension
    for i in range(num_slices):
        score = dice_score_per_slice(vol1[i], vol2[i])
        if score is not None:
            dice_scores.append(score)
    if dice_scores:
        return np.mean(dice_scores)
    return 0  # Return 0 or an appropriate value if no slices were annotated in both volumes


def remove_non_intersecting_objects(vol1, vol2):
    """Removes objects in vol2 that have no intersection with any object in vol1."""
    labeled_vol2, num_features = label(vol2)
    intersection_mask = vol1 * vol2  # Elements where both volumes have segmentations
    valid_labels = np.unique(labeled_vol2[intersection_mask > 0])
    filtered_vol2 = np.isin(labeled_vol2, valid_labels) * vol2
    return filtered_vol2


def plot_slices(vol1, vol2, slice_no):
    """Helper function to visualize slices."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(vol1[slice_no], cmap='gray')
    ax[0].set_title('Volume 1 Slice')
    ax[1].imshow(vol2[slice_no], cmap='gray')
    ax[1].set_title('Volume 2 Slice')
    plt.show()


def flip_volume(vol, axis=0):
    """Flip the volume along the specified axis. Default is 0 (z-axis)."""
    return np.flip(vol, axis=axis)


def rotate_volume(vol, k, axis):
    """Rotate the volume 90*k degrees along the specified axis."""
    return np.rot90(vol, k=k, axes=(axis, (axis + 1) % 3))


base_path1 = '/home/navid/Downloads/new_ducts/1-18 ducts'
base_path2 = '/home/navid/Desktop/Papers/paper2/pancreas_segmentaion/Duct/duct_dataset_normalized'

all_dice_scores = []  # To store all Dice scores
rotations = [(k, axis) for k in range(1, 4) for axis in range(3)]  # 90, 180, 270 degrees for each of 3 axes

# Loop through each subject
for i in range(1, 19):  # Folders from 1 to 18
    path1 = os.path.join(base_path1, str(i), 'duct.nii.gz')
    path2 = os.path.join(base_path2, str(i), 'mask.nii.gz')

    if os.path.exists(path1) and os.path.exists(path2):
        img1 = sitk.ReadImage(path1)
        img2 = sitk.ReadImage(path2)
        img1.SetOrigin(img2.GetOrigin())
        img1.SetDirection(img2.GetDirection())

        vol1 = sitk.GetArrayFromImage(img1)
        vol2 = sitk.GetArrayFromImage(img2)
        # vol2_filtered = remove_non_intersecting_objects(vol1, vol2)

        # Evaluate each rotation
        # for k, axis in rotations:
        vol2_rotated = rotate_volume(vol2, 2, 1)
        dice_score = dice_score_3d(vol1, vol2_rotated)
        all_dice_scores.append(dice_score)
        print(f"3D Dice Score for subject {i} with rotation {2 * 90} degrees around axis {1}: {dice_score}")
    else:
        print(f"Missing files for subject {i}, skipping.")

# Calculate the overall average Dice score across all rotations and subjects
if all_dice_scores:
    overall_average_dice = np.mean(all_dice_scores)
    print(f"Overall Average 3D Dice Score across all rotations: {overall_average_dice}")
else:
    print("No Dice scores calculated.")
