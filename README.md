## UROP2025-ultrasound-reconstruction

This repository provides Python code for 3D organ reconstruction from 2D ultrasound scans (DICOM format), integrating artifact detection, segmentation, pass-based filtering, and voxel-to-mesh reconstruction.

The scripts in Mini Project 1 (image handling basics), Mini Project 2 (thresholding segmentation), and Mini Project 3 (simple 3D reconstruction) were practice exercises for foundational methods. They are included for completeness but are not central to the main pipeline.

The remaining folders contain the more advanced and directly useful scripts:

**Artifact Detection:**
- unified_check.py: Runs detection for noncontact, blur, shadow, and posterior acoustic enhancement (parameters adjustable per scan).
- unified_check_helper_funcs: Helper scripts used in unified_check, each detecting one artifact type.
- PNN_reconstruct.py: Pixel-Nearest Neighbor (PNN) reconstruction, mapping 2D frames into a 3D voxel grid.
- multipass_reconstruct.py: PNN reconstruction tuned to a specific multi-pass test dataset.
- combined_PNN_and_artifact_detection.py: Combines artifact detection/removal with PNN for improved reconstruction.
- add_artifacts.py: Artificially introduces artifacts (blur, shadow) into ultrasound frames.

**Pass-Overlap Thresholding:**
- pass_detection.py: detects start and end frame indicies of back-and-forth passes of ultrasound probe during a scan.
- pass_thresholding_scan1.py: applies pass-overlap thresholding, requiring multi-pass agreement in reconstruction.

**UNet and Engaging:**
- complete_pipeline.py: End-to-end pipeline:
    - Load frames (optionally add synthetic artifacts)
    - (Optional) Artifact detection → filter flagged frames
    - UNet segmentation (via MIT Engaging GPU cluster: upload data, generate & run predict.sh, download results)
    - PNN reconstruction with UNet segmentations
    - Mesh generation from voxel grid (including volume calculation)
    - Interactive mesh visualization
- engaging_sh_scripts: Shell scripts (train.sh, predict.sh) to be uploaded to the MIT Engaging Directory and submitted manually to train or run inference with UNet segmentation model.
- UNet_setup_code: Helper functions for UNet training and inference (create_json.py, load_training_dataset.py, nifti_predictions_to_array.py, etc.).
- imports for complete_pipeline: Helper functions for other pipeline steps.
- nnUNet_raw/Dataset100KidneyUS: Example dataset for training UNet segmentation.
  
**Visualize Data and Results:**
- visualize_frames.py: Frame-by-frame viewer with slider control.
- visualize_2Dcross-sections_and_3Dgrid.py: Interactive voxel grid explorer (2D cross-sections + 3D rendering).
- compare_two_surface_meshes.py: Aligns and compares two meshes using ICP; provides overlays, error heatmaps, volumes, and distance metrics (Chamfer/Hausdorff).
