# Segmentation of the pancreatic anatomy
 
**Background:** The pancreas is a complex abdominal organ with many anatomical variations, making automated pancreas segmentation from medical images a challenging application.

**Purpose:** This paper presents a framework for segmenting individual pancreatic sub-regions and the pancreatic duct from three-dimensional (3D) computed tomography (CT) images.

**Methods:** A multi-agent reinforcement learning (RL) network was used to detect landmarks of the head, neck, body, and tail of the pancreas, and landmarks along the pancreatic duct in a selected target CT image. Using the landmark detection results, an atlas of pancreases was non-rigidly registered to the target image, resulting in anatomical probability maps for the pancreatic sub-regions and duct. The probability maps were augmented with multi-label 3D U-Net architectures to obtain the final segmentation results.

**Results:** To evaluate the performance of our proposed framework, we computed the Dice similarity coefficient (DSC) between the predicted and ground truth manual segmentations on a database of 82 CT images with manually segmented pancreatic sub-regions and 37 CT images with manually segmented pancreatic ducts. For the four pancreatic sub-regions, the mean DSC improved from 0.38, 0.44, and 0.39 with standard 3D U-Net, Attention U-Net, and Swin U-Net architectures, to 0.51, 0.47, and 0.49, respectively, when utilizing the proposed RL-based framework. For the pancreatic duct, the RL-based framework achieved a mean DSC of 0.70, significantly outperforming the standard approaches and existing methods on different datasets.

**Conclusions:** The resulting accuracy of the proposed RL-based segmentation framework demonstrates an improvement against segmentation with standard U-Net architectures.

![anatomy](https://github.com/sepidehamiri/Segmentation-of-the-pancreatic-anatomy/assets/18999283/885c4fed-4fd3-45e2-bdfc-339867b759ac)
