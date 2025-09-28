Fish Classification Using Deep Learning
Author: Vijayarajan


Introduction

This report presents the results of a fish classification project performed using my own
dataset of 3,187 images across 11 fish classes. The goal was to identify fish species
accurately using state-of-the-art deep learning models. Multiple architectures were tested
including MobileNetV2, VGG16, ResNet50, and a custom CNN model.

Dataset

The dataset consists of 11 classes: animal fish, animal fish bass, black sea sprat, gilt head
bream, horse mackerel, red mullet, red sea bream, sea bass, shrimp, striped red mullet,
and trout. The dataset is relatively balanced except for the 'animal fish bass' class which
only had 13 samples, making classification harder for this class.
 
Methodology

Images were resized to 224x224 pixels, normalized, and augmented with rotations, flips,
and zoom transformations. All models were trained with Adam optimizer and categorical
cross-entropy loss. Early stopping was used to avoid overfitting.

Results Overview

Model Accuracy Macro F1 Observation
Custom CNN 0.50 - Too shallow, underfit the dataset
VGG16 (frozen) 0.97 - Strong performance
VGG16 (fine-tuned) 0.96 - Slight overfitting reduced performance
MobileNetV2 0.95 0.89 Good but struggled with minority class
MobileNetV2 (unfreeze 15 layers) 0.96 0.90 Improved recall and accuracy
ResNet50 (unfreeze 15 layers) 1.00 0.98 Best model, near-perfect classification


 Conclusion:

The results clearly show that ResNet50 outperformed all other models, achieving 100%
accuracy on my dataset. VGG16 and MobileNetV2 are strong alternatives with slightly
lower accuracy but still robust performance. My own custom CNN model achieved only
50% accuracy, which highlights the importance of deeper architectures and transfer
learning for this problem. Future improvements can focus on data balancing, model
pruning for deployment, and real-time classification APIs.
