# Emotion Recognition via Computer Vision

![Alt text](assets/group_example.png?raw=true "Model Architecture")

This project investigates the usage of three different model configurations for the task of facial
emotion recognition (FER). An end-to-end classification pipeline is constructed containing a Viola-Jones face
recognition algorithm; a support vector machine with Scale Invariant Feature Transform feature descriptors,
a multi-layer perceptron with Histogram of Gradients feature descriptors, and a deep convolutional neural
network (DCNN) are examined as the pipelines’ classifiers. The DCNN is identified to greatly outperform
the other methods in both accuracy and speed.

Each model and its corresponding files for optimization can be found under the `src/` directory.

The paper detailing the results of this project can be found at the root level in a .pdf. 