# Reimplementing MobileNetv3 and Analyzing its Performance

## Table of Contents
- [Introduction](#introduction)
- [Developer](#developer)
- [Libraries Used](#libraries-used)
- [MobileNetV3 and MobileNetV2 Implementation Using Python and TensorFlow](#mobilenetv3-and-mobilenetv2-implementation-using-python-and-tensorflow)
  - [Steps Involved](#steps-involved)
  - [Repository Used](#repository-used)
- [Training MobileNetV3 Using PyTorch](#training-mobilenetv3-using-pytorch)
  - [Repository Used](#repository-used-1)
  - [Steps Involved](#steps-involved-1)
-[Developer](#developer)
- [Image Classification Using the Trained MobileNetV3 Model](#image-classification-using-the-trained-mobilenetv3-model)
 

## Introduction
This project aims to compare the performances of MobileNetV2 and MobileNetV3 architectures using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, making it a suitable benchmark for evaluating different neural network architectures.

**Dataset URL:** [CIFAR-10 Dataset](#)

## Developer
This code is entirely developed by Manjushree Kumaravel.

## Libraries Used
The following libraries were used in this project:
- TensorFlow: For building and training the MobileNetV3 model.
- PyTorch: Used for training MobileNetV3 model with an alternate approach.
- Matplotlib: Utilized for visualizing the training results.
- NumPy: Used for numerical computations and handling arrays.
- TorchVision: Employed for image dataset handling and transformations.
### For TensorFlow 

- `tensorflow` for building and training neural networks.
- `tensorflow.keras.layers` for defining various layers in the network.
- `tensorflow.keras.models` for creating the model architecture.
- `tensorflow.keras.datasets` for loading the CIFAR-10 dataset.
- `tensorflow.keras.utils` for data preprocessing and utilities.

### For PyTorch 

- `torch` and `torchvision` for building and training neural networks using PyTorch.
- `torch.utils.data` for handling data loaders.
- Various helper functions imported from the repository:
    - `helper_utils` for setting seeds and deterministic behavior.
    - `helper_evaluate` for computing metrics and evaluation.
    - `helper_plotting` for visualizing training progress.
    - `helper_data` for loading and transforming CIFAR-10 data.

---

## MobileNetV3 and MobileNetV2 Implementation Using Python and TensorFlow
### Steps Involved
1. **Cell 1: Import Necessary Libraries**  
   Importing the required libraries and modules for building and training the MobileNetV3 model using TensorFlow.

2. **Cell 2: Define MobileNetV3 Architecture**  
   Defining the architecture for MobileNetV3 using functions and building different components of the model.

3. **Cell 3: Create the MobileNetV3 and MobileNetV2 Model**  
   Creating instances of the MobileNetV3 and MobileNetV2 models using predefined architectures available in TensorFlow and adjusting them based on the CIFAR-10 dataset.

4. **Cell 4: Compile the Model**  
   Compiling the model by specifying the optimizer, loss function, and evaluation metrics.

5. **Cell 5: Load and Preprocess Data (CIFAR-10 Dataset)**  
   Loading and preprocessing the CIFAR-10 dataset as an example. This part can be replaced with custom dataset loading and preprocessing.

6. **Cell 6: Train the Model**  
   Training the MobileNetV3 model on the dataset with specified parameters like batch size and epochs.

7. **Cell 7: Plot the Results of V2 and V3 Architecture**  
   Visualizing the training and validation accuracy, as well as the loss, for both MobileNetV2 and MobileNetV3 architectures.

### Repository Used
The project utilizes the following repository:
- **GitHub Repository:** [deeplearning-models](#)  
  This repository was used for PyTorch-based implementation, containing utility functions, data loading, and model training scripts.

---

## Training MobileNetV3 Using PyTorch
### Repository Used
The project utilizes the following repository:
- **GitHub Repository:** [deeplearning-models](#)  
  This repository was used for PyTorch-based implementation, containing utility functions, data loading, and model training scripts.
### Steps Involved
1. **Cell 1: Cloning GitHub Repo and Importing Libraries**  
   Cloning a GitHub repository and importing necessary libraries and modules for training MobileNetV3 using PyTorch.

2. **Cell 2: Settings**  
   Setting parameters such as random seed, batch size, number of epochs, and the device for training.

3. **Cell 4: Loading CIFAR 10**  
   Loading the CIFAR-10 dataset, performing transformations, and checking the dataset structure.

4. **Cell 5: Model**  
   Loading MobileNetV3 model architecture using PyTorch.

5. **Cell 6: Optimizing the Model Using Adam Optimizer**  
   Optimizing the model using the Adam optimizer and training the classifier.

6. **Cell 7: Plotting Loss and Accuracy**  
   Plotting the training loss and accuracy obtained during the training process.

---
## Developer
This code is entirely developed by Manjushree Kumaravel.

## Image Classification Using the Trained MobileNetV3 Model
A utility function `predict_image_label` is provided to predict labels for new images using the trained model.

