README: Image Classification with EfficientNet
This notebook demonstrates an image classification workflow using a pre-trained EfficientNet_B0 model from torchvision. The process involves loading and preparing an image dataset, setting up a transfer learning model, training it, and making predictions on new images.

1. Data Preparation
Unzipping Data: The notebook starts by unzipping the imagewoof200.zip file, which contains the image dataset.
Path Setup: pathlib is used to define paths to the extracted image data, specifically for the imagewoof200 directory.
Data Structure Check: A helper function check_data is defined and used to inspect the directory structure and count of images within the imagewoof200/test and imagewoof200/train subdirectories.
2. DataLoaders
Transforms: Initial manual transforms are defined for resizing, converting to PyTorch tensors, and normalizing images. Later, automatic transforms from the EfficientNet weights are used for consistency with the pre-trained model.
create_dataloaders Function: This function utilizes torchvision.datasets.ImageFolder to load images from specified directories (train_dir, test_dir). It then wraps these datasets in torch.utils.data.DataLoader instances for efficient batch processing during training and evaluation. It also extracts class_names from the dataset.
3. Model Setup (Transfer Learning)
Device Configuration: Checks for CUDA availability and sets the device to cuda if a GPU is present, otherwise cpu.
Load Pre-trained Model: An EfficientNet_B0 model is loaded using torchvision.models.efficientnet_b0 with pre-trained IMAGENET1K_V1 weights. The model is moved to the configured device.
Model Summary: torchinfo.summary is used to display a detailed summary of the model architecture, including input/output shapes, parameter counts, and trainability.
Feature Freezing: The feature extraction layers (model.features) of the pre-trained model are frozen by setting param.requires_grad = False. This prevents their weights from being updated during training, leveraging the learned features.
Classifier Head Modification: The original classifier head of the EfficientNet model, designed for 1000 ImageNet classes, is replaced with a new torch.nn.Sequential classifier suitable for the specific number of classes in the current dataset (len(class_names)). This new head is made trainable.
Updated Model Summary: Another torchinfo.summary is run to confirm that only the parameters of the new classifier head are trainable.
4. Training
Loss Function and Optimizer: nn.CrossEntropyLoss() is chosen as the loss function, and torch.optim.Adam is used as the optimizer with a learning rate of 0.001.
Training Functions: Three helper functions (train_step, test_step, train) are defined:
train_step: Performs a single training pass over all batches in a dataloader, calculating loss, backpropagating, and updating model weights.
test_step: Performs a single evaluation pass over all batches in a dataloader without updating weights, calculating loss and accuracy.
train: Orchestrates the training and testing loops for a specified number of epochs, calling train_step and test_step and storing results.
Training Execution: The train function is called with the model, dataloaders, optimizer, loss function, device, and 10 epochs, and the training results (loss and accuracy for train and test sets) are printed per epoch.
5. Prediction
transform_predict Function: This function takes a model, image path, class names, and optional transforms to make a prediction on a single image. It loads the image, applies the necessary transformations, moves it to the correct device, performs inference, and visualizes the image along with its predicted class and confidence score.
Custom Image Prediction: The transform_predict function is demonstrated by making a prediction on a golden.jpeg image, showcasing the model's ability to classify an unseen image.

For Dataset: https://www.kaggle.com/datasets/atilsamancioglu/imagewoof200
