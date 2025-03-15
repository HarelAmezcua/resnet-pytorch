# ResNet-PyTorch for Pneumonia vs Healthy X-ray Classification

This repository contains a PyTorch implementation of a ResNet model for classifying X-ray images as either pneumonia or healthy.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
Pneumonia is a serious lung infection that can be diagnosed through chest X-rays. This project leverages the ResNet architecture to classify X-ray images as either pneumonia or healthy. The model is trained on a labeled dataset of X-ray images and achieves high accuracy in distinguishing between the two classes.

## Installation
To get started, clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/HarelAmezcua/resnet-pytorch.git
cd resnet-pytorch
pip install -r requirements.txt
```

## Usage
1. **Prepare the dataset**: Ensure you have the X-ray images for training and testing. The dataset should be organized into two folders: `train` and `test`, each containing subfolders `pneumonia` and `healthy`.

2. **Train the model**: Use the following command to train the ResNet model.
```bash
python train.py --data_dir path/to/dataset --epochs 50 --batch_size 32 --learning_rate 0.001
```

3. **Evaluate the model**: After training, evaluate the model using the test dataset.
```bash
python evaluate.py --model_path path/to/saved_model.pth --data_dir path/to/test_dataset
```

## Dataset
The dataset should be organized in the following structure:
```
dataset/
    train/
        pneumonia/
        healthy/
    test/
        pneumonia/
        healthy/
```
You can download a publicly available dataset such as the [Kaggle Chest X-ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Training
The training script (`train.py`) trains the ResNet model on the provided dataset. You can customize the training parameters such as the number of epochs, batch size, and learning rate.

Example command:
```bash
python train.py --data_dir path/to/dataset --epochs 50 --batch_size 32 --learning_rate 0.001
```

## Evaluation
The evaluation script (`evaluate.py`) assesses the model's performance on the test dataset. It calculates metrics such as accuracy, precision, recall, and F1-score.

Example command:
```bash
python evaluate.py --model_path path/to/saved_model.pth --data_dir path/to/test_dataset
```

## Results
The model trained on the X-ray dataset achieves the following results:
- **Accuracy**: X%
- **Precision**: X%
- **Recall**: X%
- **F1-score**: X%

(Note: Replace `X%` with actual values obtained from your model's evaluation.)

## Contributing
Contributions are welcome! If you have any improvements or new features to add, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

This README provides a comprehensive guide for users to understand, install, use, and contribute to the ResNet model for X-ray classification. Feel free to modify and expand it based on your specific needs.
