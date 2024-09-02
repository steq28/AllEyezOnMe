![Untitled-1](https://github.com/user-attachments/assets/343b0635-b1bd-4280-a782-fd7b8a8c0eb0)

# AllEyezOnMe

AllEyezOnMe is a machine learning project focused on recognizing sign language, specifically the American Sign Language (ASL) alphabet and numbers, through image processing. The project uses a Random Forest classifier and MediaPipe for hand landmark detection, allowing for real-time recognition of ASL gestures using a webcam.

## Table of Contents

-   [AllEyezOnMe](#alleyezonme)
    -   [Table of Contents](#table-of-contents)
    -   [Overview](#overview)
    -   [Dataset](#dataset)
    -   [Installation](#installation)
    -   [Usage](#usage)
        -   [1. Creating the Dataset](#1-creating-the-dataset)
        -   [2. Training the Model](#2-training-the-model)
        -   [3. Real-Time Prediction](#3-real-time-prediction)
    -   [Scripts and Notebooks Breakdown](#scripts-and-notebooks-breakdown)
    -   [Acknowledgments](#acknowledgments)
    -   [License](#license)

## Overview

This project combines computer vision and machine learning to identify ASL letters and numbers from hand gestures captured by a webcam. It is composed of the following key components:

1. **Dataset Preparation**: Extracting features from images of hands showing ASL signs.
2. **Model Training**: Training a Random Forest classifier to recognize ASL letters and numbers.
3. **Real-Time Prediction**: Using the trained model to predict ASL signs from live webcam input.

## Dataset

To enhance the diversity and robustness of the model, multiple datasets from Kaggle were combined:

-   [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
-   [ASL Dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset)
-   [Sign Language Gesture Images Dataset](https://www.kaggle.com/datasets/ahmedkhanak1995/sign-language-gesture-images-dataset)

These datasets contain images of hands depicting ASL signs, which are used to train the model.

## Installation

To run this project, you need to have Python and Jupyter Notebook installed. Follow the steps below to set up your environment:

1. Clone this repository:

    ```bash
    git clone https://github.com/steq28/AllEyezOnMe.git
    cd AllEyezOnMe
    ```

2. Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Creating the Dataset

First, generate the dataset by processing the images to extract hand landmarks:

```bash
python create_dataset.py
```

This script will process the images found in the `./dataset/train` directory, extract the hand landmarks using MediaPipe, and save the features and labels in a file called `data.pickle`.

### 2. Training the Model

To train the model, open the `train_model.ipynb` Jupyter Notebook:

```bash
jupyter notebook train_model.ipynb
```

In this notebook, you will:

-   Load the dataset from `data.pickle`.
-   Train a Random Forest classifier.
-   Evaluate the model's performance (accuracy, confusion matrix, etc.).
-   Save the trained model in a file called `model.p`.

Follow the instructions in the notebook cells to complete the training process.

This model has **0.98945 accuracy**.

### 3. Real-Time Prediction

After training, you can run the real-time prediction script:

```bash
python main.py
```

This script will activate your webcam, process the video feed to detect hand landmarks, and predict the ASL sign being shown. The predicted sign will be displayed on the video feed.

Press `q` to exit the webcam interface.

## Scripts and Notebooks Breakdown

-   **create_dataset.py**:
    -   Processes images in the dataset to extract hand landmarks.
    -   Saves the processed data (features and labels) into `data.pickle`.
-   **train_model.ipynb**:
    -   Jupyter Notebook used to load the dataset from `data.pickle`.
    -   Trains a Random Forest classifier.
    -   Evaluates and visualizes the model's performance.
    -   Saves the trained model as `model.p`.
-   **main.py**:
    -   Activates the webcam and processes each frame to detect hand landmarks.
    -   Uses the trained model to predict the ASL sign shown in front of the webcam.
    -   Displays the prediction in real-time on the webcam feed.

## Acknowledgments

This project relies on datasets made available by the following Kaggle contributors:

-   [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
-   [ASL Dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset)
-   [Sign Language Gesture Images Dataset](https://www.kaggle.com/datasets/ahmedkhanak1995/sign-language-gesture-images-dataset)

Their contributions were crucial in creating a diverse and effective training set.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
