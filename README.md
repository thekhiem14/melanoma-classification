# Melanoma Skin Cancer Classification

This project implements a deep learning pipeline using CNN and ResNet50 to classify 7 types of skin lesions from dermoscopic images, with a focus on detecting melanoma, the most dangerous type of skin cancer.

## Report

A detailed technical report about this project is available here: [My report](https://drive.google.com/file/d/1_iBKBcYAvy1bDBIG3svwfB0PJLlmuZxr/view) <br>
It includes background theory, implementation steps, challenges, evaluation metrics, and future development directions.

## Dataset

I use the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (10,000+ dermoscopy images), available on Kaggle.

## Objectives

- Classify skin lesions into 7 classes with high accuracy.
- Maximize recall for melanoma (≥ 90%).
- Address class imbalance with resampling + focal loss.

## Techniques Used

- Convolutional Neural Networks (CNN)
- Transfer Learning with ResNet50
- Data Augmentation (Flip, Rotate, Zoom)
- `RandomOverSampler` for balancing
- Focal Loss with Class Weights
- Fine-tuning with `EarlyStopping`

## Model Architecture

- **Base**: ResNet50 (pretrained on ImageNet)
- **Top Layers**: `GlobalAveragePooling` → `Dropout` → `BatchNorm` → `Dense(7, softmax)`
- **Optimization**: Optimized using Adam and weighted Focal Loss.

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC for melanoma detection

## Results

- **Melanoma recall**: > 90%
- Balanced accuracy across all 7 classes.
- Real-time prediction with PyQt5 UI (optional).

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/thekhiem14/melanoma-classification.git](https://github.com/thekhiem14/melanoma-classification.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd melanoma-classification
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

####  Option 1: Run the notebook

Open `melanoma.ipynb` and run all cells (ensure HAM10000 images and CSVs are in place).

####  Option 2: Run the app (if implemented)

```bash
python main.py
```
## Project Structure

```text
├── melanoma.ipynb         # Main notebook with model pipeline
├── main.py                # Entry point (optional UI)
├── model_loader.py        # Load trained model
├── utils.py               # Data preprocessing
├── assets/                # Contains trained .h5 or .keras models
├── data/                  # Your CSV/image data (not uploaded to GitHub)
└── requirements.txt
