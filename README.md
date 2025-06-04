# Alzheimer's Disease Detection using Deep Learning

A deep learning project that uses Convolutional Neural Networks (CNN) to classify brain MRI images for Alzheimer's disease detection with 90% accuracy.

## Project Overview

This project implements a CNN-based image classification system to detect different stages of Alzheimer's disease from brain MRI scans. The model can classify images into four categories:
- **No Impairment** - Healthy brain scans
- **Mild Impairment** - Early-stage cognitive decline
- **Moderate Impairment** - Moderate cognitive decline  
- **Alzheimer** - Advanced Alzheimer's disease (formerly "Very Mild Impairment")

## Dataset

The project uses the "Best Alzheimer MRI Dataset" from Kaggle, which contains:
- **Training Images**: 10,240 images across 4 classes
- **Test Images**: 1,279 images across 4 classes
- **Image Format**: RGB images resized to 150x150 pixels

Dataset source: `lukechugh/best-alzheimer-mri-dataset-99-accuracy`

## Model Architecture

The CNN model consists of:
- **4 Convolutional layers** with ReLU activation (32, 64, 128, 128 filters)
- **4 MaxPooling layers** for feature reduction
- **Flatten layer** to convert 2D features to 1D
- **Dense layer** with 512 neurons and ReLU activation
- **Dropout layer** (0.5) for regularization
- **Output layer** with softmax activation for 4-class classification

## Performance Metrics

### Overall Performance
- **Accuracy**: 90%
- **Training completed in 5 epochs**

### Class-wise Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Mild Impairment | 0.78 | 0.91 | 0.84 |
| Moderate Impairment | 0.99 | 0.85 | 0.91 |
| No Impairment | 1.00 | 1.00 | 1.00 |
| Alzheimer | 0.87 | 0.85 | 0.86 |

## Requirements

```python
tensorflow>=2.12.0
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
kagglehub
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd alzheimer-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset using KaggleHub:
```python
import kagglehub
path = kagglehub.dataset_download("lukechugh/best-alzheimer-mri-dataset-99-accuracy")
```

## Usage

### Training the Model

1. **Data Loading and Preprocessing**:
   - Images are loaded and resized to 150x150 pixels
   - Pixel values are normalized to [0, 1] range
   - Labels are encoded using one-hot encoding

2. **Data Augmentation**:
   - Rotation, shifting, shearing, and zooming applied
   - Horizontal and vertical flipping enabled

3. **Model Training**:
```python
# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32,
    verbose=1
)
```

4. **Model Evaluation**:
   - Classification report generation
   - Confusion matrix visualization
   - Sample predictions display

### Making Predictions

```python
# Load the trained model
model = tf.keras.models.load_model('alzheimer_model.h5')

# Make predictions on new images
predictions = model.predict(new_images)
predicted_classes = np.argmax(predictions, axis=1)
```


## Key Features

- **High Accuracy**: Achieved 90% classification accuracy
- **Data Augmentation**: Robust training with various image transformations
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Pre-trained Model**: Saved model ready for deployment
- **Visualization Tools**: Training plots, confusion matrix, and sample predictions

## Data Preprocessing

- **Image Resizing**: All images standardized to 150x150 pixels
- **Normalization**: Pixel values scaled to [0, 1] range
- **Data Augmentation**: Applied rotation, shifting, shearing, zooming, and flipping
- **Train-Validation-Test Split**: 70%-15%-15% split for robust evaluation

## Training Results

The model showed excellent convergence:
- **Training Accuracy**: Improved from 33.8% to 93.1% over 5 epochs
- **Validation Accuracy**: Reached 91.4% with minimal overfitting
- **Best Performance**: Achieved on "No Impairment" class (100% precision and recall)

## Future Improvements

- Implement transfer learning with pre-trained models (ResNet, VGG, etc.)
- Add more sophisticated data augmentation techniques
- Experiment with ensemble methods
- Deploy the model as a web application
- Add gradual learning rate scheduling
- Implement cross-validation for more robust evaluation

## Medical Disclaimer

This project is for educational and research purposes only. The model should not be used for actual medical diagnosis without proper validation and approval from medical professionals.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- Dataset provided by Luke Chugh on Kaggle
- TensorFlow and Keras teams for the deep learning framework
- Medical imaging community for advancing AI in healthcare
