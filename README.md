# ♻️Smart Waste Management System (Image Processing Based)

**Project:** Image Processing Based Smart Waste Management System

## Description

A project that classifies waste images into categories (e.g., glass, plastic, organic) using a CNN model trained with TensorFlow/Keras. It includes preprocessing, model training, evaluation, and model export for deployment.

## Features

* Image preprocessing (resizing, normalization, augmentation)
* CNN model built with TensorFlow/Keras
* Training and validation pipeline
* Model saving (`.h5` and conversion-ready for TFLite)
* Evaluation with accuracy and confusion matrix

## Requirements

```bash
python 3.8+
numpy
pandas
tensorflow
opencv-python
matplotlib
scikit-learn
jupyter
```

## Project Structure

```
├── data/                   # dataset split into class subfolders
├── notebooks/              # Jupyter notebooks (including this notebook)
├── src/                    # preprocessing, training, and utils scripts
├── models/                 # saved models (.h5, .tflite)
├── README.md
└── requirements.txt
```

## How to run

1. Clone the repo.
2. Prepare your dataset in `data/` with subfolders for each class (e.g., `plastic/`, `organic/`).
3. Install requirements: `pip install -r requirements.txt`.
4. Open the notebook `Code for model.ipynb` in `notebooks/` and run cells to preprocess and train.

### Training (script)

```
python src/train.py --data_dir data/ --epochs 20 --batch_size 32
```

### Convert .h5 to TFLite

```python
import tensorflow as tf
model = tf.keras.models.load_model('models/model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('models/model.tflite', 'wb').write(tflite_model)
```

## Notebook highlights

* Key imports detected:

  * `from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Reshape, LSTM, Dense`
  * `import tensorflow as tf`
  * `import numpy as np`
  * `import cv2`
  * `import os`

* Key config variables detected:

  * No global config variables detected. See notebook for dataset paths and parameters.

## Model & Training Notes

Model definition, training, and saving steps are in the notebook. Make sure to adapt dataset paths before running.

## Evaluation

* The notebook computes metrics and plots a confusion matrix and training history. Use these to analyze model performance and avoid overfitting.




