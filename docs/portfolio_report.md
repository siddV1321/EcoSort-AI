# EcoSort AI — Project Report

**Student:** Siddharth Verma  
**Class:** X  
**Session:** 2026–27  
**Subject area:** Artificial Intelligence

## 1. Introduction
EcoSort AI explores how image classification can assist in recognizing common waste categories.

## 2. Problem statement
Incorrect waste segregation can make recycling more difficult. This project tests whether an image classifier can predict the category of a waste image.

## 3. Objectives
- Prepare and inspect a labeled waste-image dataset.
- Train a transfer-learning image classifier.
- Evaluate it on held-out validation data.
- Compare this Python experiment with a separate Orange Data Mining workflow.
- Build an interactive image-prediction interface.

## 4. Dataset
Source: https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset

Record the exact number of images in each class after running `src/inspect_dataset.py`. The expected classes are cardboard, glass, metal, paper, plastic, and trash; verify these against the downloaded files.

## 5. Tools
Orange Data Mining, Python 3.10, TensorFlow/Keras, Streamlit, Pillow, NumPy, Matplotlib, and scikit-learn.

## 6. Methodology
The Orange workflow uses image embeddings followed by Logistic Regression and Random Forest. The Python implementation is a separate MobileNetV2 transfer-learning experiment. Dataset splitting and metrics for each experiment must be reported separately.

## 7. Results
Fill this section only after running the project:
- Python validation accuracy:
- Macro precision:
- Macro recall:
- Macro F1:
- Best-performing classes:
- Most-confused classes:
- Orange Logistic Regression metrics:
- Orange Random Forest metrics:

## 8. Limitations
Predictions depend on image quality and the training dataset. Similar-looking or mixed-material objects can be misclassified. Validation results do not guarantee real-world performance, and local recycling rules differ.

## 9. Conclusion
Write a conclusion after reviewing the actual metrics and confusion matrix. Do not invent performance values.

## 10. References
- Kaggle Trash Type Image Dataset: https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset
- TensorFlow: https://www.tensorflow.org/
- Orange Data Mining: https://orangedatamining.com/docs/
