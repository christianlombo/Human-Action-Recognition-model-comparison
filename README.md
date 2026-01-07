# Human Activity Recognition (HAR) Model Comparison

## Project Overview
With the rise of wearable technology (smartwatches, fitness trackers), identifying human activity from raw sensor data is a critical problem in Digital Health. This project builds a system that takes smartphone accelerometer data (X, Y, Z movement) and classifies the user's action into one of six categories: **Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, or Laying.**

Unlike standard implementation projects, this repository focuses on an Algorithm Showdown: benchmarking a Classical Machine Learning model (Support Vector Machine) against a Deep Learning model (Multi-Layer Perceptron) to determine which architecture handles high-dimensional sensor data better.

---

## Technology Stack & Dataset

### Tech Stack
* **Language:** Python 3.13
* **Algorithms:** Support Vector Machine, Multi-Layer Perceptron.
* **Libraries:** Scikit-Learn, Pandas, NumPy, Seaborn .
* **Preprocessing:** `StandardScaler`.

### The Dataset
* **Source:** [Human Activity Recognition with Smartphones](https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones) (UCI Machine Learning Repository).
* **Input:** 561 extracted features from accelerometer and gyroscope readings.
* **Target:** 6 Activity Classes.

---

## How to Run
- Run the Experiment
- Execute the comparison script. This will load the data, scale it, train both models, and generate a performance graph.
- python Models/compare_models.py

---

## Methodology: 
1. Data Scaling (StandardScaler)
Both SVM and Neural Networks are mathematically sensitive to the scale of input data. If one sensor reads "100" (gravity) and another reads "0.1" (noise), the model will be biased.

- Solution: I applied StandardScaler to normalize all features to a mean of 0 and a standard deviation of 1, ensuring fair competition between the algorithms.

2. The Model Contenders
- Model A: Support Vector Machine (SVM): Used a linear kernel. SVMs are excellent at finding a "hyperplane" that divides classes in high-dimensional space.

- Model B: Neural Network (MLP): A Multi-Layer Perceptron with two hidden layers (100, 50). This represents a Deep Learning approach capable of learning non-linear feature representations.

## Results & Analysis
Winner: Support Vector Machine (SVM) (96% Accuracy).

Critical Analysis: Why did the "older" SVM beat the modern Neural Network?

Feature Engineering: The dataset provided by UCI already contained expertly engineered features (Fourier transforms, etc.). When features are this well-defined, a linear separator (SVM) is often more efficient and accurate than a Neural Network, which tries to relearn patterns from scratch.

Overfitting Risk: Neural Networks require massive amounts of data to generalize well. On this smaller dataset (~7,000 samples), the MLP is prone to overfitting, whereas the SVM is more robust.

Key Takeaway: Complexity does not always equal performance. For structured, high-dimensional tabular data, classical algorithms often outperform deep learning.

## Future Improvements
Hyperparameter Tuning: Use GridSearchCV to optimize the hidden layers of the MLP to see if it can beat the SVM.

Raw Signal Processing: Instead of using the pre-calculated features, build a 1D-CNN (Convolutional Neural Network) to learn directly from the raw wave signals.
