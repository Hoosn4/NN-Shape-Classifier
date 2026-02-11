# NN-Shape-Classifier

## 📋 Overview
This project implements a **Feedforward Neural Network** to solve a binary classification problem. The model is trained to determine if a point $(x, y)$ in a 2D plane falls inside a specific rectangular boundary defined by coordinates derived from a Student ID.

## 🧠 The "Idea"
The core concept is **Non-Linear Classification**. By using a hidden layer with non-linear activation functions, the Neural Network learns to approximate the mathematical "boundary" of a rectangle. This demonstrates how Deep Learning can be used for geometric pattern recognition.

## 🚀 Features
* **Dynamic Boundary Generation**: Uses a KFUPM Student ID to generate unique rectangle vertices ($P1$ and $P2$).
* **TensorFlow/Keras Integration**: Built using a `Sequential` model with optimized layers.
* **Performance Monitoring**: Designed to achieve >90% accuracy within 100 epochs.
* **Data Visualization**: Uses `matplotlib` to plot the decision boundary and the classified points (Inside vs. Outside).

## 🛠️ Technical Details
* **Frameworks**: TensorFlow, Keras, Scikit-learn (for data splitting).
* **Model Architecture**: 
    * Input layer for 2D coordinates.
    * Hidden layer(s) with `relu` activation.
    * Output layer with `sigmoid` activation for binary probability.
* **Optimization**: Compiled with the `adam` optimizer and `binary_crossentropy` loss function.

## 📁 File Structure
* `COE292-HW2.py`: The complete pipeline including point generation, model training, and visualization.

## 📊 Results
The script generates a visualization showing:
1.  The blue rectangular boundary.
2.  Red/Green points representing the model's classification accuracy.
