#TO DO
STUDENT_ID =  202253340 # Enter your 9-digit KFUPM Student ID

N_EPOCHS =  50 # Set N_EPOCHS to the number of epochs you wish to train your model such that it achieves the best accuracy (atleast > 90%)

# Design your NN model such that you should not need N_EPOCHS > 100

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

# (DO NOT CHANGE) Function to generate random points
def generate_points(num_points):
    """
    Generates random points in the range -100 to 100.
    """
    x = np.random.uniform(-100, 100, num_points)  # Generate random X coordinates
    y = np.random.uniform(-100, 100, num_points)  # Generate random Y coordinates
    return np.vstack((x, y)).T

# (DO NOT CHANGE) Static coordinates for the rectangle vertices based on your KFUPM ID#
P1 = (max(10,STUDENT_ID%100), max(10,int(STUDENT_ID%10000/100)))
P2 = (min(-10,-int(STUDENT_ID/10000000)), min(-10,-(int(STUDENT_ID/100000)%100)))


# (DO NOT CHANGE) Function to get ground truth classes of the points
def is_inside_rectangle(point, P1, P2):
    """
    Check if a point is inside the rectangle defined by P1 and P2.
    """
    x, y = point

    return (P2[0] <= x <= P1[0]) and (P2[1] <= y <= P1[1])


# (DO NOT CHANGE) Generate points and create labels
num_points = 1000
points = generate_points(num_points)

# (DO NOT CHANGE) Make labels for points based on the classification function
labels = np.array([1 if is_inside_rectangle(point, P1, P2) else 0 for point in points])

# (DO NOT CHANGE) Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(points, labels, test_size=0.2, random_state=42)


# TO DO

# Task 1: Build the NN model as per the instructions
def build_nn_model():
    """
    Build and return an NN model using TensorFlow/Keras.
    Hint: Use tf.Keras.Sequential(...)
    """
    model = model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),  # Input layer with 32 neurons
        tf.keras.layers.Dense(16, activation='relu'),  # Hidden layer with 16 neurons
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    return model # (DO NOT CHANGE)

# Task 2: Complete the function below to Train the NN model, as per the instructions
def train_nn_model(model, X_train, y_train, nepochs=1):
    """
    Compile and train the NN model.
    """
    # Compile the model. Hint: Use model.compile(...)
    # TO DO
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # Train the model
    model.fit(X_train, y_train, epochs=nepochs, batch_size=10, verbose=0)  # DO NOT CHANGE
    return model

# The code below is to build and train the NN model
nn_model = build_nn_model()
nn_model = train_nn_model(nn_model, X_train, y_train, nepochs=N_EPOCHS)

# Predict using the trained model
y_pred = nn_model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int).flatten()

# Check the accuracy
accuracy = np.mean(y_pred_class == y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Visualize the results
plt.figure(figsize=(8, 6))

# Plot the rectangle decision boundary
rectangle_x = [P1[0], P2[0], P2[0], P1[0], P1[0]]
rectangle_y = [P1[1], P1[1], P2[1], P2[1], P1[1]]
plt.plot(rectangle_x, rectangle_y, 'b-', label="Rectangle Shape", linewidth=2)

# Plot points and their classification using your trained model
for i, point in enumerate(X_test):
    if y_pred_class[i] == 1:  # Inside
        plt.scatter(point[0], point[1], c='green', edgecolor='black', label="Inside (1)" if i == 0 else "")
    else:  # Outside
        plt.scatter(point[0], point[1], c='red', edgecolor='black', label="Outside (0)" if i == 0 else "")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Desired NN Decision Boundary and Your NN's Classified Points")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
