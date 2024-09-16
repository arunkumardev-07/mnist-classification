 # Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Problem Statement: Handwritten Digit Recognition with Convolutional Neural Networks

Objective: Develop a Convolutional Neural Network (CNN) model to accurately classify handwritten digits (0-9) from the MNIST dataset.

Data: The MNIST dataset, a widely used benchmark for image classification, contains grayscale images of handwritten digits (28x28 pixels). Each image is labeled with the corresponding digit (0-9).

## Neural Network Model
![nueral network image](https://github.com/user-attachments/assets/87d136e3-925b-4fda-93bb-2c7937bcc1af)



## DESIGN STEPS

### STEP 1: Import libraries
### STEP 2: Load and preprocess data
### STEP 3: Define model architecture
### STEP 4: Compile the model
### STEP 5: Train the model
### STEP 6: Evaluate the model

## PROGRAM

### Name: Arunkumar.P
### Register Number: 212222040016
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.metrics import CategoricalCrossentropy
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train.shape

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

X_test_scaled.shape

model = keras.Sequential()

model.add(Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation="softmax"))

model.summary()

model.compile('adam', loss ='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_scaled, y_train_onehot, epochs=20, validation_data = (X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['loss','val_loss']].plot()

metrics[['accuracy','val_accuracy']].plot()

pred = np.argmax(model.predict(X_test),axis=1)

pred[0:10]

y_test[0:10]

print(type(y_test))
print(type(pred))
y_test = y_test.ravel()
pred=pred.ravel()

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))

img = image.load_img('/content/5.png')

type(img)

img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![dp 1](https://github.com/user-attachments/assets/169d7c7a-55a8-4d61-851a-02f45ffc8e3e)
![dp 2](https://github.com/user-attachments/assets/3db171c2-b9d6-49ed-a6a1-7291a2ed906e)


### Classification Report
![class report](https://github.com/user-attachments/assets/2d1b50ca-1fd0-40c8-9e76-af55c4c843cd)

### Confusion Matrix
![conf matrix](https://github.com/user-attachments/assets/e4d9924d-6ae4-4fa4-8c0e-a85fc239b8bd)


### New Sample Data Prediction
![download (1)](https://github.com/user-attachments/assets/df0cec52-1744-4c24-ac97-8d319a4bce93)


![mnist](https://github.com/user-attachments/assets/da2aa6cc-4f97-47b4-b4fb-877686db2c17)


## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
