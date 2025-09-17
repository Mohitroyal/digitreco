import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(len(X_train))
print(len(X_test))
print(X_train[0])
plt.matshow(X_test[1])
plt.show()
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train_flatten = X_train.reshape(len(X_train), 28*28)
print(X_train_flatten.shape)
X_test_flatten = X_test.reshape(len(X_test), 28*28)
print(X_test_flatten.shape)
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_flatten, y_train, epochs=5)
model.evaluate(X_test_flatten, y_test)
joblib.dump(model, 'mnist_model.pkl')
y_pred = model.predict(X_test_flatten)
y_pred_labels = [np.argmax(i) for i in y_pred]
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_labels)
print(con_mat)
plt.figure(figsize=(10,7))
sns.heatmap(con_mat, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()