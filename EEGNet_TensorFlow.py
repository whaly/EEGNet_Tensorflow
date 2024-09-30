import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dropout, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy, AUC, Recall, Precision
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
import numpy as np

class EEGNet(tf.keras.Model):
    def __init__(self,dropoutRate=0.25):
        super(EEGNet, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = Conv2D(16, (1, 64), padding='same', activation='elu', input_shape=(120, 64, 1))
        self.batchnorm1 = BatchNormalization()
        self.dropout1 = Dropout(dropoutRate)

        # Layer 2
        self.conv2 = Conv2D(4, (2, 32), strides=(1, 2), padding='same', activation='elu')
        self.batchnorm2 = BatchNormalization()
        self.pooling2 = MaxPooling2D(pool_size=(2, 2), padding='same')
        self.dropout2 = Dropout(dropoutRate)

        # Layer 3
        self.conv3 = Conv2D(4, (8, 4), strides=(1, 1), padding='same', activation='elu')
        self.batchnorm3 = BatchNormalization()
        self.pooling3 = MaxPooling2D(pool_size=(2, 2), padding='same')
        self.dropout3 = Dropout(dropoutRate)

        # Fully Connected Layer
        self.flatten = Flatten()
        self.fc1 = Dense(1, activation='sigmoid')
        

    def call(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.pooling2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)
        x = self.pooling3(x)

        # FC Layer
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# Instantiate the model
net = EEGNet()
net.build((None, 120, 64, 1))
net.summary()

# Compile the model
net.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy', AUC(), Precision(), Recall()])

# Define the `evaluate` function
def evaluate(model, X, Y, params=["acc"]):
    results = []
    batch_size = 100

    predicted = model.predict(X, batch_size=batch_size)

    for param in params:
        if param == "acc":
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2*precision*recall / (precision + recall))
    return results

# Generate dummy data
X_train = np.random.rand(100, 120, 64, 1).astype('float32')
y_train = np.round(np.random.rand(100).astype('float32'))

X_val = np.random.rand(100, 120, 64, 1).astype('float32')
y_val = np.round(np.random.rand(100).astype('float32'))

X_test = np.random.rand(100, 120, 64, 1).astype('float32')
y_test = np.round(np.random.rand(100).astype('float32'))

# Train the model
batch_size = 32

for epoch in range(10):  # loop over the dataset multiple times
    print("\nEpoch ", epoch)
    
    # Fit the model on training data
    history = net.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_val, y_val))

    # Validation accuracy
    params = ["acc", "auc", "fmeasure"]
    print(params)
    print("Training Loss ", history.history['loss'][-1])
    print("Train - ", evaluate(net, X_train, y_train, params))
    print("Validation - ", evaluate(net, X_val, y_val, params))
    print("Test - ", evaluate(net, X_test, y_test, params))
