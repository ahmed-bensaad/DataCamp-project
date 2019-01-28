import numpy as np 
from sklearn.base import BaseEstimator
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense,Dropout, Flatten
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import functools
import tensorflow as tf
import cv2 


class Classifier(BaseEstimator):
    def __init__(self, n_epochs = 15, batch_size = 1, lr = 1e-3):
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.model = create_model()

    def fit(self, X, y):

        X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2)
        ntrain_samples = X_train.shape[0]
        nval_samples = X_val.shape[0]
        validation_steps = nval_samples / self.batch_size 
        steps_per_epoch = ntrain_samples / self.batch_size
        validation_steps = nval_samples / self.batch_size

        self.model.fit_generator(data_generator(X_train ,y_train,self.batch_size),
                                steps_per_epoch=steps_per_epoch, epochs = 10, 
                                validation_data =data_generator(X_val ,y_val,self.batch_size),
                                 validation_steps = validation_steps )

        

    def predict(self, X):
        img = [cv2.imread(path) for path in X]
        return self.model.predict_classes(img)

    def predict_proba(self, X):
        img = [cv2.imread(path) for path in X]
        return self.clf.predict(X)

# Defining model

def create_model(input_size = (128,128,3), epochs = 15 , lr = 1e-3 ):
    model = Sequential()
    model.add(Conv2D(64, input_shape= input_size, kernel_size=5, padding="same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Conv2D(80, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Conv2D(100, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Conv2D(150, kernel_size=3, padding="valid", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(43, activation='softmax'))
    decay_rate = lr / epochs
    opt = Adam(lr=0.001, decay=decay_rate, amsgrad=True)
    model.compile(loss='categorical_crossentropy',optimizer = opt, metrics=['accuracy'])
    return model

# Data Generator
def data_generator(anns_x, anns_y, batch_size = 100, target_size = (128,128), classes = 43):
    n_samples = anns_x.shape[0]
    while True:

        for i in range(0, n_samples, batch_size):
            imgs = [cv2.imread(file) for file in anns_x.iloc[i:i+batch_size]]
            imgs_resized = [cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR) for img in imgs]
            X_batch = [img for img in imgs_resized]
            Y_batch = anns_y.iloc[i:i+batch_size]
            yield np.array(X_batch), to_categorical(np.array(Y_batch),num_classes=43)
