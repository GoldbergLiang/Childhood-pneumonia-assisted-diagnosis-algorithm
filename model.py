import numpy as np
import pandas as pd
import pylab as plt
import cv2
import os
from sklearn.metrics import confusion_matrix
import mlxtend.plotting
from tqdm import tqdm
from IPython.display import clear_output
from keras.utils.np_utils import to_categorical
from keras.models import  Model
from keras.callbacks import Callback
from keras.optimizers import rmsprop
from keras.layers import Conv2D, SeparableConv2D, Dense, concatenate, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Input


def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            if wbc_type in ['NORMAL']:
                label = 0
            elif wbc_type in ['PNEUMONIA']:
                label = 1
            else:
                label = 2
            for image_filename in tqdm(os.listdir(folder + wbc_type)):
                img = cv2.imread(folder + wbc_type + '/' + image_filename)
                if img is not None:
                    img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
                    img_arr = np.asarray(img)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

X_train, y_train = get_data('D:/Childhood-pneumonia-assisted-diagnosis-algorithm/data/TRAIN/')
X_test, y_test = get_data('D:/Childhood-pneumonia-assisted-diagnosis-algorithm/data/TEST/')
print(y_train.shape,'\n',X_test.shape)
print(y_test.shape,'\n',y_test.shape)

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
y_trainHot = to_categorical(y_train, num_classes=2)
y_testHot = to_categorical(y_test, num_classes=2)

class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        clear_output(wait=True)

        ax1.set_yscale('Log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label="acc")
        ax2.plot(self.x, self.val_acc, label="val_acc")
        ax2.legend()

        plt.show()


plot = PlotLearning()

input_layer = Input((256, 256, 3))

hidden_layer1 = Conv2D(8, (3,3), activation='relu')(input_layer)
hidden_layer2 = SeparableConv2D(8, (3,3), activation='relu')(input_layer)
hidden_layer1 = Conv2D(8, (3,3), activation='relu')(hidden_layer1)
hidden_layer2 = SeparableConv2D(8, (3,3), activation='relu')(hidden_layer2)

hidden_layer1 = BatchNormalization()(hidden_layer1)
hidden_layer2 = BatchNormalization()(hidden_layer2)

hidden_layer1 = MaxPooling2D((3,3))(hidden_layer1)
hidden_layer2 = MaxPooling2D((3,3))(hidden_layer2)

hidden_layer1 = Conv2D(16, (3,3), activation='relu')(hidden_layer1)
hidden_layer2 = SeparableConv2D(16, (3,3), activation='relu')(hidden_layer2)
hidden_layer1 = Conv2D(16, (3,3), activation='relu')(hidden_layer1)
hidden_layer2 = SeparableConv2D(16, (3,3), activation='relu')(hidden_layer2)

hidden_layer1 = BatchNormalization()(hidden_layer1)
hidden_layer2 = BatchNormalization()(hidden_layer2)

hidden_layer1 = MaxPooling2D((2,2))(hidden_layer1)
hidden_layer2 = MaxPooling2D((2,2))(hidden_layer2)

hidden_layer1 = Conv2D(32, (5,5), activation='relu')(hidden_layer1)
hidden_layer2 = SeparableConv2D(32, (5,5), activation='relu')(hidden_layer2)
hidden_layer1 = Conv2D(32, (5,5), activation='relu')(hidden_layer1)
hidden_layer2 = SeparableConv2D(32, (5,5), activation='relu')(hidden_layer2)

hidden_layer1 = MaxPooling2D((4,4))(hidden_layer1)
hidden_layer2 = MaxPooling2D((4,4))(hidden_layer2)

hidden_layer = concatenate([hidden_layer1, hidden_layer2])

hidden_layer = GlobalAveragePooling2D()(hidden_layer)
hidden_layer = Dense(60, activation='sigmoid')(hidden_layer)
hidden_layer = BatchNormalization()(hidden_layer)

output_layer = Dense(2, activation='softmax')(hidden_layer)

model = Model(input_layer, output_layer)
model.compile(
    optimizer=rmsprop(0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()


model.fit(X_train, y_trainHot, validation_data=(X_test,y_testHot), batch_size=64, epochs=20, shuffle=True, callbacks=[plot])

scores = model.evaluate(X_test, y_testHot)
print(model.metrics_names, scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

#plot confusion_matrix
pred = model.predict(X_test)
pred = np.argmax(pred,axis = 1)
y_true = np.argmax(y_testHot,axis = 1)
CM = confusion_matrix(y_true, pred)
fig, ax = mlxtend.plotting.plot_confusion_matrix(conf_mat=CM, figsize=(5, 5))
plt.show()

