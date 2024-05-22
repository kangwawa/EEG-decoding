import numpy as np
import random
import graphviz
# EEGNet-specific imports
from EEGModels import EEGNet_SimAM
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.metrics import cohen_kappa_score
# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


from matplotlib import pyplot as plt

from mne.io import concatenate_raws, read_raw_egi, read_raw_bdf, read_raw_gdf, read_epochs_eeglab

from tensorflow.keras.utils import plot_model
import os
os.environ["PATH"]+=os.pathsep+'D:/graphviz/bin/'

import warnings
warnings.simplefilter("ignore")


"""
1. EEG data is acquired by EGI 256-channel device recording (data sampling rate is 250Hz)
2. the events are recognized by EEGLAB, so the data format is .set, and the preprocessing work such as epochs segmentation has been completed.
"""

Index_left = read_epochs_eeglab('subject/Index_left_raw_21_4-40_-1-2.set')
#Index_left.resample(sfreq=128)
Index_left = Index_left.get_data()

Index_right = read_epochs_eeglab('subject/Index_right_raw_21_4-40_-1-2.set')
#Index_right.resample(sfreq=128)
Index_right = Index_right.get_data()

Thumb_left = read_epochs_eeglab('subject/Thumb_left_raw_21_4-40_-1-2.set')
#Thumb_left.resample(sfreq=128)
Thumb_left = Thumb_left.get_data()

Thumb_right = read_epochs_eeglab('subject/Thumb_right_raw_21_4-40_-1-2.set')
#Thumb_right.resample(sfreq=128)
Thumb_right = Thumb_right.get_data()


type = 4

# Slicing the train dataset
train_Index_left = Index_left[0:160,:,:]
train_Index_right = Index_right[0:160,:,:]
train_Thumb_left = Thumb_left[0:160,:,:]
train_Thumb_right = Thumb_right[0:160,:,:]


X_train = np.concatenate((train_Index_left,
                          train_Index_right,
                          train_Thumb_left,
                          train_Thumb_right),axis=0)

# Setting up labels
train_Index_left_num = np.zeros(160,int)
train_Index_right_num  = np.ones(160,int)
train_Thumb_left_num = np.ones(160,int)*2
train_Thumb_right_num = np.ones(160,int)*3

Y_train =np.concatenate((train_Index_left_num,
                         train_Index_right_num,
                         train_Thumb_left_num,
                         train_Thumb_right_num),axis=0)
Y_train = np_utils.to_categorical(Y_train,type)

# Slicing the validation dataset
validate_Index_left = Index_left[160:200,:,:]
validate_Index_right = Index_right[160:200,:,:]
validate_Thumb_left = Thumb_left[160:200,:,:]
validate_Thumb_right = Thumb_right[160:200,:,:]


X_validate = np.concatenate((validate_Index_left,
                          validate_Index_right,
                          validate_Thumb_left,
                          validate_Thumb_right),axis=0)

validate_Index_left_num = np.zeros(40,int)
validate_Index_right_num  = np.ones(40,int)
validate_Thumb_left_num = np.ones(40,int)*2
validate_Thumb_right_num = np.ones(40,int)*3

Y_validate =np.concatenate((validate_Index_left_num,
                         validate_Index_right_num,
                         validate_Thumb_left_num,
                         validate_Thumb_right_num),axis=0)
Y_validate = np_utils.to_categorical(Y_validate,type)


# Slicing the test dataset
test_Index_left = Index_left[160:200,:,:]
test_Index_right = Index_right[160:200,:,:]
test_Thumb_left = Thumb_left[160:200,:,:]
test_Thumb_right = Thumb_right[160:200,:,:]


X_test = np.concatenate((test_Index_left,
                          test_Index_right,
                         test_Thumb_left,
                         test_Thumb_right),axis=0)
test_Index_left_num = np.zeros(40,int)
test_Index_right_num  = np.ones(40,int)
test_Thumb_left_num  = np.ones(40,int)*2
test_Thumb_right_num = np.ones(40,int)*3
Y_test =np.concatenate((test_Index_left_num,
                        test_Index_right_num,
                        test_Thumb_left_num,
                        test_Thumb_right_num),axis=0)
Y_test = np_utils.to_categorical(Y_test,type)

kernels, chans, samples = 1, 21, 750
X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

model = EEGNet_SimAM(nb_classes = 4, Chans = chans, Samples = samples,
               dropoutRate = 0.5, kernLength = 125, F1 = 8, D = 2, F2 = 16,
               dropoutType = 'Dropout')



model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])

numParams    = model.count_params()

checkpointer = ModelCheckpoint(filepath='tmp/checkpoint.h5', verbose=1,
                               save_best_only=True)

class_weights = {0:1, 1:1, 2:1, 3:1}

fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 300,
                        verbose = 2, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer], class_weight = class_weights)
model.load_weights('tmp/checkpoint.h5')

###########################################
probs       = model.predict(X_test)

preds       = probs.argmax(axis = -1)

Trues       = Y_test.argmax(axis=-1)
print(preds)

print(Y_test.argmax(axis=-1))
acc         = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

kappa = cohen_kappa_score(Trues, preds, weights='quadratic')

print("Kappa value:", kappa)


#confusion matrix
names        = ['Index_left', 'Index_right','Thumb_left','Thumb_right']
plt.figure(0)
plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')
plt.show()



# Plot train & validation accuracy values
plt.plot(fittedModel.history['accuracy'])

plt.plot(fittedModel.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(fittedModel.history['loss'])

plt.plot(fittedModel.history['val_loss'])


plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



plt.plot(fittedModel.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(fittedModel.history['val_accuracy'], label='Validation Accuracy', marker='o')


plt.plot(fittedModel.history['loss'], label='Training Loss', marker='o')
plt.plot(fittedModel.history['val_loss'], label='Validation Loss', marker='o')

plt.legend()


plt.xlabel('Epochs')
plt.ylabel('Accuracy & Loss')
plt.title('Model Accuracy and Loss')

plt.show()


#绘制模型图
plot_model(model, to_file='model.png')