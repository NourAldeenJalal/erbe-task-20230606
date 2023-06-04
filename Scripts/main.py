
# ---------- Imports and setup
import os
import tensorflow as tf
import numpy as np
from scipy.io import savemat

import ReadData
import Build_Model
import Functions


# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# -------------  Data Directories
Project_dir = '....../Project'

Label_file_dir = Project_dir + '/Labels/labels.csv'
Images_dir = Project_dir + '/images/images/'
Trained_CNN_Model_dir = Project_dir + '/Pretrained Model/ResNet50/final_model.h5'

# ------------- Training Parameters
Batch_size = 20
Epochs = 200
image_size = (224,224)
learning_rate = 1e-3
decay=0.0001

#----------------- Create directory  for saving results
cwd = os.getcwd()  # Get the current working (CWD)
saveToDir = cwd+'/Results/'
if (not os.path.exists(saveToDir)): os.mkdir(saveToDir)

loss_history_dir = saveToDir + 'loss_history'

# ----------------------
# Read data and create data generator
train_generator, test_generator, weight_for_Barrett, weight_for_Inflammation = ReadData.Data_Generator(Label_file_dir, Images_dir, Batch_size, image_size )


# Build the model
model = Build_Model.Create(Trained_CNN_Model_dir)

#  -----------------  Run the experiment

# --------------------------- Callback class
class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        # save loss
        numpy_loss_history = np.array(self.losses)
        np.savetxt(loss_history_dir + '.txt', numpy_loss_history, delimiter=",")
        savemat(loss_history_dir + '.mat', {'numpy_loss_history': numpy_loss_history}, appendmat=False)

        # Evaluate at the end of every training epoch
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch+1, keys))
        Functions.evaluate_my_Model(self.model, epoch, test_generator,  list(test_generator.class_indices.keys()), saveToDir)


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=decay)

model.compile(optimizer, loss='binary_crossentropy'
              , metrics=['accuracy', Functions.f1_m]  # , metrics=['acc',f1_m,precision_m, recall_m]
              )

historycallback = CustomCallback()

history = model.fit(
    train_generator,
    steps_per_epoch=np.ceil(train_generator.samples / train_generator.batch_size),
    epochs=Epochs,
    shuffle=True,
    validation_data=test_generator,
    validation_steps=test_generator.samples / test_generator.batch_size,
    verbose=1, callbacks=[ historycallback ]
    #,class_weight = {0: weight_for_Barrett, 1: weight_for_Inflammation}
                      )


numpy_loss_history = np.array(historycallback.losses)
np.savetxt(loss_history_dir+'.txt', numpy_loss_history, delimiter=",")
savemat(loss_history_dir + '.mat', {'numpy_loss_history': numpy_loss_history}, appendmat=False)


saveToDir_loss_figures =  saveToDir+'/Loss Figures/'
if (not os.path.exists(saveToDir_loss_figures)): os.mkdir(saveToDir_loss_figures)

Functions.loss_plot(history, saveToDir_loss_figures)



d = 9