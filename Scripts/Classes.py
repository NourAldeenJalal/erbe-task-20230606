import tensorflow
import time
import Functions

# ----------------- Classes
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch+1, keys))
        Functions.evaluate_my_Model(self.model, epoch, test_generator,  list(test_generator.class_indices.keys()), saveToDir)

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        numpy_loss_history = np.array(self.losses)
        np.savetxt(loss_history_dir + '.txt', numpy_loss_history, delimiter=",")
        savemat(loss_history_dir + '.mat', {'numpy_loss_history': numpy_loss_history}, appendmat=False)

