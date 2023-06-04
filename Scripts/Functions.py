


import numpy as np
import os
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_curve, precision_recall_curve, confusion_matrix
from scipy.io import savemat
import csv
from matplotlib import pyplot
from tensorflow.keras import backend as K
#----------


def Precision_Recall_F1score_Fun(labels, probabilities, save_to_dir, ClassName, print_results, threshold=0.5):
    '''
    :param labels: array N*C, where N is number of frames and C is number of classes
    :param probabilities: array N*C, where N is number of frames and C is number of classes
    :param save_to_dir: "c/../" or None. If a directory is passed, Average Precision is saves into csv file.
                                         If None is passed, the Average Precision is not saved.
    :param ClassName:is a list containing names of the classes.
    :param print_results: True/False. If True, print: precision, recall and f1-score for every class
    :return: precision, recall and f1-score for every class. Shape of each array is
    '''

    predictions=np.greater_equal(probabilities, threshold)  # convert it to True and False using threshold of 0.5

    PRF1 = []

    PRF1.append(precision_recall_fscore_support(labels, predictions,average='binary', pos_label= 0)) # AP of False class ('Barrett')
    PRF1.append(precision_recall_fscore_support(labels, predictions, average='binary',
                                                pos_label=1))  # AP of True class ('Inflammation' )

    if save_to_dir!=None:
        # save precision, recall and F1 score results to csv file
        os.makedirs(os.path.dirname(save_to_dir + 'PRF1.csv'), exist_ok=True)
        with open(save_to_dir + 'PRF1.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=';')
            wr.writerow(['', 'Precision', 'Recall', 'F1 Score', 'occurrences'])
            mean_precision = 0
            mean_recall = 0
            mean_F1score = 0
            for i in range(len(PRF1)):
                wr.writerow([ClassName[i], PRF1[i][0], PRF1[i][1], PRF1[i][2], PRF1[i][3]])
                mean_precision = mean_precision + PRF1[i][0]
                mean_recall = mean_recall + PRF1[i][1]
                mean_F1score = mean_F1score + PRF1[i][2]
            wr.writerow(['Mean', mean_precision / len(PRF1), mean_recall / len(PRF1), mean_F1score / len(PRF1)])
            wr.writerow('')
            wr.writerow('')

    if print_results==True:
        print('precision, recall, f1score, occurrences')
        class_num= 0
        for re in PRF1:
           print( ClassName[class_num] )
           print(re)
           class_num = class_num+1

    # save  PRF1 data as .mat
    PRF1_array=np.array(PRF1)

    # return precision, recall and f1-score for every class
    return PRF1_array[:,0], PRF1_array[:,1], PRF1_array[:,2]
# ------------------
def Average(lst):
    return sum(lst) / len(lst)
# ------
# ---------------------------
def AveragePrecision_Fun(labels, probabilities, save_to_dir, ClassName, print_results):
    '''
    :param labels: array N*C, where N is number of frames and C is number of classes (usually 7 surgical tools)
    :param probabilities: array N*C, where N is number of frames and C is number of classes (usually 7 surgical tools)
    :param save_to_dir: "c/../" or None. If a directory is passed, Average Precision is saved into csv file.
                                         If None is passed, the Average Precision is not saved.
    :param ClassName:is a list containing names of the classes. e.g. ClassName = ['Barrett', 'Inflammation']
    :param print_results: True/False. If True, print Average Precision for every class
    :return: Average Precision for every class
    '''
    AP = []
    AP.append(average_precision_score(labels, probabilities,pos_label=0) ) # AP of False class ('Barrett')
    AP.append(average_precision_score(labels, probabilities, pos_label=1)) # AP of True class ('Inflammation' )

    if save_to_dir != None:

        # save Ap results to csv file
        os.makedirs(os.path.dirname(save_to_dir  + 'AveragePrecision.csv'), exist_ok=True)
        with open(save_to_dir + 'AveragePrecision.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=';')
            for i in range(len(AP)):
                wr.writerow([ClassName[i], AP[i]])

            wr.writerow(['Mean', Average(AP)])
            wr.writerow('')
            wr.writerow('')


    # save  AveragePrecision as .npy.
    np.save(save_to_dir + 'AveragePrecision.npy', np.array(AP))

    if print_results == True:
        class_num = 0
        for ap in AP:
            #if ((ClassName != None) and (len(ClassName) == NumClasses)):
            print('Average Precision - %s : %f' % (ClassName[class_num], ap))
            #else:
             #   print('Average Precision - Class%d : %f' % (class_num, ap))
            class_num = class_num + 1;

        print('mean Average Precision for all classes: %f' % (Average(AP)))

    return np.array(AP)
# -----------------------------------

# ---------------------------------------------------------------
def evaluate_my_Model(model, epoch, test_generator, ClassName, saveToDir):

    steps_test = int(np.ceil(
        test_generator.samples / test_generator.batch_size))

    # ----- get images and labels of testing data
    test_generator.reset()
    Labels =  np.empty((0,), int)
    image_shape = test_generator.image_shape
    test_images = np.empty((0, image_shape[0],image_shape[1],image_shape[2]), 'float32')
    for step in range(steps_test):
        batch_images, batch_labels= test_generator.next()
        Labels = np.append(Labels, batch_labels, axis=0)
        test_images = np.append(test_images, batch_images, axis=0)

    Labels = np.expand_dims(Labels, axis=1)
    Labels = Labels.astype(bool)

    # ----- get Probabilities and predictions of testing data
    Probabilities_testVideos = model.predict(test_images, verbose=1)
    Probabilities = np.array(Probabilities_testVideos)
    np.save(saveToDir + 'Probabilities.npy', Probabilities)
    savemat(saveToDir + 'Probabilities.mat', {'Probabilities': Probabilities}, appendmat=False)

    threshold = 0.5
    predictions = np.greater_equal(Probabilities, threshold)

    # create a directors for saveing resulta of teh corresponding epoch
    saveToDir = saveToDir + '/'
    if (not os.path.exists(saveToDir)):
        os.mkdir(saveToDir)

    saveToDir = saveToDir + '/' + str(epoch).zfill(2) + '_Epoch' + '/'
    if (not os.path.exists(saveToDir)):
        os.mkdir(saveToDir)

    # ------  Calculate Evaluation Metrics

    # Compute tn, fp, fn and tp
    tn, fp, fn, tp = confusion_matrix(Labels, predictions).ravel()
    print('tn = %d' % tn)
    print('fp = %d' % fp)
    print('fn = %d' % fn)
    print('tp = %d' % tp)
    np.save(saveToDir + 'tn_fp_fn_tp.npy', np.array([tn, fp, fn, tp]))
    os.makedirs(os.path.dirname(saveToDir + 'tn_fp_fn_tp.csv'), exist_ok=True)
    with open(saveToDir + 'tn_fp_fn_tp.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=';')
        wr.writerow([ 'tn', 'fp', 'fn', 'tp'])
        wr.writerow([tn, fp, fn, tp])

    # Compute average precision
    AP = AveragePrecision_Fun(Labels, Probabilities, saveToDir , ClassName, print_results=True)

    # Compute precision, recall and F1-score
    PRF = Precision_Recall_F1score_Fun(Labels, Probabilities, saveToDir, ClassName, print_results=True)

    return AP, PRF


# -----------------------------------------------------------------------

# --------------------- plot loss during training
def loss_plot(history, save_figure_dir):
    #training_tool_loss=[]
    #training_tool_loss+= [history.history[loss_name] for loss_name in history.model.metrics_names[1:len(history.model.targets) + 1]]
    for metrics_name in history.model.metrics_names: # [1:len(history.model.targets) + 1]
        pyplot.figure()
        pyplot.title('Training: '+ metrics_name)
        pyplot.plot(history.history[metrics_name], label=metrics_name)
        pyplot.plot(history.history['val_'+metrics_name], label= 'val_'+metrics_name)
        pyplot.legend()
        pyplot.savefig(save_figure_dir+metrics_name+'.png', bbox_inches='tight')
        pyplot.close()


# --------------------------------------------

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
