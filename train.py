import os, sys

from warnings import simplefilter 
simplefilter(action="ignore", category=FutureWarning)

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg, inception
import numpy as np
import argparse
from PIL import Image
import cv2
from random import shuffle
import sklearn.metrics as mt
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from shutil import rmtree
import time
import datetime
import pandas as pd
import seaborn as sn

TRAIN_LEN = 0
TEST_LEN = 0
STEPS_PER_TRAIN_EPOCH = 0

def confusion_matrix(data): # auxiliary function, builds a confusion matrix

    df_cm = pd.DataFrame(data, range(10), range(10))

    sn.set(font_scale=1.0) # for label size

    # create the confusion matrix
    ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap="Blues",fmt='d')
    ax.axes.set_title("Confusion Matrix" ,fontsize=18)

    # add some additional items
    plt.yticks(rotation=0)
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth")
    plt.tight_layout()

    # save the final figure
    plt.savefig(SAVE_DIR + "/confusion_matrix.png")

def prepare_data(directory): # auxiliary function, prepares the data for training and testing

    global TRAIN_LEN, STEPS_PER_TRAIN_EPOCH, TEST_LEN

    # read and shuffle the images' names
    raw_data = list(filter(lambda x : x[0]!=".",os.listdir(directory)))
    shuffle(raw_data)
    
    # ----------------------------------------------------------------------------------------
    # adjust, if necessary, the size of the raw data so that it divides nicely by "BATCH_SIZE"
    # ----------------------------------------------------------------------------------------
    remainder = len(raw_data)%BATCH_SIZE
    if(remainder!=0): raw_data = raw_data[:-remainder]

    if("train" in directory):
        TRAIN_LEN = len(raw_data)
        STEPS_PER_TRAIN_EPOCH = TRAIN_LEN//BATCH_SIZE

    else:
        TEST_LEN = len(raw_data)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # prepare an encoder to one hot encode the labels
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    base = np.asarray([0,1,2,3,4,5,6,7,8,9])
    enc = OneHotEncoder(handle_unknown='ignore')
    enc = enc.fit(base.reshape(-1,1))
    
    labels_aux = np.asarray(list(map(lambda x : int(x[0]),raw_data))).reshape(-1,1)

    # load every image into memory (the dataset is small, so it fits nicely)
    x = np.asarray(list(map(lambda x : np.asarray(Image.open(directory + "/" + x).convert("L").resize((IMAGE_SIZE,IMAGE_SIZE),Image.LANCZOS)),raw_data))).reshape(len(raw_data),IMAGE_SIZE,IMAGE_SIZE,1)

    # one hot encode the labels
    y = enc.transform(labels_aux).A

    # if needed, smooth the labels
    if("train" in directory): y = ((1-LABEL_SMOOTHING_ALPHA) * y) + (LABEL_SMOOTHING_ALPHA/10)
    
    return x, y

if __name__ == "__main__":

    #################################################################################################################################################################
    # PARSE THE ARGUMENTS
    #################################################################################################################################################################
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument("--dataset", required=False, help="Path to the dataset", default="./hand_gestures_dataset")
    args_parser.add_argument("--epochs", required=False, help="Number of training epochs", type=int, default=20)
    args_parser.add_argument("--lr", required=False, help="Learning rate", type=float, default=1e-4)
    args_parser.add_argument("--vgg_dropout_keep_prob", required=False, help="Dropout probability (to keep a neuron) for VGG19", type=float, default=0.5)
    args_parser.add_argument("--inception_dropout_keep_prob", required=False, help="Dropout probability (to keep a neuron) for InceptionV3", type=float, default=0.5)
    args_parser.add_argument("--model", required=False, help="Neural network to be used", choices=["vgg19","inceptionv3"], default="vgg19")
    args_parser.add_argument("--batch_size", required=False, help="Size of the training batches", type=int, default=64)
    args_parser.add_argument("--label_smoothing", required=False, help="How much label smoothing to use", type=float, default=0.1)
    args_parser.add_argument("--save_freq", required=False, help="How often the model is saved", type=int, default=1000)
    args_parser.add_argument("--save_dir", required=False, help="Directory where the model checkpoints will be saved", type=str, default="saved_models")

    args = vars(args_parser.parse_args())

    # ---------------------------------------------------------------
    # initialize global variables with the arguments
    # ---------------------------------------------------------------
    DATASET = args["dataset"]
    EPOCHS = args["epochs"]
    LR = args["lr"]
    VGG_DROPOUT_KEEP_PROB = args["vgg_dropout_keep_prob"]
    INCEPTION_DROPOUT_KEEP_PROB = args["inception_dropout_keep_prob"]
    MODEL = args["model"]
    BATCH_SIZE = args["batch_size"]
    LABEL_SMOOTHING_ALPHA = args["label_smoothing"]
    SAVE_FREQ = args["save_freq"]
    SAVE_DIR = args["save_dir"]
    if(MODEL=="inceptionv3"): IMAGE_SIZE = 299
    else: IMAGE_SIZE = 224

    # create a folder to save model checkpoints
    if(not os.path.exists(SAVE_DIR)): os.makedirs(SAVE_DIR)

    now = datetime.datetime.now()
    SAVE_DIR = SAVE_DIR + "/" + now.strftime("%m_%d_%Y, %H:%M:%S")
    os.makedirs(SAVE_DIR)
    
    ###################################################
    # PREPARE THE DATASETS
    ###################################################
    x_train, y_train = prepare_data(DATASET + "/train")
    x_test, y_test = prepare_data(DATASET + "/test")
    
    print("DONE PREPARING THE DATASET!")

    #######################################################################################################################################################################
    # PREPARE THE MODEL 
    #######################################################################################################################################################################
    x = tf.placeholder(shape=(None,IMAGE_SIZE,IMAGE_SIZE,1), dtype=tf.float32, name="x")
    y = tf.placeholder(shape=(None,10), dtype=tf.float32, name="y")
    is_training = tf.placeholder(dtype=tf.bool, name="is_training")

    # build the model's architecture
    with tf.variable_scope("model_definition") as scope:
        
        if(MODEL=="vgg19"): model_outputs, _ = vgg.vgg_19(inputs=x,num_classes=10,is_training=is_training,dropout_keep_prob=VGG_DROPOUT_KEEP_PROB,scope="vgg_19")
        else: model_outputs, _ = inception.inception_v3(inputs=x,num_classes=10,is_training=is_training,dropout_keep_prob=INCEPTION_DROPOUT_KEEP_PROB,scope="inception_v3")

        scope.reuse_variables()
    
    # define the loss/cost function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=model_outputs), name="loss")

    # declare an optimizer (i.e. how weights are updated)
    optimizer = tf.train.AdamOptimizer(learning_rate=LR, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)

    # prepare a metric to measure the accuracy
    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(y, 1), predictions=tf.argmax(model_outputs,1))
    
    print("DONE PREPARING THE MODEL!")

    # save the op
    tf.compat.v1.add_to_collection("prediction_op", model_outputs)

    with tf.Session() as sess:

        # initialize the local and global variables
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # create an op to save the model
        saver = tf.train.Saver(save_relative_paths=True)

        #################################################################################################################################################################################
        # MAIN LOOP
        #################################################################################################################################################################################
        plot_train_loss = []
        plot_train_acc = []

        plt.ion()

        print("STARTING TRAINING!")

        for i in range(EPOCHS): # iterate over the epochs
            
            train_losses_aux = []
            train_accuracies_aux = []
            
            t0 = time.time()

            ##########################################################################################################################################################################
            # TRAIN THE MODEL WITH THE TRAIN SET
            ##########################################################################################################################################################################
            for j in range(STEPS_PER_TRAIN_EPOCH): # iterate over the steps per train epoch (i.e. the number of weight updates per epoch)
                                
                # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
                # retrieve the train loss and accuracy
                # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
                batch_start = (j * BATCH_SIZE)
                batch_end = batch_start + BATCH_SIZE

                _, loss_aux, train_acc_aux  = sess.run([optimizer, loss, acc_op], feed_dict={x: x_train[batch_start:batch_end], y: y_train[batch_start:batch_end], is_training: True})

                # ----------------------------------------
                # save some performance metrics
                # ----------------------------------------
                train_losses_aux.append(loss_aux)
                train_accuracies_aux.append(train_acc_aux)

            epoch_time = time.time() - t0

            ##########################################################################################
            # PLOT THE TRAIN LOSS AND ACCURACY
            ##########################################################################################
            # get the mean of both metrics
            plot_train_loss.append(sum(train_losses_aux)/len(train_losses_aux))
            plot_train_acc.append(sum(train_accuracies_aux)/len(train_accuracies_aux))

            # ----------------------------------------------------------------------------------------
            # plot the figure
            # ----------------------------------------------------------------------------------------
            plt.clf()
            plt.title(MODEL.upper() + " Training")

            # plot the train loss
            plt.plot([j+1 for j in range(i+1)],plot_train_loss,c="g",linestyle="-",label="Train Loss")

            # plot the train accuracy
            plt.plot([j+1 for j in range(i+1)],plot_train_acc,c="g",linestyle="--",label="Train Acc.")

            plt.xlabel("Epoch")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="best")
            plt.grid(axis='both')

            plt.plot()

            # save the model
            if((i%SAVE_FREQ)==0): saver.save(sess, SAVE_DIR + "/" + MODEL + "_" + str(i+1))

            # if we are finished with training, save the final model
            elif((i%EPOCHS)==0): saver.save(sess, SAVE_DIR + "/" + MODEL + "_final_model")

            print("EPOCH " + str(i+1) + " (" + str(round(epoch_time,1)) + "s) | Train Loss: " + str(round(plot_train_loss[-1],3)) + " | Train Acc.: " + str(round(plot_train_acc[-1],3)))

        plt.savefig(SAVE_DIR + "/training.png")
        plt.close()

        ######################################################################################################################
        # EVALUATE THE MODEL'S PERFORMANCE ON THE TEST SET
        ######################################################################################################################
        total = 0.0
        predictions = []
        for i in range(TEST_LEN): # iterate over the test images

            pred, test_acc = sess.run([model_outputs, acc_op], feed_dict={x: [x_test[i]], y: [y_test[i]], is_training: False})
            
            total += test_acc
            predictions.append(np.squeeze(pred).tolist())

        print("\nTEST ACCURACY: " + str(round((total/TEST_LEN)*100.0,2)) + "%")

        ##############################################################
        # BUILD AND SAVE A CONFUSION MATRIX
        ##############################################################
        predictions = list(np.argmax(np.asarray(predictions), axis=1))
        y_true = list(np.argmax(np.asarray(y_test), axis=1))
        c_matrix = mt.confusion_matrix(y_true, predictions)

        confusion_matrix(c_matrix)