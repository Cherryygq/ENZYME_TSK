import argparse
import pandas as pd
import numpy as np
import os
import pickle

from keras_self_attention import SeqSelfAttention
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support,f1_score
import time
from keras.layers import Input

# from keras import backend as K


from sklearn.metrics import f1_score
#from getData import get_data

# check if user parameters are valid
parser = argparse.ArgumentParser(description='Run one DL models on protein dataset.')
parser.add_argument('--model', default='ABLE',
                    help='Name of the model to run it on. Must be one of CNN, GRU, LSTM, BILSTM, ABLE, DEEPEC')
parser.add_argument('-e', '--epochs', nargs='?', type=int, default=50, help='Number of epochs for training')
parser.add_argument('-b', '--batch', nargs='?', type=int, default=128, help='Batch size for training')
parser.add_argument('-l', '--lr', nargs='?', type=float, default=1e-4, help='Learning rate for Adam optimizer')
args = parser.parse_args()

if args.model not in ["ANN", "CNN", "GRU", "LSTM", "BILSTM", "ABLE", "DEEPEC"]:
    print("Model", args.model, "is not defined. Please make changes to dl_models.py and this file")
    exit(0)
print("Model", args.model)
## Keras/Tensorflow Imports and Setup
import keras
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical, np_utils
from keras.regularizers import l2
from tensorflow.compat.v1 import ConfigProto  # GPU
from imblearn.over_sampling import SMOTE, ADASYN
from keras import backend as K

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# LIMIT = 3 * 1024
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

# import all the neural network models
from models import get_dl_model

# Load Dataset
# X,y = get_data()

with open('./dataset/X.pickle', 'rb') as infile:
    X = pickle.load(infile)
    # print('X_train.shape', X.shape)


with open('./dataset/y.pickle', 'rb') as infile:
    y = pickle.load(infile)

# X = X[y != 7]
# y = y[y != 7]
for i,label in enumerate(y):
    y[i] = label - 1
print('X.shape:',X.shape)
print('y.shape:',y.shape)

kf = KFold(n_splits=5, shuffle=False)
# kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
# SAMPLING_METHODS = ["NONE", "SMOTE", "ADASYN"] # sampling options
SAMPLING_METHODS = ["SMOTE"]

if os.path.exists('./pickle/8_14_ours_' + args.model + "_" + SAMPLING_METHODS[0] + '_results.pickle'):
    with open('./pickle/8_14_ours_' + args.model + "_" + SAMPLING_METHODS[0] + '_results.pickle', 'rb') as f:
        all_results = pickle.load(f)
    resumed_run = True
    last_iter_fold = all_results[-1]['fold']
    last_iter_model = all_results[-1]['model']
    last_iter_sampling = all_results[-1]['sampling']
else:
    all_results = []
    resumed_run = False
    last_iter_fold = None
    last_iter_model = None
    last_iter_sampling = None

fold = 1
for train_index, test_index in kf.split(X, y):
    if resumed_run:
        if fold < last_iter_fold:
            # this fold has already been done, skip
            print("K Fold Cross Validation || Fold #", fold, "already done. Skipped.")
            fold += 1
            continue

    print("K Fold Cross Validation || Fold #", fold)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('X_train.shape', X_train.shape)
    print('X_test.shape', X_test.shape)
    print('y_train[0:20]:', y_train[0:20])
    print('y_test[0:20]:', y_test[0:20])

    # y_train_dummy = np_utils.to_categorical(y_train)

    for sampling_method in SAMPLING_METHODS:
        print("K Fold", fold, "sampling methods begin")

        if resumed_run:
            if SAMPLING_METHODS.index(last_iter_sampling) > SAMPLING_METHODS.index(sampling_method):
                print("Fold #", fold, ", sampling", sampling_method, "already done. Skipped.")
                continue
            elif SAMPLING_METHODS.index(last_iter_sampling) == SAMPLING_METHODS.index(sampling_method):
                print("Fold #", fold, ", sampling", sampling_method, "already done. Skipped.")
                resumed_run = False
                continue



        if sampling_method == "NONE":
            X_resampled = X_train
            y_resampled = np_utils.to_categorical(y_train)
            # y_resampled = y_train
            print('X_resampled.shape', X_resampled.shape)

        else:


            if sampling_method == "SMOTE":
                start_smote = time.time()
                # X_resampled, X_val, y_resampled, y_val = train_test_split(X_train, y_train, test_size=0.1,
                #                                                           stratify=y_train)
                num = y_train.shape[0]
                X_resampled, X_val = X_train[0:int(num * 0.9)], X_train[int(num * 0.9):num]
                y_resampled, y_val = y_train[0:int(num * 0.9)], y_train[int(num * 0.9):num]
                print('X_resampled.shape: ', X_resampled.shape)
                print('y_resampled.shape: ', y_resampled.shape)
                print('y_resampled[0:20]:', y_resampled[0:20])
                print('X_val.shape: ', X_val.shape)
                print('y_val.shape: ', y_val.shape)
                print('y_val[0:20]:', y_val[0:20])
                y_val = np_utils.to_categorical(y_val)
                X_resampled = np.reshape(X_resampled, newshape=(X_resampled.shape[0], 300))
                print("Sampling with SMOTE begin")
                X_resampled, y_resampled = SMOTE(random_state=1, n_jobs=3).fit_resample(X_resampled, y_resampled)
                X_resampled = np.reshape(X_resampled, newshape=(X_resampled.shape[0], 3, 100))
                y_resampled = np_utils.to_categorical(y_resampled)

                all_data_current_fold = [X_resampled, y_resampled, X_val, y_val, X_test, y_test]
                # with open(CACHED_FOLD_FILE, 'wb') as handle:
                #     pickle.dump(all_data_current_fold, handle)

                all_data_current_fold = None

                print("SMOTE complete in %.2f seconds" % (time.time() - start_smote))

            elif sampling_method == "ADASYN":
                start_adasyn = time.time()
                X_resampled, X_val, y_resampled, y_val = train_test_split(X_train, y_train, test_size=0.1,
                                                                          stratify=y_train)
                y_val = np_utils.to_categorical(y_val)
                X_resampled = np.reshape(X_resampled, newshape=(X_resampled.shape[0], 300))
                print("Sampling with ADASYN begin")
                X_resampled, y_resampled = ADASYN(random_state=1, n_jobs=3).fit_resample(X_resampled, y_resampled)
                X_resampled = np.reshape(X_resampled, newshape=(X_resampled.shape[0], 3, 100))
                y_resampled = np_utils.to_categorical(y_resampled)
                print("ADASYN complete in %.2f seconds" % (time.time() - start_adasyn))

                all_data_current_fold = [X_resampled, y_resampled, X_val, y_val, X_test, y_test]


                all_data_current_fold = None

        start_train = time.time()
        model = get_dl_model(args.model)
        # out = get_dl_model(args.model)
        # model = Model(Input(shape=(3,100)), out)
        print(model.summary())
        opt = keras.optimizers.Adam(learning_rate=args.lr)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        keras_saved_file = "models/8_14_ours_" + args.model + "_" + sampling_method + "_fold" + str(fold) + ".h5"

        mcp_save = keras.callbacks.ModelCheckpoint(keras_saved_file, save_best_only=True, monitor='val_loss', verbose=2)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=2,
                                                      mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        callbacks_list = [reduce_lr, mcp_save]
        if sampling_method == "NONE":
            print("X_resampled: ",X_resampled.shape)
            history = model.fit(X_resampled, y_resampled, batch_size=args.batch, epochs=args.epochs,
                                validation_split=0.1, shuffle=False, callbacks=callbacks_list)

        else:
            print("Sampled train dataset: ", X_resampled.shape, y_resampled.shape)
            print("Validation dataset: ", X_val.shape, y_val.shape)
            history = model.fit(X_resampled, y_resampled, batch_size=args.batch, epochs=args.epochs,
                                validation_data=(X_val, y_val), shuffle=True, callbacks=callbacks_list)

        model = load_model(keras_saved_file, custom_objects={'SeqSelfAttention': SeqSelfAttention})
        end_train = time.time()

        # layer_output = model.get_layer(name="fea_256").output
        layer_output2 = model.get_layer(name="last_6").output
        layer_input = model.input
        # get_layer_output = K.function([layer_input], [layer_output])
        get_layer_output2 = K.function([layer_input], [layer_output2])
        # train_fea256 = np.array(get_layer_output(X_train)[0])
        train_fea6 = np.array(get_layer_output2(X_train)[0])
        # print(train_fea6.shape)

        y_pred = model.predict(X_test)
        # test_fea256 = np.array(get_layer_output(X_test)[0])
        y_labels = np.argmax(y_pred, axis=1)
        end_test = time.time()
        np.savez_compressed('./p6_npz_8_14/p6_8_14_ours_fold' + str(fold) + '.npz', train_fea6=train_fea6,
                            train_y=y_train, test_fea6=y_pred, test_y=y_test)
        # np.savez_compressed('./p6_npz_8_14/p6_mat_ABLE_fold' + str(fold) + '_.npz', train_fea6=train_fea6,
        #                     train_fea256=train_fea256, train_y=y_train, test_fea6=y_pred, test_fea256=test_fea256,
        #                     test_y=y_test)  # ABLE
        print("train_fea.shape", train_fea6.shape)
        print("y_train.shape", y_train.shape)
        print("test_fea.shape", y_pred.shape)
        print("test_y.shape", y_test.shape)


        # dump raw predictions
        Y_SAVE_LOCATION_PREFIX = "./results/dl/" + args.model + "_" + sampling_method + "_" + str(fold) + "_" + str(
            args.epochs) + "_" + str(args.batch)
        # np.save(Y_SAVE_LOCATION_PREFIX + "_PRED.npy", y_labels)
        # np.save(Y_SAVE_LOCATION_PREFIX + "_TRUE.npy", y_test)

        # print(f1_score(y_test, y_pred, average='macro'))
        # print(f1_score(y_test, y_pred, average='weighted'))
        print(precision_recall_fscore_support(y_test, y_labels, average='macro'))
        print(precision_recall_fscore_support(y_test, y_labels, average='weighted'))
        print(classification_report(y_test, y_labels, output_dict=True))


        # Metrics
        filename = "./results/dl/" + "8_14_ours_" + args.model + "_" + sampling_method + "_fold" + str(fold) + "_" + str(
            args.epochs) + "_" + str(args.batch) + ".npy"

        results_dict = {
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "initial_lr": args.lr,
            "fold": fold,
            "train_examples": X_resampled.shape[0],
            "sampling": sampling_method,
            "confusion_matrix": confusion_matrix(y_test, y_labels),
            "report": classification_report(y_test, y_labels, output_dict=True),
            "train_time": end_train - start_train,
            "test_time": end_test - end_train,
            "filename": filename
        }

        #buyao
        np.save(filename, history.history)
        print("Model", args.model, "done running on fold", fold, "with sampling strategy", sampling_method, "in",
              round(time.time() - start_train, 3), "s")


        all_results.append(results_dict)
        with open('./pickle/8_14_ours_' + args.model + "_" + sampling_method + '_results.pickle', 'wb') as handle:
            pickle.dump(all_results, handle)

        # buyao
        print("Model", args.model, "on fold", fold, "with sampling strategy", sampling_method, "completed in total of",
              round(time.time() - start_train, 2), "seconds")
        model = None
        history = None
        K.clear_session()
    fold += 1
    # break
