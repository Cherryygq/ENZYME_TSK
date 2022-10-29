import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Conv1D, Bidirectional, GRU, Flatten, Permute, Concatenate, \
    Multiply, GlobalAveragePooling1D, Reshape, GlobalAveragePooling2D, Add, BatchNormalization, GlobalMaxPooling1D, \
    Activation, AvgPool1D
from keras import backend as K
import keras

# GPU
from tensorflow.compat.v1 import ConfigProto
from keras_self_attention import SeqSelfAttention


def get_dl_model(model_name):
    if model_name == "ANN":
        inp = Input(shape=(3, 100))
        x = Dense(256, activation='relu')(inp)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(7, activation='softmax')(x)
        model = Model(inp, out)

    elif model_name == "CNN":
        inp = Input(shape=(3, 100))
        x = Conv1D(128, (3), padding='same', activation='relu')(inp)
        x = Conv1D(64, (3), padding='same', activation='relu')(x)
        x = Conv1D(32, (3), padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(7, activation='softmax')(x)
        model = Model(inp, out)

    elif model_name == "LSTM":
        inp = Input(shape=(3, 100))
        x = LSTM(32, activation='tanh', return_sequences=True)(inp)
        x = Dropout(0.2)(x)
        x = LSTM(32, activation='tanh')(x)
        out = Dense(7, activation='softmax')(x)
        model = Model(inp, out)

    elif model_name == "BILSTM":
        inp = Input(shape=(3, 100))
        x = Bidirectional(LSTM(32, activation='tanh', return_sequences=True))(inp)
        x = Dropout(0.4)(x)
        x = Bidirectional(LSTM(32, activation='tanh'))(x)
        out = Dense(7, activation='softmax')(x)
        model = Model(inp, out)

    elif model_name == "GRU":
        inp = Input(shape=(3, 100))
        x = GRU(32, activation='tanh', return_sequences=True)(inp)
        x = GRU(32, activation='tanh')(x)
        out = Dense(7, activation='softmax')(x)
        model = Model(inp, out)

    elif model_name == "ABLE":

        # inp = Input(shape=(3, 100))
        # x0 = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(inp)
        # squeeze_tensor = GlobalAveragePooling1D(data_format='channels_last')(x0)
        # print('squeeze_tensor.shape', squeeze_tensor.shape)
        # fc_out_1 = Dense(256, activation='relu')(squeeze_tensor)
        # fc_out_2 = Dense(256, activation='sigmoid')(fc_out_1)
        # print('fc_out_2.shape', fc_out_2.shape)
        # x1 = Multiply()([x0, fc_out_2])
        #
        # x2 = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(x0)
        # squeeze_tensor = GlobalAveragePooling1D(data_format='channels_last')(x2)
        # print('squeeze_tensor.shape', squeeze_tensor.shape)
        # fc_out_11 = Dense(256, activation='relu')(squeeze_tensor)
        # fc_out_22 = Dense(256, activation='sigmoid')(fc_out_11)
        # print('fc_out_22.shape', fc_out_22.shape)
        # x2 = Multiply()([x2, fc_out_22])
        #
        # x = Concatenate(axis=-1)([x1, x2])
        # squeeze_tensor = GlobalAveragePooling1D(data_format='channels_last')(x)
        # print('squeeze_tensor.shape', squeeze_tensor.shape)
        # fc_out_1 = Dense(512, activation='relu')(squeeze_tensor)
        # fc_out_2 = Dense(512, activation='sigmoid')(fc_out_1)
        # print('fc_out_2.shape', fc_out_2.shape)
        # x = Multiply()([x, fc_out_2])
        #
        # x = Flatten()(x)
        # fea = Dense(256, activation='relu', name="fea256")(x)
        # x = Dense(128, activation='relu')(fea)  # new1
        # out = Dense(6, activation='softmax')(x)
        # model = Model(inp, out)

        # ours
        inp = Input(shape=(3, 100))
        x1 = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(inp)
        x2 = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(x1)
        x2 = Concatenate(axis=-1)([x1, x2])
        x3 = SeqSelfAttention(attention_activation='tanh')(x2)
        x3 = Concatenate(axis=-1)([x2, x3])
        x4 = BatchNormalization()(x3)

        squeeze_tensor = GlobalAveragePooling1D(data_format='channels_last')(x4)
        squeeze_tensor = Reshape((1, 1024))(squeeze_tensor)
        fc_out_1 = Dense(1024, activation='relu', use_bias=True)(squeeze_tensor)
        fc_out_1 = Dropout(0.2)(fc_out_1)
        fc_out_2 = Dense(1024, activation='sigmoid', use_bias=True)(fc_out_1)
        fc_out_2 = Dropout(0.2)(fc_out_2)
        x5 = Multiply()([x4, fc_out_2])
        con = Conv1D(1, (1), activation='sigmoid', use_bias=True, kernel_initializer='he_normal')(x4)
        x6 = Multiply()([x4, con])
        x = Add()([x5, x6])
        x = BatchNormalization()(x)

        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.4)(x)
        out = Dense(6, activation='softmax', name="last_6")(x)
        model = Model(inp, out)

    # # ori ABLE
    # inp = Input(shape=(3, 100))
    # x = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(inp)
    # x = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(x)
    # x = SeqSelfAttention(attention_activation='tanh')(x)
    # x = Flatten()(x)
    # out = Dense(6, activation='softmax', name="dense")(x)
    # model = Model(inp, out)

    elif model_name == "DEEPEC":
        inp = Input(shape=(3, 100))
        x = Conv1D(128, (3), padding='same', activation='relu')(inp)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        out = Dense(7, activation='softmax')(x)
        model = Model(inp, out)

    return model


