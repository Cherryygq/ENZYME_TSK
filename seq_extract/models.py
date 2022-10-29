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


def get_model(model_name):

    if model_name == "BABC":

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

   

    return model


