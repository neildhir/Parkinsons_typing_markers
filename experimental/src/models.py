from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Conv1D, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam



def mk_cnn_model(input_shape):
    reg = 1e-6
    drop = 0.3
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv1D(filters = 16, kernel_size = 4, strides= 1, padding='valid',  activation='relu', use_bias=True,
                     kernel_regularizer = regularizers.l2(reg),
                     bias_regularizer = regularizers.l2(reg)))
    # model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(Conv1D(filters = 8, kernel_size = 4, strides= 1, padding='valid',  activation='relu', use_bias=True,
                     kernel_regularizer = regularizers.l2(reg),
                     bias_regularizer = regularizers.l2(reg)))
    # model.add(BatchNormalization())
    model.add(Dropout(drop))

    '''
    model.add(Flatten())
    model.add(Dense(2, activation='softmax' ,use_bias=True,
                    kernel_regularizer = regularizers.l2(reg),
                    bias_regularizer = regularizers.l2(reg)))
    '''
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(2, activation='softmax'))


    return model


def mk_composite_model(input_shape):
    reg = 1e-6
    drop = 0.5
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters = 16, kernel_size = 4, strides= 1, padding='valid',  activation='relu', name = 'conv1d_1',
                     kernel_regularizer = regularizers.l2(reg),
                     bias_regularizer = regularizers.l2(reg)))
    # model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(Conv1D(filters = 8, kernel_size = 4, strides= 1, padding='valid',  activation='relu', name = 'conv1d_2',
                     kernel_regularizer = regularizers.l2(reg),
                     bias_regularizer = regularizers.l2(reg)))
    # model.add(BatchNormalization())
    model.add(Dropout(drop))
    # model.add(Flatten())
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(2, activation='softmax'))
    return model