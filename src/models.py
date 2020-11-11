from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    LeakyReLU,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def mk_composite_model(input_shape,cfg,mode):
    reg = cfg.train[mode].reg
    drop = cfg.train[mode].drop

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters = cfg.train.network.filters_1,
                     kernel_size = cfg.train.network.kernel_size,
                     strides = cfg.train.network.strides,
                     padding= cfg.train.network.padding,
                     activation=cfg.train.network.activation,
                     name = 'conv1d_1',
                     kernel_regularizer = regularizers.l2(reg),
                     bias_regularizer = regularizers.l2(reg)))
    # model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(Conv1D(filters = cfg.train.network.filters_2,
                     kernel_size = cfg.train.network.kernel_size,
                     strides=cfg.train.network.strides,
                     padding=cfg.train.network.padding,
                     activation=cfg.train.network.activation,
                     name = 'conv1d_2',
                     kernel_regularizer = regularizers.l2(reg),
                     bias_regularizer = regularizers.l2(reg)))
    # model.add(BatchNormalization())
    model.add(Dropout(drop))
    # model.add(Flatten())
    model.add(Bidirectional(LSTM(cfg.train.network.lstm_hidden)))
    model.add(Dense(2, activation=cfg.train.network.outact))
    return model
