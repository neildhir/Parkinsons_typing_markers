from keras.layers import Conv1D, Dense, GlobalMaxPool1D, MaxPooling1D, Dropout

# =============
# MODEL BLOCKS
# =============


def character_dense_dropout_block(flattened, units, dropout_rates, **params):
    """
    To be used with char_cnn_model() from Zhang et al.'s paper.

    Parameters
    ----------
    flattened : [type]
        [description]
    units : [type]
        [description]
    rates : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    assert len(units) == len(dropout_rates)

    # Create multiple filters on the fly
    j = 0
    while units:
        unit = units.pop(0)

        # Assign appropriate activation function for dense layers
        if units:
            # List is not empty
            activation_func = params["dense_activation"]
        elif not units:
            # List is empty, alas we have reached the end of it and switch activation
            activation_func = params["last_activation"]

        # Dense
        flattened = Dense(
            unit,
            kernel_initializer=params["dense_kernel_initializer"],
            bias_initializer=params["dense_bias_initializer"],
            activation=activation_func,
        )(flattened)

        # Dropout
        if dropout_rates[j]:
            # Only enters this logic if the entry is != None
            flattened = Dropout(rate=dropout_rates[j])(flattened)

        # Increment index counter
        j += 1

    return flattened


def character_1D_convolution_maxpool_block_v2(embedded, nb_filters, filter_lengths, pool_lengths, **params: dict):
    """
    To be used with char_cnn_model() from Zhang et al.'s paper.

    Parameters
    ----------
    embedded : [type]
        A sentencen which has been one-hot encoded (on character-level)
    nb_filters : tuple, optional
        [description]
    filter_lengths : tuple, optional
        [description]
    pool_length : tuple, optional
        The pooling sizes, we use None if a layers is not meant to have pooling

    Returns
    -------
    [type]
        [description]
    """

    assert len(nb_filters) == len(filter_lengths) == len(pool_lengths)

    # Create multiple filters on the fly
    for i in range(len(nb_filters)):

        # Convolution
        embedded = Conv1D(
            filters=nb_filters[i],
            kernel_size=filter_lengths[i],
            padding=params["conv_padding"],
            kernel_initializer=params["conv_kernel_initializer"],
            bias_initializer=params["conv_bias_initializer"],
            activation=params["conv_activation"],
        )(embedded)

        # Max pooling
        if pool_lengths[i]:
            embedded = MaxPooling1D(pool_size=pool_lengths[i])(embedded)

    return embedded


def character_1D_convolution_block(
    embedded, nb_filter=(32, 64), filter_length=(3, 3), subsample=(2, 1), pool_length=(2, 2)
):

    assert len(nb_filter) == len(filter_length) == len(subsample) == len(pool_length)

    # Create multiple filters on the fly
    for i in range(len(nb_filter)):
        # convolution
        embedded = Conv1D(
            filters=nb_filter[i],
            kernel_size=filter_length[i],
            padding="valid",
            activation="relu",  # TODO: may be a more suitable activation func. here
            kernel_initializer="glorot_normal",
            strides=subsample[i],
        )(embedded)
        # pooling
        if pool_length[i]:
            embedded = Dropout(0.1)(embedded)
            embedded = MaxPooling1D(pool_size=pool_length[i])(embedded)

    return embedded
