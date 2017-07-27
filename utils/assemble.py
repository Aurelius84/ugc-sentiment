"""
Utility functions for constructing MLC models.
"""
from keras.layers import Conv1D, Embedding, Flatten, MaxPool1D
from keras.layers import Dense, Dropout, Input, Activation
from keras.layers import ActivityRegularization
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import GRU
from keras.models import Model


def assemble(name, params):
    if name == 'deep_conv':
        return assemble_deep_conv(params)
    elif name == 'deep_lstm':
        return assemble_deep_lstm(params)
    else:
        raise ValueError("Unknown name of the model: %s." % name)


def assemble_deep_conv(params):
    """
    Construct one of the ADIOS models. The general structure is the following:
                                X-H-(Y0|H0)-H1-Y1,
    where all the H-layers are optional and depend on whether they are
    specified in the params dictionary.
    """
    # X

    input_shape = (params['X']['sequence_length'], params['X']['embedding_dim']
                   ) if params['iter']['model_type'] == "CNN-static" else (
                       params['X']['sequence_length'], )
    X = Input(shape=input_shape, dtype='int32', name='X')

    # embedding
    # Static model do not have embedding layer
    if params['iter']['model_type'] == "CNN-static":
        embedding = Dropout(0.1)(X)
    elif 'embedding_dim' in params['X'] and params['X']['embedding_dim']:
        embedding = Embedding(
            output_dim=params['X']['embedding_dim'],
            input_dim=params['X']['vocab_size'],
            input_length=params['X']['sequence_length'],
            name="embedding",
            mask_zero=False)(X)
        embedding = Dropout(0.1)(embedding)
    else:
        exit('embedding_dim param is not given!')

    # multi-layer Conv and max-pooling
    conv_layer_num = len(params['Conv1D'])
    for i in range(1, conv_layer_num + 1):
        H_input = embedding if i == 1 else H
        conv = Conv1D(
            filters=params['Conv1D']['layer%s' % i]['filters'],
            kernel_size=params['Conv1D']['layer%s' % i]['filter_size'],
            padding=params['Conv1D']['layer%s' % i]['padding_mode'],
            # activation='relu',
            strides=1,
            bias_regularizer=l2(0.01))(H_input)
        if 'batch_norm' in params['Conv1D']['layer%s' % i]:
            conv = BatchNormalization(
                **params['Conv1D']['layer%s' % i]['batch_norm'])(conv)
        conv_activation = Activation('relu')(conv)
        # conv_batch_norm = selu(BatchNormalization()(conv))
        conv_pooling = MaxPool1D(
            pool_size=params['Conv1D']['layer%s' % i]['pooling_size'],
            strides=1)(conv_activation)
        # dropout
        if 'dropout' in params['Conv1D']['layer%s' % i]:
            H = Dropout(
                params['Conv1D']['layer%s' % i]['dropout'])(conv_pooling)
            # H = dropout_selu(conv_pooling, params['Conv1D']['layer%s' % i]['dropout'])

        # flatten
    H = Flatten(name='H')(H)

    # Y output
    kwargs = params['Y']['kwargs'] if 'kwargs' in params['Y'] else {}
    if 'W_regularizer' in kwargs:
        kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
    Y = Dense(
        params['Y']['dim'],
        # activation='sigmoid',
        name='Y_active',
        bias_regularizer=l2(0.01),
        **kwargs)(H)
    # batch_norm
    if 'batch_norm' in params['Y']:
        Y = BatchNormalization(**params['Y']['batch_norm'])(Y)
    Y = Activation('sigmoid')(Y)
    if 'activity_reg' in params['Y']:
        Y = ActivityRegularization(name='Y', **params['Y']['activity_reg'])(Y)

    return Model(inputs=X, outputs=Y)


def assemble_deep_lstm(params):
    """
    Construct one of the ADIOS models. The general structure is the following:
                                X-H-H1-Y,
    where all the H-layers are optional and depend on whether they are
    specified in the params dictionary.
    """
    # X

    input_shape = (params['X']['sequence_length'], params['X']['embedding_dim']
                   ) if params['iter']['model_type'] == "CNN-static" else (
                       params['X']['sequence_length'], )
    X = Input(shape=input_shape, dtype='int32', name='X')

    # embedding
    # Static model do not have embedding layer
    if params['iter']['model_type'] == "CNN-static":
        embedding = Dropout(0.1)(X)
    elif 'embedding_dim' in params['X'] and params['X']['embedding_dim']:
        embedding = Embedding(
            output_dim=params['X']['embedding_dim'],
            input_dim=params['X']['vocab_size'],
            input_length=params['X']['sequence_length'],
            name="embedding",
            mask_zero=True)(X)
        embedding = Dropout(0.1)(embedding)
    else:
        exit('embedding_dim param is not given!')

    # multi-layer LSTM
    lstm_layer_num = len(params['LSTM'])
    for i in range(1, lstm_layer_num):
        lstm_input = embedding if i == 1 else lstm_out
        lstm_seq_out = GRU(
            params['LSTM']['layer%s' % i]['cell'],
            return_sequences=True)(lstm_input)
        # batch_norm
        if 'batch_norm' in params['LSTM']['layer%s' % i]:
            kwargs = params['LSTM']['layer%s' % i]['batch_norm']
            lstm_seq_out = BatchNormalization(**kwargs)(lstm_seq_out)
        # dropout
        if 'dropout' in params['LSTM']['layer%s' % i]:
            lstm_seq_out = Dropout(
                params['LSTM']['layer%s' % i]['dropout'])(lstm_seq_out)
    # last lstm
    lstm_out = GRU(
        params['LSTM']['layer%s' % lstm_layer_num]['cell'])(lstm_seq_out)
    # batch_norm
    if 'batch_norm' in params['LSTM']['layer%s' % lstm_layer_num]:
        kwargs = params['LSTM']['layer%s' % lstm_layer_num]['batch_norm']
        lstm_out = BatchNormalization(**kwargs)(lstm_out)
    # dropout
    if 'dropout' in params['LSTM']['layer%s' % lstm_layer_num]:
        lstm_out = Dropout(
            params['LSTM']['layer%s' % lstm_layer_num]['dropout'],
            name='H')(lstm_out)

    # Y output
    kwargs = params['Y']['kwargs'] if 'kwargs' in params['Y'] else {}
    if 'W_regularizer' in kwargs:
        kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
    Y = Dense(
        params['Y']['dim'],
        # activation='sigmoid',
        name='Y_active',
        bias_regularizer=l2(0.01),
        **kwargs)(lstm_out)
    # batch_norm
    if 'batch_norm' in params['Y']:
        Y = BatchNormalization(**params['Y']['batch_norm'])(Y)
    Y = Activation('softmax')(Y)
    if 'activity_reg' in params['Y']:
        Y = ActivityRegularization(name='Y', **params['Y']['activity_reg'])(Y)

    return Model(inputs=X, outputs=Y)
