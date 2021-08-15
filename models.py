from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Activation, concatenate, BatchNormalization, Dropout
from keras.layers import Conv2DTranspose, ConvLSTM2D, Bidirectional, TimeDistributed, AveragePooling2D, MaxPooling2D, Lambda
from keras.layers import GlobalAveragePooling2D, add
from keras.models import Model
from keras.regularizers import l1,l2
from keras import backend as K
import tensorflow as tf

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, n_layers=2):

    for i in range(n_layers):
        if i==0:
            x = Conv2D(n_filters, kernel_size, kernel_initializer="he_normal", padding="same")(input_tensor)
        else:
            x = Conv2D(n_filters, kernel_size, kernel_initializer="he_normal", padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x

def encoder(x, filters=16, n_block=3, batchnorm=False, dropout=False):
    skip = []
    for i in range(n_block):
        x = conv2d_block(x, filters * 2**i, kernel_size=3, batchnorm=batchnorm)
        skip.append(x)
        x = MaxPool2D(2)(x)
        if dropout:
            x = Dropout(0.2)(x)
    return x, skip

def decoder(x, skip, filters, n_block=3, batchnorm=False, dropout=False):
    for i in reversed(range(n_block)):
        x = Conv2DTranspose(filters * 2**i, 3, strides=2, padding='same')(x)
        x = concatenate([x, skip[i]])
        if dropout:
            x = Dropout(0.2)(x)
        x = conv2d_block(x, filters * 2**i, kernel_size=3, batchnorm=batchnorm)
    return x

def UNet(n_classes, filters=64, n_block=4, BN=False, DP=False):
    """
    Function to create the U-Net architecture.

    Parameters
    ----------
    n_classes : int
        number of classes
    filters : int
        number of filters in the first convolutional layers
    n_block : int
        number of blocks for encoder/decoder path
    BN : bool
        if True, adds Batch Normalization on each convolutional layer
    DP : bool
        if True, adds Dropout layers
    """

    inp = Input(shape=(None,None, 1))
    
    enc, skip = encoder(inp, filters, n_block, BN, DP)
    bottle = conv2d_block(enc, filters * 2**n_block, 3, BN)
    dec = decoder(bottle, skip, filters, n_block, BN, DP)
    output = Conv2D(n_classes, (1, 1), activation='softmax')(dec)

    model = Model(inp, output, name='UNet')

    return model

def bottleneck(x, filters_bottleneck, mode='cascade', depth=6):
    """
    Bottleneck for the Atrous U-Net architecture.

    Parameters
    ----------
    x : Layer
        previous layer
    filters_bottleneck : int
        number of filter at the bottleneck's convolutional layer 
    mode : str
        'cascade' or 'parallel'
    depth : int
        number of atrous convolutional layers
    """

    dilated_layers = []
    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv2D(filters_bottleneck, (3,3),
                       activation='relu', padding='same', dilation_rate=2**i)(x)
            dilated_layers.append(x)
        return add(dilated_layers)
    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(
                Conv2D(filters_bottleneck, (3,3),
                       activation='relu', padding='same', dilation_rate=2**i)(x)
            )
        return add(dilated_layers)

def AtrousUNet(n_classes, filters=64, n_block=4, BN=False, mode="cascade"):
    """
    Function to create the Atrous U-Net architecture.

    Parameters
    ----------
    n_classes : int
        number of classes
    filters : int
        number of filters in the first convolutional layers
    n_block : int
        number of blocks for encoder/decoder path
    BN : bool
        if True, adds Batch Normalization on each convolutional layer
    mode : str
        mode of the bottleneck with dilated convolutional. i.e. 'cascade' or 'parallel'
    """

    inp = Input(shape=(None,None, 1))
    
    enc, skip = encoder(inp, filters, n_block, BN)
    bottle = bottleneck(enc, filters_bottleneck=filters * 2**n_block, mode=mode)
    dec = decoder(bottle, skip, filters, n_block, BN)
    output = Conv2D(n_classes, (1, 1), activation='softmax')(dec)

    model = Model(inp, output, name='AtrousUNet')

    return model

##################################
#           LSTM's version
##################################


#################
# N-to-1
#################

def UConvLSTM_Nto1(n_classes, filters=32, ts=5):
    """
    Unidirectional ConvLSTM N-to-1

    Parameters
    ----------
    n_classes : int
        number of classes
    filters : int
        number of filters in the first convolutional layers
    ts : int
        numer of time-steps (window size)
    """

    in_im = Input(shape=(ts, None, None, 1))
    x = ConvLSTM2D(filters=filters, kernel_size=(3,3), padding="same")(in_im)
    out = Conv2D(n_classes, (1,1), activation = 'softmax', padding='same')(x)
    model = Model(in_im, out)
    return model


def BConvLSTM_Nto1(n_classes, filters=32, ts=5):
    """
    Bidirectional ConvLSTM N-to-1

    Parameters
    ----------
    n_classes : int
        number of classes
    filters : int
        number of filters in the first convolutional layers
    ts : int
        numer of time-steps (window size)
    """

    in_im = Input(shape=(ts, None, None, 1))
    x = Bidirectional(ConvLSTM2D(filters, 3, padding="same"), merge_mode='concat')(in_im)
    out = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(x)
    model = Model(in_im, out)
    return model


def conv2d_block_TD(input_tensor, n_filters, kernel_size=3, batchnorm=True, n_layers=1):

    for i in range(n_layers):
        if i==0:
            x = TimeDistributed(Conv2D(n_filters, kernel_size, kernel_initializer="he_normal", padding="same"))(input_tensor)
        else:
            x = TimeDistributed(Conv2D(n_filters, kernel_size, kernel_initializer="he_normal", padding="same"))(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x

def encoder_TD(x, filters=16, n_block=3, batchnorm=False, dropout=False):
    skip = []
    for i in range(n_block):
        x = conv2d_block_TD(x, filters * 2**i, batchnorm=batchnorm)
        skip.append(x)
        x = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(x)
        if dropout:
            x = Dropout(0.2)(x)
    return x, skip

def conv2d_transpose_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):

    x = Conv2DTranspose(n_filters, kernel_size, kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def BiUNetConvLSTM_no_skip_connect(n_classes, filters=16, n_block=4, filters_lstm=64, ts=5, BN=True, DP=False):

    inp = Input(shape=(ts, None, None, 1))

    enc, skip = encoder_TD(inp, filters, n_block, BN, DP)
    p1, p2, p3, p4 = skip

    bottle = Bidirectional(
        ConvLSTM2D(filters=filters_lstm, kernel_size=(3,3), return_sequences=False, padding="same"),
            merge_mode='concat')(enc)

    d4 = conv2d_transpose_block(bottle, filters*8)

    d3 = conv2d_block(d4,filters*8)
    d3 = conv2d_transpose_block(d4,filters*4)

    d3 = conv2d_block(d3,filters*4)
    d2 = conv2d_transpose_block(d3,filters*2)

    d2 = conv2d_block(d2,filters*2)
    d1 = conv2d_transpose_block(d2,filters)

    out = conv2d_block(d1,filters)
    out = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(out)

    model = Model(inp, out)
    
    return model


def BiUNetConvLSTM(n_classes, filters=16, n_block=4, filters_lstm=64, ts=5, BN=True, DP=False):
    """
    Function to create the Bidirectional U-Net ConvLSTM architecture

    Parameters
    ----------
    n_classes : int
        number of classes
    filters : int
        number of filters in the first convolutional layers
    n_block : int
        number of blocks for encoder/decoder path
    filters_lstm : int
        number of filters in the ConvLSTM layer
    ts : int
        numer of time-steps (window size)      
    BN : bool
        if True, adds Batch Normalization on each convolutional layer
    DP : bool
        if True, adds Dropout layers
    """

    inp = Input(shape=(ts, None, None, 1))

    enc, skip = encoder_TD(inp, filters, n_block, BN, DP)
    p1, p2, p3, p4 = skip

    bottle = Bidirectional(
        ConvLSTM2D(filters=filters_lstm, kernel_size=(3,3), return_sequences=False, padding="same"),
            merge_mode='concat')(enc)

    p4 = Lambda(lambda n: n[:,-1])(p4)
    d4 = conv2d_transpose_block(bottle, filters*8)
    d4 = concatenate([d4, p4], axis=-1)

    p3 = Lambda(lambda n: n[:,-1])(p3)
    d3 = conv2d_block(d4,filters*8)
    d3 = conv2d_transpose_block(d4,filters*4)
    d3 = concatenate([d3, p3], axis=-1)

    p2 = Lambda(lambda n: n[:,-1])(p2)
    d3 = conv2d_block(d3,filters*4)
    d2 = conv2d_transpose_block(d3,filters*2)
    d2 = concatenate([d2, p2], axis=-1)

    p1 = Lambda(lambda n: n[:,-1])(p1)
    d2 = conv2d_block(d2,filters*2)
    d1 = conv2d_transpose_block(d2,filters)
    d1 = concatenate([d1, p1], axis=-1)

    out = conv2d_block(d1,filters)
    out = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(out)

    model = Model(inp, out)
    
    return model


#################
# N-to-N
#################


def UConvLSTM_NtoN(n_classes, filters=32, ts=5):
    """
    Unidirectional ConvLSTM N-to-N

    Parameters
    ----------
    n_classes : int
        number of classes
    filters : int
        number of filters in the first convolutional layers
    ts : int
        numer of time-steps (window size)
    """
    in_im = Input(shape=(ts, None, None, 1))
    x = ConvLSTM2D(filters=filters, kernel_size=(3,3), return_sequences=True, padding="same")(in_im)
    out = TimeDistributed(Conv2D(n_classes, (1,1), activation = 'softmax', padding='same'))(x)
    model = Model(in_im, out)

    return model


def BConvLSTM_NtoN(n_classes, filters=32, ts=5):
    """
    Bidirectional ConvLSTM N-to-N

    Parameters
    ----------
    n_classes : int
        number of classes
    filters : int
        number of filters in the first convolutional layers
    ts : int
        numer of time-steps (window size)
    """
    in_im = Input(shape=(ts, None, None, 1))
    x = Bidirectional(
        ConvLSTM2D(filters=filters, kernel_size=(3,3), return_sequences=True, padding="same"), 
            merge_mode='concat')(in_im)
    out = TimeDistributed(Conv2D(n_classes, (1, 1), activation='softmax', padding='same'))(x)
    model = Model(in_im, out)

    return model


def conv2d_transpose_block_TD(input_tensor, n_filters, kernel_size=3, batchnorm=True):

    x = TimeDistributed(Conv2DTranspose(n_filters, kernel_size, strides=2, kernel_initializer="he_normal",
     padding="same"))(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def BiUNetConvLSTM_NtoN(n_classes, filters=16, n_block=4, filters_lstm=64, ts=5, BN=True, DP=False):

    inp = Input(shape=(ts, None, None, 1))

    enc, skip = encoder_TD(inp, filters, n_block, BN, DP)
    p1, p2, p3, p4 = skip

    bottle = Bidirectional(
        ConvLSTM2D(filters=filters_lstm, kernel_size=(3,3), return_sequences=True, padding="same"),
            merge_mode='concat')(enc)

    d4 = conv2d_transpose_block_TD(bottle, filters*8)
    d4 = concatenate([d4, p4], axis=4)

    d3 = conv2d_block_TD(d4,filters*8)
    d3 = conv2d_transpose_block_TD(d4,filters*4)
    d3 = concatenate([d3, p3], axis=4)

    d3 = conv2d_block_TD(d3,filters*4)
    d2 = conv2d_transpose_block_TD(d3,filters*2)
    d2 = concatenate([d2, p2], axis=4)

    d2 = conv2d_block_TD(d2,filters*2)
    d1 = conv2d_transpose_block_TD(d2,filters)
    d1 = concatenate([d1, p1], axis=4)

    out = conv2d_block_TD(d1,filters)
    out = TimeDistributed(Conv2D(n_classes, (1, 1), activation='softmax', padding='same'))(out)

    model = Model(inp, out, name='BiUNetConvLSTM')
    
    return model

def ASPP_over_time(x, filters_bottleneck, mode='cas', depth=6, activation='tanh'):
    """
    Bottleneck used for the Atrous U-Net architecture, but customizes to
    LSTM(sequence) version.

    Parameters
    ----------
    x : Layer
        previous layer
    filters_bottleneck : int
        number of filter at the bottleneck's convolutional layer 
    mode : str
        'cascade' or 'parallel'
    depth : int
        number of atrous convolutional layers
    activation : str
        activation for the ConvLSTM layer. i.e. 'relu' or 'tanh'

    """

    dilated_layers = []

    if mode == 'cas':  # cascade, used in the competition
        for i in range(depth):
            x = Bidirectional(
                ConvLSTM2D(filters_bottleneck, (3,3),
                       return_sequences=True, padding="same", dilation_rate=2**i),
                    merge_mode='concat')(x)
            dilated_layers.append(x)
        return add(dilated_layers)

    elif mode == 'par':  # parallel, Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(
                Bidirectional(
                ConvLSTM2D(filters_bottleneck, (3,3),
                       return_sequences=True, padding="same", dilation_rate=2**i),
                    merge_mode='concat')(x)
            )
        return add(dilated_layers)


def BiAtrousUNetConvLSTM_NtoN(n_classes, filters=16, n_block=4, filters_lstm=256, ts=5, BN=True, DP=False, mode="par"):
    """
    Function to create the Bidirectional U-Net ConvLSTM architecture

    Parameters
    ----------
    n_classes : int
        number of classes
    filters : int
        number of filters in the first convolutional layers
    n_block : int
        number of blocks for encoder/decoder path
    filters_lstm : int
        number of filters in the ConvLSTM layer
    ts : int
        numer of time-steps (window size)      
    BN : bool
        if True, adds Batch Normalization on each convolutional layer
    DP : bool
        if True, adds Dropout layers
    mode : str
        mode of the bottleneck with dilated convolutional. i.e. 'cascade' or 'parallel'
    """

    inp = Input(shape=(ts, None, None, 1))

    enc, skip = encoder_TD(inp, filters, n_block, BN, DP)
    p1, p2, p3, p4 = skip

    bottle = ASPP_over_time(enc, filters_bottleneck=filters_lstm, mode=mode) # filters * 16

    d4 = conv2d_transpose_block_TD(bottle, filters*8)
    d4 = concatenate([d4, p4], axis=4)

    d3 = conv2d_block_TD(d4,filters*8)
    d3 = conv2d_transpose_block_TD(d4,filters*4)
    d3 = concatenate([d3, p3], axis=4)

    d3 = conv2d_block_TD(d3,filters*4)
    d2 = conv2d_transpose_block_TD(d3,filters*2)
    d2 = concatenate([d2, p2], axis=4)

    d2 = conv2d_block_TD(d2,filters*2)
    d1 = conv2d_transpose_block_TD(d2,filters)
    d1 = concatenate([d1, p1], axis=4)

    out = conv2d_block_TD(d1,filters)
    out = TimeDistributed(Conv2D(n_classes, (1, 1), activation='softmax', padding='same'))(out)

    model = Model(inp, out, name='BiAtrousUNetConvLSTM')
    
    return model