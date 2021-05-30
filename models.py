from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Activation, concatenate, BatchNormalization, Dropout
from keras.layers import Conv2DTranspose, ConvLSTM2D, Bidirectional, TimeDistributed, AveragePooling2D
from keras.models import Model
from keras.regularizers import l1,l2

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(n_filters, kernel_size, kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(n_filters, kernel_size, kernel_initializer="he_normal",padding="same")(x)
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
    
    inp = Input(shape=(None,None, 1))
    
    enc, skip = encoder(inp, filters, n_block, BN, DP)
    bottle = conv2d_block(enc, filters * 2**n_block, 3, BN)
    dec = decoder(bottle, skip, filters, n_block, BN, DP)
    output = Conv2D(n_classes, (1, 1), activation='softmax')(dec)

    model = Model(inp, output, name='U-Net')

    return model

def UConvLSTM_Nto1(n_classes, filters=32, ts=5):
    ''' ts: numer of time-steps (window size) '''
    in_im = Input(shape=(ts, None, None, 1))
    x = ConvLSTM2D(filters=filters, kernel_size=(3,3), padding="same")(in_im)
    out = Conv2D(n_classes, (1,1), activation = 'softmax', padding='same')(x)
    model = Model(in_im, out)
    return model

def BConvLSTM_Nto1(n_classes, filters=32, ts=5):
    ''' ts: numer of time-steps (window size) '''
    in_im = Input(shape=(ts, None, None, 1))
    x = Bidirectional(ConvLSTM2D(filters, 3, padding="same"), merge_mode='concat')(in_im)
    out = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(x)
    model = Model(in_im, out)
    return model

def convolution_layer(x,filter_size, dilation_rate=1, kernel_size=3, weight_decay=1E-4):
  x = Conv2D(filter_size, kernel_size, padding='same')(x)
  x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
  x = Activation('relu')(x)
  return x

def transpose_layer(x, filter_size, dilation_rate=1, kernel_size=3, strides=(2,2), weight_decay=1E-4):
  x = Conv2DTranspose(filter_size, kernel_size, strides=strides, padding='same')(x)
  x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
  x = Activation('relu')(x)
  return x

def convolution_layer_over_time(x, filter_size, dilation_rate=1, kernel_size=3, weight_decay=1E-4):
  x = TimeDistributed(Conv2D(filter_size, kernel_size, padding='same'))(x)
  x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
  x = Activation('relu')(x)
  return x

def BUnetConvLSTM_Nto1(n_classes, filters=16, filters_lstm=64, ts=5):
    ''' ts: numer of time-steps (window size) '''
    in_im = Input(shape=(ts, None, None, 1))

    p1=convolution_layer_over_time(in_im, filters)			
    p1=convolution_layer_over_time(p1, filters)
    e1=TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
    p2=convolution_layer_over_time(e1, filters*2)
    e2=TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
    p3=convolution_layer_over_time(e2, filters*4)
    e3=TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

    x=Bidirectional(ConvLSTM2D(filters_lstm, 3, return_sequences=False, padding="same"), merge_mode='concat')(e3)

    d3=transpose_layer(x, filters*4)
    d3=convolution_layer(d3, filters*4)
    d2=transpose_layer(d3, filters*2)
    d2=convolution_layer(d2, filters*2)
    d1=transpose_layer(d2, filters)
    out=convolution_layer(d1, filters)
    out = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(out)
    model=Model(in_im, out)

    return model