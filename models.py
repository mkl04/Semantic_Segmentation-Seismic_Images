from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Activation, concatenate, BatchNormalization, Dropout
from keras.layers import Conv2DTranspose, ConvLSTM2D, Bidirectional, TimeDistributed, AveragePooling2D, MaxPooling2D, Lambda, GlobalAveragePooling2D
from keras.models import Model
from keras.regularizers import l1,l2
from keras import backend as K
import tensorflow as tf

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

def conv2d_block_ts(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = TimeDistributed(Conv2D(n_filters, kernel_size, kernel_initializer="he_normal", padding="same"))(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = TimeDistributed(Conv2D(n_filters, kernel_size, kernel_initializer="he_normal",padding="same"))(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder_ts(x, filters=16, n_block=3, batchnorm=False, dropout=False):
    skip = []
    for i in range(n_block):
        x = conv2d_block_ts(x, filters * 2**i, kernel_size=3, batchnorm=batchnorm)
        skip.append(Lambda(lambda n: n[:,-1])(x))
        x = TimeDistributed(MaxPool2D(2))(x)
        if dropout:
            x = Dropout(0.2)(x)
    return x, skip

def UNet_ts(n_classes, filters=64, n_block=4, BN=False, DP=False, ts=5):
    ''' Nto1 
    ts: numer of time-steps (window size) '''
    inp = Input(shape=(ts, None, None, 1))
    
    enc, skip = encoder_ts(inp, filters, n_block, BN, DP)
    bottle = Bidirectional(ConvLSTM2D(filters * 2**n_block, 3, return_sequences=False, padding="same"), merge_mode='concat')(enc)
    dec = decoder(bottle, skip, filters, n_block, BN, DP)
    output = Conv2D(n_classes, (1, 1), activation='softmax')(dec)

    model = Model(inp, output, name='U-Net_ConvLSTM')

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


def BUnetConvLSTM_Nto1_skip(n_classes, filters=16, filters_lstm=64, ts=5):
    ''' ts: numer of time-steps (window size) '''
    in_im = Input(shape=(ts, None, None, 1))

    p1=convolution_layer_over_time(in_im, filters)			
    p1=convolution_layer_over_time(p1, filters)
    e1=TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(p1)
    p2=convolution_layer_over_time(e1, filters*2)
    e2=TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(p2)
    p3=convolution_layer_over_time(e2, filters*4)
    e3=TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(p3)

    x=Bidirectional(ConvLSTM2D(filters_lstm, 3, return_sequences=False, padding="same"), merge_mode='concat')(e3)

    p3 = Lambda(lambda n: n[:,-1])(p3)
    d3=transpose_layer(x, filters*4)
    d3 = concatenate([d3, p3], axis=-1)
    d3=convolution_layer(d3, filters*4)

    p2 = Lambda(lambda n: n[:,-1])(p2)
    d2=transpose_layer(d3, filters*2)
    d2 = concatenate([d2, p2], axis=-1)
    d2=convolution_layer(d2, filters*2)

    p1 = Lambda(lambda n: n[:,-1])(p1)
    d1=transpose_layer(d2, filters)
    d1 = concatenate([d1, p1], axis=-1)

    out=convolution_layer(d1, filters)
    out = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(out)
    model=Model(in_im, out)

    return model


def dilated_layer(x, filter_size, dilation_rate=1, kernel_size=3, weight_decay=1E-4):
    '''r: dilated_rate'''
    x = Conv2D(filter_size, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    return x

def dilated_layer_over_time(x, filter_size, dilation_rate=1, kernel_size=3, weight_decay=1E-4):
    '''r: dilated_rate'''
    x = TimeDistributed(Conv2D(filter_size, kernel_size, padding='same', dilation_rate=dilation_rate)) (x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay)) (x)
    x = Activation('relu')(x)
    return x

def im_pooling_layer(x, filter_size):
    pooling=True
    shape_before=tf.shape(x)
    if pooling==True:
        mode=2
        if mode==1:
            x=TimeDistributed(GlobalAveragePooling2D())(x)
            x=K.expand_dims(K.expand_dims(x,2),2)
        elif mode==2:
            x=TimeDistributed(AveragePooling2D((32,32)))(x)
            
    x=dilated_layer_over_time(x, filter_size, 1, kernel_size=1)

    if pooling==True:
        x = TimeDistributed(Lambda(lambda y: K.tf.image.resize_bilinear(y,size=(32,32))))(x)
    return x

def spatial_pyramid_pooling(in_im, filter_size, max_rate=8, global_average_pooling=False):
    x=[]
    if max_rate>=1:
        x.append(dilated_layer_over_time(in_im,filter_size,1))
    if max_rate>=2:
        x.append(dilated_layer_over_time(in_im,filter_size,2)) #6
    if max_rate>=4:
        x.append(dilated_layer_over_time(in_im,filter_size,4)) #12
    if max_rate>=8:
        x.append(dilated_layer_over_time(in_im,filter_size,8)) #18
    # if global_average_pooling==True:
    #     x.append(im_pooling_layer(in_im, filter_size))
    out = concatenate(x, axis=4)
    return out

def BAtrousConvLSTM(n_classes, filters=16, filters_lstm=128, ts=5, gap=False):
    ''' ts: numer of time-steps (window size) '''
    in_im = Input(shape=(ts, None, None, 1))

    x=dilated_layer_over_time(in_im, filters)
    x=dilated_layer_over_time(x, filters)
    x=spatial_pyramid_pooling(x, filters, max_rate=8, global_average_pooling=gap)
    
    x=Bidirectional(ConvLSTM2D(filters_lstm, 3, return_sequences=False, padding="same"), merge_mode='concat')(x)

    out = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(x)
    model=Model(in_im, out)

    return model


def UConvLSTM_NtoN(n_classes, filters=32, ts=5):
    ''' ts: numer of time-steps (window size) '''
    in_im = Input(shape=(ts, None, None, 1))
    x = ConvLSTM2D(filters=filters, kernel_size=(3,3), return_sequences=True, padding="same")(in_im)
    out = TimeDistributed(Conv2D(n_classes, (1,1), activation = 'softmax', padding='same'))(x)
    model = Model(in_im, out)

    return model

def BConvLSTM_NtoN(n_classes, filters=32, ts=5):
    ''' ts: numer of time-steps (window size) '''
    in_im = Input(shape=(ts, None, None, 1))
    x = Bidirectional(
        ConvLSTM2D(filters=filters, kernel_size=(3,3), return_sequences=True, padding="same"), 
            merge_mode='concat')(in_im)
    out = TimeDistributed(Conv2D(n_classes, (1, 1), activation='softmax', padding='same'))(x)
    model = Model(in_im, out)

    return model


def transpose_layer_over_time(x, filter_size, dilation_rate=1, kernel_size=3, strides=(2,2), weight_decay=1E-4):
    x = TimeDistributed(Conv2DTranspose(filter_size, kernel_size, strides=strides, padding='same'))(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    return x

def BUnetConvLSTM_NtoN(n_classes, filters=16, filters_lstm=64, ts=5):
    ''' ts: numer of time-steps (window size) '''
    in_im = Input(shape=(ts, None, None, 1))
    p1=dilated_layer_over_time(in_im,filters)			
    p1=dilated_layer_over_time(p1,filters)
    e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)

    p2=dilated_layer_over_time(e1,filters*2)
    e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)

    p3=dilated_layer_over_time(e2,filters*4)
    e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

    x = Bidirectional(
        ConvLSTM2D(filters=filters_lstm, kernel_size=(3,3), return_sequences=True, padding="same"),
            merge_mode='concat')(e3)

    d3 = transpose_layer_over_time(x,filters*4)
    d3 = concatenate([d3, p3], axis=4)

    d3=dilated_layer_over_time(d3,filters*4)
    d2 = transpose_layer_over_time(d3,filters*2)
    d2 = concatenate([d2, p2], axis=4)

    d2=dilated_layer_over_time(d2,filters*2)
    d1 = transpose_layer_over_time(d2,filters)
    d1 = concatenate([d1, p1], axis=4)

    out=dilated_layer_over_time(d1,filters)
    out = TimeDistributed(Conv2D(n_classes, (1, 1), activation='softmax', padding='same'))(out)
    model = Model(in_im, out)

    return model

def BUnetConvLSTM2_NtoN(n_classes, filters=16, filters_lstm=64, ts=5):
    ''' 
    n_blocks = 4
    Just one conv layer at beginning to reduce number of parameters

    ts: numer of time-steps (window size) '''

    in_im = Input(shape=(ts, None, None, 1))
    p1=dilated_layer_over_time(in_im,filters)			
    # p1=dilated_layer_over_time(p1,filters)
    e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)

    p2=dilated_layer_over_time(e1,filters*2)
    e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)

    p3=dilated_layer_over_time(e2,filters*4)
    e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

    p4=dilated_layer_over_time(e3,filters*8)
    e4 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p4)

    x = Bidirectional(
        ConvLSTM2D(filters=filters_lstm, kernel_size=(3,3), return_sequences=True, padding="same"),
            merge_mode='concat')(e4)

    d4 = transpose_layer_over_time(x,filters*8)
    d4 = concatenate([d4, p4], axis=4)

    d3=dilated_layer_over_time(d4,filters*8)
    d3 = transpose_layer_over_time(d4,filters*4)
    d3 = concatenate([d3, p3], axis=4)

    d3=dilated_layer_over_time(d3,filters*4)
    d2 = transpose_layer_over_time(d3,filters*2)
    d2 = concatenate([d2, p2], axis=4)

    d2=dilated_layer_over_time(d2,filters*2)
    d1 = transpose_layer_over_time(d2,filters)
    d1 = concatenate([d1, p1], axis=4)

    out=dilated_layer_over_time(d1,filters)
    out = TimeDistributed(Conv2D(n_classes, (1, 1), activation='softmax', padding='same'))(out)
    model = Model(in_im, out)
    
    return model