from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Activation, concatenate, BatchNormalization, Conv2DTranspose, Dropout
from keras.models import Model

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