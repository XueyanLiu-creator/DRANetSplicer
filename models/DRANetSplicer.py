from tensorflow.keras.layers import (
    Input,
    add,
    Conv2D,
    Dense,
    Activation,
    BatchNormalization,
    AveragePooling2D,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from models.ECAttention import eca_attention

def identity_block(input_tensor, kernel_size, filters):

    x = eca_attention(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    
    x = add([x, input_tensor])
    return x

def conv_block(input_tensor, kernel_size, filters, pool_size=2):

    x = eca_attention(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = AveragePooling2D(pool_size=pool_size)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)

    shortcut = BatchNormalization()(input_tensor)
    shortcut = Activation('relu')(shortcut)
    shortcut = Conv2D(filters, 1, strides=1, padding='same')(shortcut)
    shortcut = AveragePooling2D(pool_size=pool_size)(shortcut)

    x = add([x, shortcut])
    return x

def DRANetSplicer(data_shape=(402, 4, 1)):

    input = Input(shape=data_shape)

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(64, (7,4), strides=1, padding='valid')(x)

    x = conv_block(x, (7,1), 64, pool_size=(1,1))
    x = identity_block(x, (7,1), 64)

    x = conv_block(x, (7,1), 128, pool_size=(3,1))
    x = identity_block(x, (7,1), 128)

    x = conv_block(x, (7,1), 128, pool_size=(4,1))
    x = identity_block(x, (7,1), 128)

    x = conv_block(x, (7,1), 256, pool_size=(4,1))
    x = identity_block(x, (7,1), 256)

    x = GlobalAveragePooling2D()(x)

    output = Dense(2, activation="softmax")(x)

    model = Model(inputs=input, outputs=output)

    return model
