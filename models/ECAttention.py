import math
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Conv1D,
    Reshape,
    Activation,
    GlobalAveragePooling2D,
    multiply
)

def eca_attention(inputs_tensor=None, gamma=2, b=1):

    channels = K.int_shape(inputs_tensor)[-1]
    t = int(abs((math.log(channels,2)+b)/gamma))
    k = t if t%2 else t+1

    x = GlobalAveragePooling2D()(inputs_tensor)
    x = Reshape((channels, 1))(x)
    x = Conv1D(1, kernel_size=k, padding="same", use_bias=False)(x)
    x = Activation('sigmoid')(x)  #shape=[batch,chnnels,1]
    x = Reshape((1, channels))(x)

    output = multiply([inputs_tensor, x])
    
    return output
