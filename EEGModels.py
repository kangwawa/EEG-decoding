from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
import tensorflow as tf



######################——————EEGNet——————##########################
def EEGNet_SimAM(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    # Part 1 - Conventional Convolution
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    # Part 1 - Depthwise convolution
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(name="ba")(block1)
    block1 = Activation('elu')(block1)
    # Introduction of the SimAM attention module
    block1, attention = simam(block1, num_features=block1.shape[-1], name='simam')
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    # Part 2 - separable convolution
    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)



######################——————SmimAM——————##########################
def simam(x, num_features, name):
    # Calculating scale and shift in simam
    scale = tf.Variable(initial_value=tf.ones([num_features]), name=name + '_scale')
    shift = tf.Variable(initial_value=tf.zeros([num_features]), name=name + '_shift')


    mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    std = tf.reduce_mean(tf.square(x - mean), axis=[1, 2], keepdims=True)
    x = (x - mean) * tf.math.rsqrt(std + 1e-5)
    x = tf.multiply(x, tf.reshape(scale, [1, 1, 1, num_features])) + tf.reshape(shift, [1, 1, 1, num_features])


    return x, scale
