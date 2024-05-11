import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D
from keras.layers import add
from keras.models import Model
from keras.layers import BatchNormalization, Activation

from CustomLayers.ConvBlock2D import conv_block_2D

kernel_initializer = 'he_uniform'
interpolation = "bilinear"


def attention_gate(g, s, num_filters):
    print(g.shape, s.shape)
    Wg = Conv2D(num_filters, 1, padding="same")(g)
    Wg = BatchNormalization()(Wg)
 
    Ws = Conv2D(num_filters, 1, padding="same")(s)
    Ws = BatchNormalization()(Ws)
 
    out = Activation("relu")(Wg + Ws)
    out = Conv2D(num_filters, 1, padding="same")(out)
    out = Activation("sigmoid")(out)
 
    return out * s
def attention_gate_new(g, s, num_filters):
    # print(g.shape, s.shape,'before')
    Wg = Conv2D(num_filters, 1,strides=(2,2), padding="same")(g)
    Wg = BatchNormalization()(Wg)
 
    Ws = Conv2D(num_filters, 1 ,padding="same")(s)
    Ws = BatchNormalization()(Ws)
    # print(Wg.shape,Ws.shape,'after')
    out = Activation("relu")(Wg + Ws)
    out = Conv2D(num_filters, 1, padding="same")(out)
    out = Activation("sigmoid")(out)
 
    return out * s
def create_model(img_height, img_width, input_chanels, out_classes, starting_filters):
    input_layer = tf.keras.layers.Input((img_height, img_width, input_chanels))

    print('Starting DUCK-Net')

    p1 = Conv2D(starting_filters * 2, 2, strides=2, padding='same')(input_layer)
    p2 = Conv2D(starting_filters * 4, 2, strides=2, padding='same')(p1)
    p3 = Conv2D(starting_filters * 8, 2, strides=2, padding='same')(p2)
    p4 = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(p3)
    p5 = Conv2D(starting_filters * 32, 2, strides=2, padding='same')(p4)

    t0 = conv_block_2D(input_layer, starting_filters, 'duckv2', repeat=1)

    l1i = Conv2D(starting_filters * 2, 2, strides=2, padding='same')(t0)
    s1 = add([l1i, p1])
    t1 = conv_block_2D(s1, starting_filters * 2, 'duckv2', repeat=1)

    l2i = Conv2D(starting_filters * 4, 2, strides=2, padding='same')(t1)
    s2 = add([l2i, p2])
    t2 = conv_block_2D(s2, starting_filters * 4, 'duckv2', repeat=1)

    l3i = Conv2D(starting_filters * 8, 2, strides=2, padding='same')(t2)
    s3 = add([l3i, p3])
    t3 = conv_block_2D(s3, starting_filters * 8, 'duckv2', repeat=1)

    l4i = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(t3)
    s4 = add([l4i, p4])
    t4 = conv_block_2D(s4, starting_filters * 16, 'duckv2', repeat=1)

    l5i = Conv2D(starting_filters * 32, 2, strides=2, padding='same')(t4)
    s5 = add([l5i, p5])
    t51 = conv_block_2D(s5, starting_filters * 32, 'resnet', repeat=2)
    t53 = conv_block_2D(t51, starting_filters * 16, 'resnet', repeat=2)
    t53=attention_gate_new(t4,t53,starting_filters*16)
    l5o = UpSampling2D((2, 2), interpolation=interpolation)(t53)
    # l5o = attention_gate(l5o, t4, starting_filters*16)
    print(l5o.shape, t4.shape)
    c4 = add([l5o, t4])
    q4 = conv_block_2D(c4, starting_filters * 8, 'duckv2', repeat=1)
    q4=attention_gate_new(t3,q4,starting_filters*8)
    l4o = UpSampling2D((2, 2), interpolation=interpolation)(q4)
    # l4o = attention_gate(l4o, t3, starting_filters*8)

    c3 = add([l4o, t3])
    q3 = conv_block_2D(c3, starting_filters * 4, 'duckv2', repeat=1)
    q3=attention_gate_new(t2,q3,starting_filters*4)
    l3o = UpSampling2D((2, 2), interpolation=interpolation)(q3)
    # l3o = attention_gate(l3o, t2, starting_filters*4)

    c2 = add([l3o, t2])
    q6 = conv_block_2D(c2, starting_filters * 2, 'duckv2', repeat=1)
    q6=attention_gate_new(t1,q6,starting_filters*2)
    l2o = UpSampling2D((2, 2), interpolation=interpolation)(q6)
    # l2o = attention_gate(l2o, t1, starting_filters*2)

    c1 = add([l2o, t1])
    q1 = conv_block_2D(c1, starting_filters, 'duckv2', repeat=1)
    q1=attention_gate_new(t0,q1,starting_filters)
    l1o = UpSampling2D((2, 2), interpolation=interpolation)(q1)
    # l1o = attention_gate(l1o, t0, starting_filters)

    c0 = add([l1o, t0])
    z1 = conv_block_2D(c0, starting_filters, 'duckv2', repeat=1)

    output = Conv2D(out_classes, (1, 1), activation='sigmoid')(z1)

    model = Model(inputs=input_layer, outputs=output)

    return model