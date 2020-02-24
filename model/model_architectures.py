"""
get a specific model out of the following option:
get_unet- classic unet
get_model_resnet_* - resnet model
get_model_SEresnet_* - squeeze excitatin resnet model

===========================================================================
We are using the Segmentation Models  package written by Pavel Yakubovskiy:
https://github.com/qubvel/segmentation_models
===========================================================================
"""

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Conv2DTranspose, Dropout, Activation, GlobalMaxPool2D, Lambda, LeakyReLU, AveragePooling2D, UpSampling2D, add, multiply
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K
import tensorflow as tf

############################################################
#  Layers
############################################################

def conv2d_block(input_tensor, filters, kernel_size=(3,3), padding="same", kernel_initializer='glorot_normal', kernel_regularizer='l2',  batchnorm=True, activation="LeakyReLU"): #"relu
    # first layer
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation == "LeakyReLU":
        x = LeakyReLU(alpha=0.01)(x)
    else:
        x = Activation(activation=activation)(x)
    # second layer
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation == "LeakyReLU":
        x = LeakyReLU(alpha=0.01)(x)
    else:
        x = Activation(activation=activation)(x)
    return x

def AttnGatingBlock(x, g, inter_shape, name):
    ''' take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same geature channels (theta_x)
    then, upsample g to be same size as x
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''

    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name='xl' + name)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same', name='g_up' + name)(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same', name='psi' + name)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3], name)
    y = multiply([upsample_psi, x], name='q_attn' + name)

    result = Conv2D(shape_x[3], (1, 1), padding='same', name='q_attn_conv' + name)(y)
    result_bn = BatchNormalization(name='q_attn_bn' + name)(result)
    return result_bn

def expend_as(tensor, rep,name):
	my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep},  name='psi_up'+name)(tensor)
	return my_repeat

def UnetConv2D(input, outdim, is_batchnorm, name):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer='glorot_normal', padding="same", name=name + '_1')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_act')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer='glorot_normal', padding="same", name=name + '_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)
    return x

def UnetGatingSignal(input, is_batchnorm, name):
    ''' this is simply 1x1 convolution, bn, activation '''
    shape = K.int_shape(input)
    x = Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same", kernel_initializer='glorot_normal', name=name + '_conv')(
        input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_act')(x)
    return x

############################################################
#  Architectures
############################################################

def get_model_resnet18(Config, encoder_weights=None, freeze_encoder=False, inference=False):
    BACKBONE = 'resnet18'
    return build_model(BACKBONE, Config, encoder_weights=encoder_weights, freeze_encoder=freeze_encoder,  inference=inference)

def get_model_resnext50(Config, encoder_weights=None, freeze_encoder=False, inference=False):
    BACKBONE = 'resnext50'
    return build_model(BACKBONE, Config, encoder_weights=encoder_weights, freeze_encoder=freeze_encoder,  inference=inference)

def get_model_resnet50(Config, encoder_weights=None, freeze_encoder=False, inference=False):
    BACKBONE = 'resnet50'
    return build_model(BACKBONE, Config, encoder_weights=encoder_weights, freeze_encoder=freeze_encoder,  inference=inference)

def get_model_SEresnet50(Config, encoder_weights=None, freeze_encoder=False, inference=False):
    BACKBONE = 'seresnet50'
    return build_model(BACKBONE, Config, encoder_weights=encoder_weights, freeze_encoder=freeze_encoder,  inference=inference)

def get_model_SEresnet18(Config, encoder_weights=None, freeze_encoder=False, inference=False):
    BACKBONE = 'seresnet18'
    return build_model(BACKBONE, Config, encoder_weights=encoder_weights, freeze_encoder=freeze_encoder,  inference=inference)

def get_model_efficientnetb0(Config, encoder_weights=None, freeze_encoder=False, inference=False):
    BACKBONE = 'efficientnetb0'
    return build_model(BACKBONE, Config, encoder_weights=encoder_weights, freeze_encoder=freeze_encoder,  inference=inference)

def get_model_unet_vgg(Config, encoder_weights=None, freeze_encoder=False, inference=False):
    BACKBONE = 'vgg16'
    return build_model(BACKBONE, Config, encoder_weights=encoder_weights, freeze_encoder=freeze_encoder,  inference=inference)

def get_unet(Config):
    inputs = Input(shape=Config.input_shape)
    kernel_regularizer = Config.kernel_regularizer

    conv1 = conv2d_block(inputs, filters=64, kernel_initializer='glorot_normal', kernel_regularizer='l2')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # drop1 = Dropout(rate=0.2)(pool1)

    conv2 = conv2d_block(pool1, filters=128, kernel_initializer='glorot_normal', kernel_regularizer='l2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(rate=0.2)(pool2)

    conv3 = conv2d_block(drop2, filters=256, kernel_initializer='glorot_normal', kernel_regularizer='l2')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # drop3 = Dropout(rate=0.2)(pool3)

    conv4 = conv2d_block(pool3, filters=512, kernel_initializer='glorot_normal', kernel_regularizer='l2')

    deconv1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4)
    conc1 = concatenate([deconv1, conv3], axis=3)
    # pool4 = Dropout(rate=0.2)(conc1)
    conv5 = conv2d_block(conc1, filters=128, kernel_initializer='glorot_normal', kernel_regularizer='l2')

    deconv2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    conc2 = concatenate([deconv2, conv2], axis=3)
    pool5 = Dropout(rate=0.2)(conc2)
    conv6 = conv2d_block(pool5, filters=64, kernel_initializer='glorot_normal', kernel_regularizer='l2')

    deconv3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    conc3 = concatenate([deconv3, conv1], axis=3)
    pool6 = Dropout(rate=0.2)(conc3)
    conv7 = conv2d_block(pool6, filters=32, kernel_initializer='glorot_normal', kernel_regularizer='l2')

    outputs = Conv2D(Config.num_classes, (1, 1), activation='sigmoid', kernel_initializer='glorot_normal', kernel_regularizer='l2')(conv7)

    model = Model(inputs=inputs, outputs=outputs)

    if Config.type == 'SEGMENTATION':
        model.compile(optimizer=Adam(lr=Config.init_lr), loss=dice_coef_loss, metrics=[dice_coef, 'accuracy'])

    if Config.type == 'PROBABILITY':
        class_weights = Config.weights
        model.compile(optimizer=Adam(lr=Config.init_lr), loss=weighted_pixelwise_crossentropy(class_weights), metrics=['accuracy', dice_coef_liver, dice_coef_lesion])
    return model

def attentionUnetPyramid(Config, freeze_encoder=None, inference=False):
    """ inspired by:
    'A NOVEL FOCAL TVERSKY LOSS FUNCTION WITH IMPROVED ATTENTION U-NET FOR LESION SEGMENTATION' by Nabila Abraham
    https://arxiv.org/pdf/1810.07842.pdf
    """
    Config.class_mode = 'pyramid'

    img_input = Input(shape=Config.input_shape, name='input_scale1')
    scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(img_input)
    scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)
    scale_img_4 = AveragePooling2D(pool_size=(2, 2), name='input_scale4')(scale_img_3)

    conv1 = UnetConv2D(img_input, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    input2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_scale2')(scale_img_2)
    input2 = concatenate([input2, pool1], axis=3)
    conv2 = UnetConv2D(input2, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    input3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv_scale3')(scale_img_3)
    input3 = concatenate([input3, pool2], axis=3)
    conv3 = UnetConv2D(input3, 128, is_batchnorm=True, name='conv3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    input4 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv_scale4')(scale_img_4)
    input4 = concatenate([input4, pool3], axis=3)
    conv4 = UnetConv2D(input4, 64, is_batchnorm=True, name='conv4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    center = UnetConv2D(pool4, 512, is_batchnorm=True, name='center')

    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu',
                                       kernel_initializer='glorot_normal')(center), attn1], name='up1')

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate(
        [Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='glorot_normal')(up1),
         attn2], name='up2')

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate(
        [Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='glorot_normal')(up2),
         attn3], name='up3')

    up4 = concatenate(
        [Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='glorot_normal')(up3),
         conv1], name='up4')

    conv6 = UnetConv2D(up1, 256, is_batchnorm=True, name='conv6')
    conv7 = UnetConv2D(up2, 128, is_batchnorm=True, name='conv7')
    conv8 = UnetConv2D(up3, 64, is_batchnorm=True, name='conv8')
    conv9 = UnetConv2D(up4, 32, is_batchnorm=True, name='conv9')

    out6 = Conv2D(1, (1, 1), activation='sigmoid', name='pred1')(conv6)
    out7 = Conv2D(1, (1, 1), activation='sigmoid', name='pred2')(conv7)
    out8 = Conv2D(1, (1, 1), activation='sigmoid', name='pred3')(conv8)
    out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)

    model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])

    if inference:
        return model

    loss = {'pred1': focal_tversky,
            'pred2': focal_tversky,
            'pred3': focal_tversky,
            'final': focal_tversky}

    loss_weights = {'pred1': 1,
                    'pred2': 1,
                    'pred3': 1,
                    'final': 1}

    model.compile(optimizer=SGD(lr=0.01, momentum=0.9, decay=1e-6), loss=loss, loss_weights=loss_weights,
                  metrics=[dice_coef])
    return model


def attentionUnet(Config, freeze_encoder=None, inference=False):
    Config.class_mode = 'pyramid'

    img_input = Input(shape=Config.input_shape, name='input_scale1')

    conv1 = UnetConv2D(img_input, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = UnetConv2D(pool1, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 128, is_batchnorm=True, name='conv3')
    # conv3 = Dropout(0.2,name='drop_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 64, is_batchnorm=True, name='conv4')
    # conv4 = Dropout(0.2, name='drop_conv4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    center = UnetConv2D(pool4, 512, is_batchnorm=True, name='center')

    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu',
                                       kernel_initializer='glorot_normal')(center), attn1], name='up1')

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate(
        [Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='glorot_normal')(up1),
         attn2], name='up2')

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate(
        [Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='glorot_normal')(up2),
         attn3], name='up3')

    up4 = concatenate(
        [Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='glorot_normal')(up3),
         conv1], name='up4')

    conv6 = UnetConv2D(up1, 256, is_batchnorm=True, name='conv6')
    conv7 = UnetConv2D(up2, 128, is_batchnorm=True, name='conv7')
    conv8 = UnetConv2D(up3, 64, is_batchnorm=True, name='conv8')
    conv9 = UnetConv2D(up4, 32, is_batchnorm=True, name='conv9')

    out6 = Conv2D(1, (1, 1), activation='sigmoid', name='pred1')(conv6)
    out7 = Conv2D(1, (1, 1), activation='sigmoid', name='pred2')(conv7)
    out8 = Conv2D(1, (1, 1), activation='sigmoid', name='pred3')(conv8)
    out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)

    model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])

    loss = {'pred1': focal_tversky,
            'pred2': focal_tversky,
            'pred3': focal_tversky,
            'final': focal_tversky}

    loss_weights = {'pred1': 1,
                    'pred2': 1,
                    'pred3': 1,
                    'final': 1}
    model.compile(optimizer=Adam(lr=0.001), loss=loss, loss_weights=loss_weights,
                  metrics=[dice_coef])
    return model


############################################################
#  Build model
############################################################

def build_model(BACKBONE, Config, encoder_weights=None, freeze_encoder=False, inference=False):
    from segmentation_models import Unet

    if Config.num_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    # define model
    model = Unet(BACKBONE, input_shape=Config.input_shape, classes=Config.num_classes, activation=activation, encoder_weights=encoder_weights, encoder_freeze=freeze_encoder)

    # Compile (Training mode)
    if inference:
        return model
    else:
        if Config.class_mode == 'liver' or Config.class_mode == 'lesion':
            loss = focal_tversky
            # optimizer = 'Adam'
            optimizer = SGD(lr=Config.init_lr, momentum=0.9, decay=1e-4, nesterov=False)
            metrics = [dice_coef]

        elif Config.class_mode == 'liver_lesion':
            loss = weighted_categorical_crossentropy(Config.weights)
            # loss = combined_dice_wp_crossentropy(Config.weights)
            optimizer = 'Adam'
            # optimizer = SGD(lr=Config.init_lr, momentum=0.9, decay=1e-4, nesterov=False)
            metrics = ['accuracy', dice_coef_liver, dice_coef_lesion]

        elif Config.class_mode == 'lesion_combined':
            loss1 = weighted_categorical_crossentropy(Config.weights)
            loss2 = dice_coef_loss
            # loss2 = tversky_loss
            loss ={'softmax_output': loss1, 'sigmoid_output': loss2}# loss_weights
            optimizer = 'Adam'
            metrics = {'softmax_output': [dice_coef_liver, dice_coef_lesion],
                        'sigmoid_output': [dice_coef, dice_coef_lesion]}
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            return model

        elif Config.class_mode == 'liver_lesion_pyramid':
            out3 = model.layers[-1].output
            out2 = model.get_layer('decoder_stage3_relu2').output
            out2 = Conv2D(Config.num_classes, (1, 1), activation='sigmoid', name='pred2')(out2)
            out1 = model.get_layer('decoder_stage2_relu2').output
            out1 = Conv2D(Config.num_classes, (1, 1), activation='sigmoid', name='pred1')(out1)
            out0 = model.get_layer('decoder_stage1_relu2').output
            out0 = Conv2D(Config.num_classes, (1, 1), activation='sigmoid', name='pred0')(out0)

            model = Model(inputs=[model.input], outputs=[out0, out1, out2, out3])

            loss = {'pred0': weighted_categorical_crossentropy(Config.weights),
                    'pred1': weighted_categorical_crossentropy(Config.weights),
                    'pred2': weighted_categorical_crossentropy(Config.weights),
                    'softmax': weighted_categorical_crossentropy(Config.weights)}

            optimizer = 'Adam'
            metrics = [dice_coef_lesion]

        else:
            ValueError(' Please define a valid class_mode in config: liver / lesion / liver_lesion / lesion_pyramid / liver_lesion_pyramid / lesion_combined')

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

############################################################
#  Loss Functions
############################################################
def combined_dice_wp_crossentropy(weights):
    def loss(y_true, y_pred):
        Kweights = K.constant(weights)
        if not K.is_tensor(y_pred): y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)

        wcce_loss = K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
        dice_loss = dice_lesion_loss(y_true, y_pred)

        return 0.5*wcce_loss + 0.5*dice_loss

    return loss

def weighted_categorical_crossentropy(weights):
    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        if not K.is_tensor(y_pred): y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce

def weighted_pixelwise_crossentropy(class_weights):
    def loss(y_true, y_pred):
        epsilon = tf.convert_to_tensor(1e-8, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights))

    return loss

def tversky(y_true, y_pred):
    smooth = 1
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)
################################################################
# Metric function
################################################################
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

################################################################
# Custom Loss funtion
################################################################

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def dice_lesion_loss(y_true, y_pred):
    return 1.-dice_coef_lesion(y_true, y_pred)

def combined_dice_crossentropy(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + K.categorical_crossentropy(target=y_true, output=y_pred)

def dice_coef_liver(y_true, y_pred, class_idx=1):
    # calculates dice per class- default: liver class
    return dice_coef(y_true[:, :, :, class_idx], y_pred[:, :, :, class_idx])

def dice_coef_lesion(y_true, y_pred, class_idx=2):
    # calculates dice per class- default: lesion class
    return dice_coef(y_true[:, :, :, class_idx], y_pred[:, :, :, class_idx])

def dice_coef_lesion(y_true, y_pred, class_idx=2):
    # calculates dice per class- default: cyst class
    return dice_coef(y_true[:, :, :, class_idx], y_pred[:, :, :, class_idx])

def sum_dice_lesion_loss(y_true, y_pred):
    return 1.-(0.2*dice_coef_liver(y_true, y_pred)+0.8*dice_coef_lesion(y_true, y_pred))

########################################
# Generalized Losses
########################################

def dice_coef_multilabel(y_true, y_pred, numLabels=5):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,index,:,:,:], y_pred[:,index,:,:,:])
    return dice


