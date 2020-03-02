"""
Script for training lesion
Define training params in the lesion_config.json file
The input should be liver crops generated via data/generate_liver_crops_train.py
"""
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from LiverLesionSeg.model.ConfigClass import ConfigClass
from LiverLesionSeg.utils.DataGenerator import myDataGenerator
from LiverLesionSeg.utils import utils
######################################################################################
from LiverLesionSeg.model.model_architectures import get_model_SEresnet50 as get_model
######################################################################################

# Get config file
config_path = './lesion_config.json'
Config = ConfigClass(config_path)
Config.display()

# Define log dir
log_str = 'experiment_04/'
logdir = './logs/lesion_seg/' + log_str

# Load trained lesion network to continue training from last point?
load_model_weights = None

# training params
val_prec = 0.15
initial_epoch = 0
encoder_weights = 'imagenet'
freeze_encoder = False

# Tensorboard checkpoint monitoring metrics
monitor = 'val_softmax_dice_coef_lesion'
monitor_mode = 'max'
# Save augmentations for dbg?
save_to_dir = None # 'D:/michal/Liver Data LiTS challenge/data/LiTS_challenge/aug/'

# Load data
print('Loading liver crops for lesion segmentation!!!!!/n')
data_path = Config.data_path
masks_path = Config.labels_path

train_filenames, val_filenames = utils.split_filenames_train_val(data_path, val_prec=val_prec)

# Split validation and training paths
train_img_paths = [os.path.join(data_path, filename) for filename in train_filenames]
train_mask_paths = [os.path.join(masks_path, filename.replace('ct', 'seg')) for filename in train_filenames]
val_img_paths = [os.path.join(data_path, filename) for filename in val_filenames]
val_mask_paths = [os.path.join(masks_path, filename.replace('ct', 'seg')) for filename in val_filenames]

print('training patients indices: ', utils.get_unique_indices(train_filenames))
print('validation patients indices: ', utils.get_unique_indices(val_filenames))

# Get model
print('\ngetting lesion model...')
model_lesion = get_model(Config, freeze_encoder=freeze_encoder)

if load_model_weights:
    print('loading lesion weights')
    model_lesion.load_weights(load_model_weights)
print('\nDone!\n')

# init data generator
dc = utils.DataClass()
image_datagen = myDataGenerator(rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    height_shift_range=0.05,
    width_shift_range=0.05,
    data_format='channels_last',
    subtract_mean=Config.mean,
    devide_by_std=Config.std,
    s_p_noise=0.05,
    gray_modify=0.1,
    preprocessing_function=dc.clip_vals)

train_generator = image_datagen.flow_from_paths(train_img_paths, train_mask_paths, batch_size=Config.batch_size, target_size=(Config.img_width, Config.img_height), num_classes=Config.num_classes, interpolation='bicubic', class_mode=Config.class_mode, save_to_dir=save_to_dir)
val_generator = image_datagen.flow_from_paths(val_img_paths, val_mask_paths, batch_size=Config.batch_size, target_size=(Config.img_width, Config.img_height), num_classes=Config.num_classes, interpolation='bicubic', class_mode=Config.class_mode, save_to_dir=save_to_dir)

# Add info to log
dict_info = {
    'encoder_weights': encoder_weights,
    'load_model_weights': load_model_weights,
    'val_prec': val_prec,
    'freeze_encoder': freeze_encoder
}
# Write config to log file
Config.write_to_file(filepath=logdir+'/log_info.json', model=model_lesion, **dict_info)

# Callbacks
# ----------
model_checkpoint = ModelCheckpoint(logdir + '/weights.h5', monitor=monitor, save_best_only=True, mode=monitor_mode)
# model_checkpoint2 = ModelCheckpoint(logdir + '/weights_last.h5', monitor='val_loss', mode='min')
# model_checkpoint3 = ModelCheckpoint(logdir + '/weights{epoch:05d}.h5',save_weights_only=True, period=5)) # save weights every 5 epochs
model_tensorboard = TensorBoard(log_dir=logdir, write_graph=True, write_images=True)
model_LRSchedule = LearningRateScheduler(utils.step_decay)
reduce_lr = ReduceLROnPlateau(factor=Config.rop_factor, patience=Config.rop_patience, min_lr=Config.rop_min_lr)
model_earlystop = EarlyStopping(monitor=monitor, patience=10, mode=monitor_mode)

model_callbacks = [model_checkpoint,
                   model_tensorboard,
                   model_LRSchedule,
                   reduce_lr,
                   model_earlystop]


# Train
print('/nTraining')
# fits the model on batches with real-time data augmentation:
model_lesion.fit_generator(train_generator,
                           verbose=1,
                           validation_data=val_generator,
                           steps_per_epoch=len(train_img_paths) / Config.batch_size,
                           validation_steps=len(val_img_paths) / Config.batch_size,
                           epochs=Config.epochs,
                           callbacks=model_callbacks,
                           initial_epoch=initial_epoch)

print('done training')
