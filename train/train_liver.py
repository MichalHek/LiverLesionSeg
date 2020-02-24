"""
Script for training liver
Define training params in the liver_config.json file
"""
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from liverLesionSeg2.model.ConfigClass import ConfigClass
from liverLesionSeg2.utils.DataGenerator import myDataGenerator
from liverLesionSeg2.utils import utils
######################################################################################
from liverLesionSeg2.model.model_architectures import get_model_SEresnet50 as get_model
######################################################################################

# Get config file
config_path = './liver_config.json'
Config = ConfigClass(config_path)
Config.display()

# Define log dir
log_str = 'experiment_01/' # Define experiment name
logdir = './logs/liver_seg/' + log_str

# Load trained lesion network to continue training from last point?
load_model_weights = None

# training params
val_prec = 0.15
initial_epoch = 0
encoder_weights = 'imagenet' # None- train model from scrath ; imagenet - load pretrained imagenet weights

monitor = 'val_loss'
monitor_mode = 'min'

# Load data
print('Loading LiTS data!!!!!\n')
data_path = Config.data_path
masks_path = Config.labels_path

train_filenames, val_filenames = utils.split_filenames_train_val(data_path, val_prec=val_prec)

# Split validation and training paths
train_img_paths = [os.path.join(data_path, filename) for filename in train_filenames]
train_mask_paths = [os.path.join(masks_path, filename.replace('ct', 'seg')) for filename in train_filenames]
val_img_paths = [os.path.join(data_path, filename) for filename in val_filenames]
val_mask_paths = [os.path.join(masks_path, filename.replace('ct', 'seg'))  for filename in val_filenames]

print('training patients indices: ', utils.get_unique_indices(train_filenames))
print('validation patients indices: ', utils.get_unique_indices(val_filenames))

# Get model
print('\ngetting model...')
model = get_model(Config, encoder_weights=encoder_weights)

if load_model_weights:
    print('loading weight:', load_model_weights)
    model.load_weights(load_model_weights, by_name=True)
print('\nDone!\n')

# init data generator
dc = utils.DataClass()
image_datagen = myDataGenerator(rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=False,
    height_shift_range=0.05,
    width_shift_range=0.05,
    data_format='channels_last',
    subtract_mean=Config.mean,
    devide_by_std=Config.std,
    s_p_noise=0.05,
    gray_modify=0.1,
    preprocessing_function=dc.clip_vals)

train_generator = image_datagen.flow_from_paths(train_img_paths,train_mask_paths, batch_size=Config.batch_size, target_size=(Config.img_width, Config.img_height), num_classes= Config.num_classes, interpolation='bicubic', class_mode=Config.class_mode)
val_generator = image_datagen.flow_from_paths(val_img_paths,val_mask_paths, batch_size=Config.batch_size, target_size=(Config.img_width, Config.img_height), num_classes= Config.num_classes, interpolation='bicubic', class_mode=Config.class_mode)

# Write config to log file
Config.write_to_file(filepath=logdir+'/log_info.json', model=model)

# Callbacks
model_checkpoint = ModelCheckpoint(logdir + '/weights.h5', monitor=monitor, save_best_only=True, mode=monitor_mode)
# model_checkpoint2 = ModelCheckpoint(logdir + '/weights_last.h5', monitor='val_loss', mode='min')
model_tensorboard = TensorBoard(log_dir=logdir, write_graph=True, write_images=True)
model_LRSchedule = LearningRateScheduler(utils.step_decay)
reduce_lr = ReduceLROnPlateau(factor=Config.rop_factor, patience=Config.rop_patience, min_lr=Config.rop_min_lr)
model_earlystop = EarlyStopping(monitor=monitor, patience=Config.es_patience, mode=monitor_mode)


# Callbacks
# ----------
model_callbacks = []
# Checkpoint
model_callbacks.append(ModelCheckpoint(logdir + '/weights.h5', monitor=monitor, save_best_only=True, mode=monitor_mode))
# Tensorboard
model_callbacks.append(TensorBoard(log_dir=logdir, write_graph=True, write_images=True))
# Reduce lr
model_callbacks.append(LearningRateScheduler(utils.step_decay))
# LRSchedule
model_callbacks.append(ReduceLROnPlateau(factor=Config.rop_factor, patience=Config.rop_patience, min_lr=Config.rop_min_lr))
# Early stop
model_callbacks.append(EarlyStopping(monitor=monitor, patience=10, mode=monitor_mode))


# Train
print('\nTraining')
# fits the model on batches with real-time data augmentation:
model.fit_generator(train_generator,
                    verbose=1,
                    validation_data=val_generator,
                    steps_per_epoch=len(train_img_paths) / Config.batch_size,
                    validation_steps=len(val_img_paths) / Config.batch_size,
                    epochs=Config.epochs,
                    callbacks=model_callbacks,
                    initial_epoch=initial_epoch)

print('done training')
