import cv2
import numpy as np
import os
from scipy import ndimage

from keras.utils import Progbar
from liverLesionSeg2.model.model_architectures import get_model_SEresnet50 as get_model
from liverLesionSeg2.model.ConfigClass import ConfigClass
from liverLesionSeg2.utils import utils

src_dir = 'D:/michal/Liver Data LiTS challenge/data/LiTS_challenge/'
orig_height, orig_width = (512, 512)
num_channels = 3

# Model weights path
net_dir = 'D:/michal/liverLesions/LesionNet/train/logs/segmentation/liver/lesion_seg_SEresnet50/imagenet_dice_02/'
weights_path = net_dir + 'weights.h5'

# Load Config from log dir
config_path = net_dir + 'log_info.json'
Config = ConfigClass(config_path)
liver_crop_h, liver_crop_w = (Config.img_height, Config.img_width)

# create output paths
data_path = src_dir + 'ct_test_p' # all test
dst_data_path = src_dir + 'ct_liver_crops_test'
dst_masks_path = src_dir + 'seg_liver_crops_test'
if not os.path.exists(dst_data_path):
    os.mkdir(dst_data_path)
if not os.path.exists(dst_masks_path):
    os.mkdir(dst_masks_path)

# Init liver crops dict
crop_dict = {}

# Load data
print('Running liver detection on data!!!!!\n')
filenames, _ = utils.split_filenames_train_val(data_path, val_prec=0)
all_filenames_split = utils.split_to_patients(filenames)

# Display data info
num_patients = len(all_filenames_split)
num_files = sum([len(x) for x in all_filenames_split])
print('number of samples: ', num_files)
print('num patients: ', num_patients)

# Load liver segmentation model + weights
print('\ngetting liver model...')
model_liver = get_model(Config, inference=True)
print('loading liver weight:', weights_path)
model_liver.load_weights(weights_path)
print('\nDone!\n')

## Run model & extract liver ROIs
#####################
n = 0
a = Progbar(num_files)
dc = utils.DataClass()

for i in range(num_patients):
    # load data to array
    img_filenames = all_filenames_split[i]
    mask_arr = np.zeros((len(img_filenames), orig_width, orig_height))
    img_arr = np.zeros((len(img_filenames), liver_crop_w, liver_crop_h, num_channels))
    for k, filename in enumerate(img_filenames):
        img = np.load(os.path.join(data_path, filename))
        if not (orig_height == liver_crop_h and orig_width == liver_crop_w):
            img = cv2.resize(img, (liver_crop_h, liver_crop_w), interpolation=cv2.INTER_CUBIC)
            img = dc.normalize_img(img) # TODO: Attention to normaliztion!!!!
        img_arr[k] = img

    # run model
    pred = model_liver.predict(img_arr, verbose=1)
    treshold = 0.2
    seg = np.where((np.squeeze(pred) > treshold), 1, 0)

    # resize to original size + post processing
    for k, filename in enumerate(img_filenames):
        mask_liver = seg[k]
        if mask_liver.max() > 0: # liver exists
            if not (orig_height == liver_crop_h and orig_width == liver_crop_w):
                mask_liver = cv2.resize(mask_liver, (orig_height, orig_width), interpolation=cv2.INTER_NEAREST)

                # post processing
                mask_liver = ndimage.binary_fill_holes(mask_liver).astype(int)
                mask_liver = ndimage.binary_closing(mask_liver, structure=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), iterations=1).astype(int)

            mask_arr[k] = mask_liver

    # 3D CC
    seg_liver_CC = utils.get_largestCC(mask_arr, dbg=False)

    # 3-D Crop coordinates
    (h1, h2, w1, w2) = utils.get_crop_coordinates_3D(seg_liver_CC, dbg=False)

    # Crop in 3D
    crop_img_arr = img_arr[:, h1:h2, w1:w2, :]
    crop_mask_arr = seg_liver_CC[:, h1:h2, w1:w2]

    for k, filename in enumerate(img_filenames):
        SAVE_CURR = True
        mask_liver = crop_mask_arr[k]
        if mask_liver.max() > 0:  # liver exists

            crop_img = crop_img_arr[k]
            crop_dict[filename] = (h1, h2, w1, w2)

            np.save(os.path.join(dst_data_path, filename), crop_img)
            np.save(os.path.join(dst_masks_path, filename.replace('ct', 'seg')), mask_liver)

            img_name = filename.replace('npy', 'png')
            cv2.imwrite(os.path.join(dst_data_path, img_name), utils.array_to_img(crop_img, clip=True).astype('uint8'))
            cv2.imwrite(os.path.join(dst_masks_path, img_name.replace('ct', 'seg')), utils.array_to_img(mask_liver, classification=True, num_classes=3))

        n += 1
        a.update(n)

    # Save crops data dict
    try:
        import cPickle as pickle
    except ImportError:  # python 3.x
        import pickle

    with open(dst_masks_path+'/crop_list.p', 'wb') as fp:
        pickle.dump(crop_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    fp.close()
