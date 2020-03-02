"""
Script for exporting model's results into niftii files to submit to LiTS challenge.
Define the model path where the model's weights are saved + liver crops path (generated via data/generate_liver_crops_test.py).
"""
import os
import numpy as np
import cv2
from LiverLesionSeg.model.ConfigClass import ConfigClass
from LiverLesionSeg.utils import utils
######################################################################################
from LiverLesionSeg.model.model_architectures import get_model_SEresnet50 as get_model
######################################################################################
isDbg = False

# Model path
net_dir = '../train/logs/lesion_seg/experiment_04/'

# Data paths
data_path = 'D:/michal/Liver Data LiTS challenge/data/LiTS_challenge/ct_test_p'
masks_path = None
liver_crops_dir = 'D:/michal/Liver Data LiTS challenge/data/LiTS_challenge/ct_liver_test_resnet'

# dst directory- where files are exported to
nifti_output_dir = 'D:/michal/Liver Data LiTS challenge/data/LiTS_results/results_experiment_04'

# Load Config from log dir
log_info_path = net_dir + 'log_info.json'
Config = ConfigClass(log_info_path)

# Weights Path
weights_path = net_dir + 'weights.h5'

# Results output log path
results_filepath = net_dir + '/info.txt'
results_additional_text = weights_path

# params
smooth = 1
orig_height, orig_width = (512, 512)
liver_crop_h, liver_crop_w = (320, 320)

dc = utils.DataClass()

# Load data
val_filenames_split = utils.get_filenames_by_patients(data_path)
# Display data info
num_files = sum([len(x) for x in val_filenames_split])
num_patients = len(val_filenames_split)
print('len val_filenames: ', num_files)
print('num patients: ', num_patients)

# Load liver crops info
dict_file = liver_crops_dir.replace('ct', 'seg')+'/crop_list.p'
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
with open(dict_file, 'rb') as fp:
    crop_dict = pickle.load(fp)
masks_crop_dir = liver_crops_dir.replace('ct', 'seg')

# Load model
print('Getting lesion model...')
model = get_model(Config, inference=True)

print('loading weights...')
model.load_weights(weights_path)
print('Done!')

def get_crops_with_liver(filenames):
    idx_list = []
    filenames_with_liver = []
    for k, filename in enumerate(filenames):
        liver_crop_path = os.path.join(liver_crops_dir, filename)
        if os.path.exists(liver_crop_path):
            idx_list.append(k)
            filenames_with_liver.append(filename)
    return filenames_with_liver, idx_list

for i in range(num_patients):
    # Get data paths
    val_img_filenames = val_filenames_split[i]
    val_filenames_with_liver, idx_list = get_crops_with_liver(val_img_filenames)
    val_paths_with_liver = [os.path.join(liver_crops_dir, filename) for filename in val_filenames_with_liver]
    val_masks_with_liver = [liver_crop_path.replace('ct', 'seg') for liver_crop_path in val_paths_with_liver]

    # test one image to get size
    liver_crop = np.load(os.path.join(liver_crops_dir, val_filenames_with_liver[0]))
    curr_liver_crop_w, curr_liver_crop_h, _ = liver_crop.shape

    # init arrays
    image_arr = np.zeros((len(idx_list), liver_crop_w, liver_crop_h, Config.img_z))
    mask_arr = np.zeros((len(idx_list), curr_liver_crop_w, curr_liver_crop_h))
    seg_lesion_arr = np.zeros((len(val_img_filenames), orig_width, orig_height)).astype('uint8')

    for k, liver_crop_path in enumerate(val_paths_with_liver):
        # Load data
        liver_crop = np.load(liver_crop_path)
        liver_label = np.load(val_masks_with_liver[k])

        if not (liver_crop_h == curr_liver_crop_h and liver_crop_w == curr_liver_crop_w):
            liver_crop = cv2.resize(liver_crop, (liver_crop_h, liver_crop_w), interpolation=cv2.INTER_CUBIC)
        liver_crop = dc.normalize_img(liver_crop)

        image_arr[k] = liver_crop
        mask_arr[k] = liver_label

    pred = model.predict(image_arr, verbose=1)
    multOut = False
    treshold = 0.2
    if len(pred)==2:
        seg_lesion2 = np.squeeze(np.argmax(pred[1], axis=-1))
        pred=pred[0]
        multOut = True
        pred = np.squeeze(np.argmax(pred, axis=-1))
    elif len(pred)==4:
        pred = pred[-1]
        if Config.class_mode == 'pyrmaid':
            pred = np.where((np.squeeze(pred) > treshold), 2, 0)
        elif Config.class_mode == 'liver_lesion_pyramid':
            pred = np.squeeze(np.argmax(pred, axis=-1))
    elif Config.class_mode == 'lesion':
        pred = np.where((np.squeeze(pred) > treshold), 2, 0)
    else:
        pred = np.squeeze(np.argmax(pred, axis=-1))

    # Connected component
    cc_thresh = 50
    seg_CC = utils.get_CC_largerThanTh(np.where(pred == 2, 1, 0), thresh=cc_thresh, dbg=False)
    pred[pred == 2] = 1
    pred[seg_CC == 1] = 2
    results_additional_text = results_additional_text + '\n CC with thresh = ' + str(cc_thresh) + '\n'

    n = 0
    for k in idx_list:
        seg_lesion = pred[n]
        if not (liver_crop_h == curr_liver_crop_h and liver_crop_w == curr_liver_crop_w):
            seg_lesion = cv2.resize(seg_lesion, (curr_liver_crop_h, curr_liver_crop_w), interpolation=cv2.INTER_NEAREST)

        liver_crop_mask = mask_arr[n]
        liver_crop_path = val_filenames_with_liver[n]
        (h1, h2, w1, w2) = crop_dict[liver_crop_path]

        filename = os.path.split(val_img_filenames[k])[-1]

        # Clean results (exclude pixels outside of liver boundary)
        seg_lesion[liver_crop_mask == 0] = 0

        seg_lesion_arr[k, h1:h2, w1:w2] = seg_lesion
        n += 1

    utils.convert_volume_to_nifti(volume_arr=seg_lesion_arr,
                                    filenames=val_img_filenames,
                                    output_dir=nifti_output_dir)

print("Done saving Niftis")

