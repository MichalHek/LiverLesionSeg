"""
Script for evaluating liver network
Define the model path where the model's weights are saved + liver data path (generated via data/preprocess_lits.py).
Turn isDbg flag on to inspect results.
"""
import os
import numpy as np
import cv2
from keras.utils import Progbar
from medpy import metric
import tensorflow as tf
from LiverLesionSeg.model.ConfigClass import ConfigClass
from LiverLesionSeg.utils import utils
######################################################################################
from LiverLesionSeg.model.model_architectures import get_model_SEresnet50 as get_model
######################################################################################
DEVICE = "/gpu:0"

# Model path
net_dir = '../train/logs/liver_seg/experiment_01/'
# Data paths
data_path = 'D:/michal/Liver Data LiTS challenge/data/LiTS_challenge/ct_p_samp_all_liver'
masks_path = 'D:/michal/Liver Data LiTS challenge/data/LiTS_challenge/seg_p_samp_all_liver'

# Load Config from log dir
log_info_path = net_dir + 'log_info.json'
Config = ConfigClass(log_info_path)

# Weights Path
weights_path = net_dir + 'weights.h5'

# Results output log path
results_filepath = net_dir + '/info.txt'

# Dbg?
isDbg = False

if isDbg:
    dbg_step = 1 # step between images
    min_dice_dbg = 0 #0.4

# params
orig_height, orig_width = (512, 512)

dc = utils.DataClass()

# Load data
_, val_filenames = utils.split_filenames_train_val(data_path, val_prec=0.15)
val_filenames_split = utils.split_to_patients(val_filenames)
val_masks_split = []

for i,val_filename in enumerate(val_filenames_split):
    val_filenames_split[i] = [os.path.join(data_path,filename) for filename in val_filename]
    val_masks_split.append([os.path.join(masks_path,filename).replace('ct', 'seg') for filename in val_filename])
# Display data info
num_patients = len(val_filenames_split)
num_files = len(val_filenames)

# Load model
print('getting model...')
with tf.device(DEVICE):
    model = get_model(Config, inference=True)
print('loading weights...')
model.load_weights(weights_path)
print('Done!')

# Run test on batches
# ===================
# Global
dice_liver = np.zeros(num_patients)
precision_liver = np.zeros(num_patients)
sensitivity_liver = np.zeros(num_patients)

dice_lesion = np.zeros(num_patients)
precision_lesion = np.zeros(num_patients)
sensitivity_lesion = np.zeros(num_patients)

# Local
dice_liver_local = np.zeros(num_files)
precision_liver_local = np.zeros(num_files)
sensitivity_liver_local = np.zeros(num_files)

dice_lesion_local = np.zeros(num_files)
precision_lesion_local = np.zeros(num_files)
sensitivity_lesion_local = np.zeros(num_files)
#######################################

print('len val_filenames: ', num_files)
print('num patients: ', num_patients)

n = 0
a = Progbar(num_files)

for i in range(num_patients):

    val_img_paths = val_filenames_split[i]
    val_mask_paths = val_masks_split[i]
    image_arr = np.zeros((len(val_img_paths), Config.img_height, Config.img_width, Config.img_z))
    mask_arr = np.zeros((len(val_img_paths), orig_height, orig_width))

    begin_idx = n
    end_idx = n + len(val_img_paths)

    for k in range(len(image_arr)):
        n += 1
        a.update(n)

        img = np.load(val_img_paths[k])
        mask = np.load(val_mask_paths[k])

        if not (orig_height == Config.img_height and orig_width == Config.img_z):
            img = cv2.resize(img, (Config.img_height, Config.img_width), interpolation=cv2.INTER_CUBIC)
        img = dc.normalize_img(img)

        image_arr[k] = img
        mask_arr[k] = mask

    pred = model.predict(image_arr)
    treshold = 0.2
    if Config.num_classes ==1:
        pred_seg = np.where((np.squeeze(pred) > treshold), 2, 0)
    else:
        pred_seg = (np.argmax(pred, axis=-1))

    if not (orig_height == Config.img_height and orig_width == Config.img_width):
        seg_new = np.zeros((len(pred_seg), orig_height, orig_width))
        for j in range(len(seg_new)):
            seg_new[j] = cv2.resize(pred_seg[j], (orig_height, orig_width), interpolation=cv2.INTER_NEAREST)
        pred_seg = seg_new

    ## Apply Connected Component
    pred_seg = utils.get_largestCC(pred_seg)
    # seg = utils.get_CC_largerThanTh(seg)

    # liver
    ########
    y_true_liver = np.where(mask_arr >=1 , 1., 0.)
    pred_liver = np.where(pred_seg >= 1, 1., 0.)

    sum_ground_truth = 0
    sum_prediction = 0
    sum_intersection = 0

    j = begin_idx
    for k in range(len(val_img_paths)):
        curr_sum_ground_truth = np.sum(y_true_liver[k])  # float 64
        curr_sum_prediction = np.sum(pred_liver[k])
        curr_sum_intersection = np.sum((pred_liver[k] * y_true_liver[k]))

        # GLOBAL
        sum_ground_truth += curr_sum_ground_truth
        sum_prediction += curr_sum_prediction
        sum_intersection += curr_sum_intersection

        # LOCAL
        dice_liver_local[j] = metric.dc(pred_liver[k], y_true_liver[k])
        precision_liver_local[j] = metric.sensitivity(pred_liver[k], y_true_liver[k])
        sensitivity_liver_local[j] = metric.precision(pred_liver[k], y_true_liver[k])

        if isDbg:
            # Original
            if (j % dbg_step == 0) or dice_liver_local[j] < min_dice_dbg:
                img = utils.dbg_orig_img(data_path, val_img_paths[k])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # Liver - gt
                img = utils.apply_mask(img, mask_arr[k] , (0, 0, 255), mask_idx=1)
                img = utils.apply_mask(img, mask_arr[k] , (0, 0, 255), mask_idx=2)
                cv2.putText(img, '+ GT', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100))
                cv2.imshow('image', utils.array_to_img(img, clip=True).astype('uint8'))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # Liver - prediction
                img = utils.apply_mask(img, pred_liver[k], (0, 127, 255), mask_idx=1)
                cv2.putText(img, ' + pred:', (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100))
                cv2.imshow('image', utils.array_to_img(img, clip=True).astype('uint8'))
                print('Dice = ',dice_liver_local[j], '  Precision = ',precision_liver_local[j],'  Sensitivity = ',sensitivity_liver_local[j])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        j += 1

    dice_liver[i] = 2. * sum_intersection / (sum_ground_truth + sum_prediction)  # TODO add +LossFuncs.smooth
    precision_liver[i] = sum_intersection / sum_prediction
    sensitivity_liver[i] = sum_intersection / sum_ground_truth
    print('Dice=', dice_liver[i], ' ;  Precision=', precision_liver[i], ' ;  Sensitivity=',
          sensitivity_liver[i])


######### Results #########################
print(' GLOBAL:')
print('dice_liver = ', np.nanmean(dice_liver))
print('precision_liver = ', np.nanmean(precision_liver))
print('sensitivity_liver = ', np.nanmean(sensitivity_liver))

print(' LOCAL:')
print('dice_liver = ', np.nanmean(dice_liver_local))
print('precision_liver = ', np.nanmean(precision_liver_local))
print('sensitivity_liver = ', np.nanmean(sensitivity_liver_local))

##########################################
####### write results to info file #######
##########################################
if results_filepath:
    print ('writing results into file:', results_filepath)

    text_file = open(results_filepath, "a")
    text_file.write("\n Results\n#######\n")
    text_file.write("GLOBAL:\n")
    text_file.write('dice_liver = '+ str(np.nanmean(dice_liver))+'\n')
    text_file.write('precision_liver = '+ str(np.nanmean(precision_liver))+'\n')
    text_file.write('sensitivity_liver = '+ str(np.nanmean(sensitivity_liver))+'\n')
    text_file.write("\n")
    text_file.write("LOCAL:\n")
    text_file.write('dice_liver = '+ str(np.nanmean(dice_liver_local))+'\n')
    text_file.write('precision_liver = '+ str(np.nanmean(precision_liver_local))+'\n')
    text_file.write('sensitivity_liver = '+ str(np.nanmean(sensitivity_liver_local))+'\n')

    text_file.close()
