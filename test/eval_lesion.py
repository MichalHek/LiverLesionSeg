import os
import numpy as np
import cv2
from keras.utils import Progbar
from medpy import metric
import tensorflow as tf
from LiverLesionSeg.model.ConfigClass import ConfigClass
from LiverLesionSeg.utils import utils
############################################################################
from LiverLesionSeg.model.model_architectures import get_model_SEresnet50 as get_model
############################################################################
DEVICE = "/gpu:0"

# Model path
net_dir = '../train/logs/lesion_seg/experiment_04/'
# Data paths
data_path = 'D:/michal/Liver Data LiTS challenge/data/LiTS_challenge/ct_p_samp'
masks_path = 'D:/michal/Liver Data LiTS challenge/data/LiTS_challenge/seg_p_samp'
liver_crops_dir = 'D:/michal/Liver Data LiTS challenge/data/LiTS_challenge/ct_liver_samp' #gt crops

# Load Config from log dir
log_info_path = net_dir + 'log_info.json'
Config = ConfigClass(log_info_path)

# Weights Path
weights_path = net_dir + 'weights.h5'

# Results output log path
results_filepath = net_dir + '/info.txt'
results_additional_text = weights_path
# Save to CSV
csv_results_path = './lesion_seg_results.csv'

# Dbg?
isDbg = False

if isDbg:
    dbg_step = 10 # step between images
    min_dice_dbg = 0 #0.5

# params
smooth = 1
orig_height, orig_width = (512, 512)
liver_crop_h, liver_crop_w = (320, 320)

dc = utils.DataClass()

# Load data
_, val_filenames = utils.split_filenames_train_val(data_path, is_sort=True)
val_filenames_split = utils.split_to_patients(val_filenames)
# Display data info
num_patients = len(val_filenames_split)
num_files = len(val_filenames)

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
with tf.device(DEVICE):
    # Get Lesion model
    print('Getting lesion model...')
    model = get_model(Config, inference=True)
print('loading weights...')
model.load_weights(weights_path)
print('Done!')

# Local Only
dice_lesion = np.zeros(num_files)
precision_lesion = np.zeros(num_files)
sensitivity_lesion = np.zeros(num_files)
print('len val_filenames: ', num_files)
print('num patients: ', num_patients)

n = 0
a = Progbar(num_files)

for i in range(num_patients):
    val_img_filenames = val_filenames_split[i]
    val_mask_filenames = [val_img_filename.replace('ct', 'seg') for val_img_filename in val_img_filenames]

    begin_idx = n
    end_idx = n + len(val_img_filenames)

    seg_lesion_arr = np.zeros((len(val_img_filenames), orig_width, orig_height)).astype('uint8')

    for k, filename in enumerate(val_img_filenames):

        liver_crop_path = os.path.join(liver_crops_dir, filename)
        mask_crop_path = liver_crop_path.replace('ct', 'seg')

        if os.path.exists(liver_crop_path):
            mask = np.load(os.path.join(masks_path, filename.replace('ct', 'seg')))

            # liver model found liver
            liver_crop = np.load(liver_crop_path)
            curr_liver_crop_w, curr_liver_crop_h, _ = liver_crop.shape

            if not (liver_crop_h == curr_liver_crop_h and liver_crop_w == curr_liver_crop_w):
                liver_crop = cv2.resize(liver_crop, (liver_crop_h, liver_crop_w), interpolation=cv2.INTER_CUBIC)

            liver_crop = dc.normalize_img(liver_crop)

            liver_crop_expand = np.expand_dims(liver_crop, axis=0)

            pred = model.predict(liver_crop_expand, verbose=0)
            multOut = False
            treshold = 0.2
            if len(pred)==2:
                seg_lesion2 = np.squeeze(np.argmax(pred[1], axis=-1))
                pred=pred[0]
                multOut = True
                seg_lesion = np.squeeze(np.argmax(pred, axis=-1))
            elif len(pred)==4:
                pred = pred[-1]
                seg_lesion = np.where((np.squeeze(pred) > treshold), 2, 0)
                print(filename, np.sum(seg_lesion == 1))
            elif Config.class_mode == 'lesion':
                seg_lesion = np.where((np.squeeze(pred) > treshold), 2, 0)
            else:
                seg_lesion = np.squeeze(np.argmax(pred, axis=-1))
                print(filename, np.sum(seg_lesion> 1))


            # seg_lesion_final = np.zeros((orig_height, orig_width))
            (h1, h2, w1, w2) = crop_dict[filename]

            if not (liver_crop_h == curr_liver_crop_h and liver_crop_w == curr_liver_crop_w):
                seg_lesion = cv2.resize(seg_lesion, (curr_liver_crop_h, curr_liver_crop_w), interpolation=cv2.INTER_NEAREST)
                if multOut:
                    seg_lesion2 =  cv2.resize(seg_lesion, (curr_liver_crop_h, curr_liver_crop_w), interpolation=cv2.INTER_NEAREST)

            # Clean Result- remove lesion pixels that are outside the liver
            liver_crop_mask = np.load(mask_crop_path)
            if multOut:
                seg_lesion[seg_lesion2==1]=2
            seg_lesion[liver_crop_mask==0] = 0

            try:
                seg_lesion_arr[k, h1:h2, w1:w2] = seg_lesion
            except:
                cv2.imshow('Image', utils.array_to_img(seg_lesion, clip=True).astype('uint8'))

            # lesion
            ########
            seg_lesion_final = np.copy(seg_lesion_arr[k])

            y_true_lesion = np.where(mask == 2, 1., 0.)
            seg_lesion_tmp = np.where(seg_lesion_final == 2, 1., 0.)

            dice_lesion[n] = metric.dc(seg_lesion_tmp,y_true_lesion)
            precision_lesion[n] = metric.sensitivity(seg_lesion_tmp,y_true_lesion)
            sensitivity_lesion[n] = metric.precision(seg_lesion_tmp,y_true_lesion)

            #############################################################################################
            print('dice=', dice_lesion[n], ' ;  precision=', precision_lesion[n], ' ;  sensitivity=',
                  sensitivity_lesion[n])
            if isDbg:
                if (n % dbg_step ==0) or dice_lesion[n]<min_dice_dbg:
                    # Original Image
                    img = utils.dbg_orig_img(data_path, filename)
                    cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # liver
                    if Config.num_classes>1:
                        img = utils.apply_mask(img, mask, (0, 0, 255), mask_idx=1)
                        cv2.putText(img, '+ GT', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 100))
                        cv2.imshow('image', utils.array_to_img(img, clip=True).astype('uint8'))

                    # Liver GT + Lesions GT (I = alpha F + (1 - alpha) B)
                    img = utils.apply_mask(img, mask, (127, 0 , 0), mask_idx=2)
                    cv2.putText(img, '+ GT', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
                    cv2.imshow('image', utils.array_to_img(img, clip=True).astype('uint8'))
                    cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # Liver Patch on Image
                    cv2.putText(img, '+ Liver Crop', (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
                    cv2.rectangle(img, (w1, h1), (w2, h2), (0, 255, 0), 2)
                    cv2.imshow('image', utils.array_to_img(img, clip=True).astype('uint8'))
                    cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # Lesions Predictions
                    if np.sum(seg_lesion_final)>( Config.num_classes-2):
                        img = utils.apply_mask(img, seg_lesion_final, (0, 255, 255), mask_idx= 2)
                        cv2.putText(img, '+ Lesions Predictions', (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255))
                    else:
                        cv2.putText(img, '+ No Lesiond Detected', (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 255))
                    cv2.imshow('image', utils.array_to_img(img, clip=True).astype('uint8'))
                    cv2.waitKey(0)

        else:
            # Ignore if no lesion exist in GT
            dice_lesion[n] = np.nan
            precision_lesion[n] = np.nan
            sensitivity_lesion[n] = np.nan
        ################################################################################################
        n += 1
        a.update(n)

######### Results ########################
Dice = np.nanmean(dice_lesion)
Precision = np.nanmean(precision_lesion)
Recall = np.nanmean(sensitivity_lesion)
F1_score = 2 * (Recall * Precision) / (Recall + Precision)

print('\ndice_lesion = ',Dice)
print('precision_lesion = ',Precision)
print('sensitivity_lesion = ', Recall)
print('F1_score = ', F1_score)


# write results to info file
############################
if results_filepath:
    print ('writing results into file:', results_filepath)

    text_file = open(results_filepath, "a")
    text_file.write("\n Results - LiTS dataset\n#######\n")
    if results_additional_text:
        text_file.write(results_additional_text+'\n')
    text_file.write('\ndice_lesion = ' + str(Dice)+'\n')
    text_file.write('precision_lesion = ' + str(Precision)+'\n')
    text_file.write('sensitivity_lesion = ' + str(Recall)+'\n')
    text_file.write('F1_score = ' + str(F1_score)+'\n')
    text_file.write("\n")
    text_file.close()

# write results to csv for comparison
######################################
if csv_results_path:
    import csv
    if not os.path.isfile(csv_results_path):
        with open(csv_results_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["model", "dice", "precision", "recall", "F1_score"])
            writer.writeheader()
            f.close()

    with open(csv_results_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([weights_path, str(Dice), str(Precision), str(Recall), str(F1_score)])
        f.close()