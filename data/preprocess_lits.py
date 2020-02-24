""" Script to preprocess the LiTS dataset.
    - Load nifti
    - normalize to uniform depth resolution using interpolation
    - save every 3 adjacent slices into .npy
"""

import nibabel as nib
import numpy as np
import os
import glob
import re

def main():
    data_challenge = 'D:/michal/Liver Data LiTS challenge/Testing Batch/' #  'Training Batch 2' ;  'Testing Batch'
    output_folder = 'D:/michal/Liver Data LiTS challenge/data/LiTS_challenge/'

    load_challenge_data(data_challenge, output_folder)

def neuroToRadio(vol, flip_flag):
    """ Change from neurological to radiological orientation. """
    vol = np.transpose(vol, axes=(1, 0, 2))
    if flip_flag:
        vol = np.fliplr(vol)
    vol = np.flipud(vol)

    return vol

def load_challenge_data(src_path, dst_path):
    # Config Params
    RGB_FLAG = 1  # build an image using adjacent slices
    LARGE_SEG_FLAG = 0  # use only images with segmentation

    # map the inputs to the function blocks
    if 'Training' in src_path:
        TEST_SET = 0
        vol_output_folder = os.path.join(dst_path, 'ct_p')
        seg_output_folder = os.path.join(dst_path, 'seg_p')
    elif 'Testing' in src_path:
        TEST_SET = 1
        vol_output_folder = os.path.join(dst_path, 'ct_test_p')
        seg_output_folder = os.path.join(dst_path, 'ct_test_p')
    else:
        ValueError('Source path must contain "Training" or "Testing" to infer in preprocessing. Dont change LiTS folder names!')

    # create output folders
    if not os.path.exists(vol_output_folder):
        os.makedirs(vol_output_folder)
    if not os.path.exists(seg_output_folder):
        os.makedirs(seg_output_folder)

    files_name = glob.glob(src_path+'*volume*.nii')

    num_files = len(files_name)

    for i,file_name  in enumerate(files_name):

        ind = int(''.join(re.findall('\d',os.path.basename(file_name))))
        print('Loading file number: ' + str(ind) + ' out of: ' + str(num_files - i))

        vol_data = nib.load(file_name)
        if not TEST_SET:
            cur_seg_name = file_name.replace('volume', 'segmentation')
            seg_data = nib.load(cur_seg_name)

        neuro_flip_flag = 0
        vol = neuroToRadio(vol_data.get_data(), neuro_flip_flag)
        if not TEST_SET:
            seg = neuroToRadio(seg_data.get_data(), neuro_flip_flag)
        slice_ind = 0
        for j in range(0, vol.shape[2]):
            slice_ind = slice_ind + 1
            if not TEST_SET:
                cur_seg = seg[:, :, j]

            if not LARGE_SEG_FLAG or sum(cur_seg[:]) > 1000:
                if not TEST_SET:
                    seg_im = cur_seg

                if RGB_FLAG:
                    if j == 0:
                        cur_im = np.concatenate((np.expand_dims(vol[:, :, j], axis=2), np.expand_dims(vol[:, :, j], axis=2),
                                                 np.expand_dims(vol[:, :, j + 1], axis=2)), axis=2)
                    elif j == vol.shape[2] - 1:
                        cur_im = np.concatenate((np.expand_dims(vol[:, :, j - 1], axis=2),
                                                 np.expand_dims(vol[:, :, j], axis=2),
                                                 np.expand_dims(vol[:, :, j], axis=2)), axis=2)
                    else:
                        cur_im = np.concatenate((np.expand_dims(vol[:, :, j - 1], axis=2),
                                                 np.expand_dims(vol[:, :, j], axis=2),
                                                 np.expand_dims(vol[:, :, j + 1], axis=2)), axis=2)

                else:
                    cur_im = vol[:, :, j]
                    cur_im = np.concatenate((cur_im, cur_im, cur_im), axis=2)

                np.save(vol_output_folder + '/ct_' + str(ind) + '_' + str(slice_ind) + '.npy', cur_im.astype('float32'))
                if not TEST_SET:
                    np.save(seg_output_folder + '/seg_' + str(ind) + '_' + str(slice_ind) + '.npy', seg_im.astype('int16'))


if __name__ == '__main__':
    main()