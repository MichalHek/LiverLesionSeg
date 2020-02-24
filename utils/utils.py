import os
import random
import cv2
import numpy as np
import math
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from skimage.measure import label

from LiverLesionSeg.model.ConfigClass import ConfigClass as Config

############################################################
#  data processing
############################################################

class DataClass():
    def __init__(self):
        self.Config = Config.getInstance()
    # normalization
    def normalize_img(self, img, clip=True, Z_score=True, cont_norm=False, samplewise=False):
        if clip:
            img = self.clip_vals(img)
        if Z_score:
            img -= self.Config.mean
            img /= (self.Config.std + K.epsilon())
        if samplewise:
            if Z_score:
                assert not samplewise == Z_score
            mean = np.mean(img, keepdims=True)
            img -= mean
            std = (np.std(img, keepdims=True) + K.epsilon())
            img /= std
        if cont_norm:
            img /= self.Config.const
        return img


    def clip_vals(self, img):
        try:
            clip_min = self.Config.clip_vals[0]
            clip_max = self.Config.clip_vals[1]
        except:
            print('config clip values are not supplied--> using defaultive values: [-200, 200]')
            clip_min = -200
            clip_max = 200

        img[img < clip_min] = clip_min
        img[img > clip_max] = clip_max
        return img


    def denormalize_image(self, image, Z_score=True, cont_norm=False):
        if Z_score:
            image *= self.Config.std
            image += self.Config.mean
        if cont_norm:
            image *= self.Config.const


def scale_to_img(img):
    img_min = img.min()
    img_max = img.max()
    return ((img - (img_min)) * (1 / (img_max - img_min)) * 255)


def split_filenames_train_val(IMG_PATH, val_prec=0.2, suffix='.npy', is_sort=True):
    img_dir, _, filenams = next(os.walk(IMG_PATH))
    if suffix:
        filenams = [filename for filename in filenams if filename.endswith(suffix)]
    # masks_dir, _, _ = next(os.walk(MASK_PATH))
    if is_sort:
        filenams = sort_filenames(filenams)
        return split_to_val2(filenams, val_prec)
    else:
        return split_to_val(filenams, val_prec, shuffle=False)

def split_K_fold(IMG_PATH, k_index, K=3, val_prec=0.2, suffix='.npy'):
    img_dir, _, filenames = next(os.walk(IMG_PATH))
    if suffix:
        filenames = [filename for filename in filenames if filename.endswith(suffix)]
    N = len(filenames)
    num_test = int(N / K)

    indices = np.arange(N)
    random.Random(4).shuffle(indices)

    idx_test_begin = np.zeros(K)
    idx_test_end = np.zeros(K)
    for i in range(K):
        idx_test_begin[i] = i*num_test
        idx_test_end[i] = min(i*num_test + num_test, N)

    train_filenames = []
    test_filenames = []

    for i in range(N):
        if i < idx_test_begin[k_index-1] or i >= idx_test_end[k_index-1] :
            train_filenames.append(filenames[indices[i]])
        else:
            test_filenames.append(filenames[indices[i]])

    num_val = int(len(train_filenames) * val_prec)
    num_train = len(train_filenames) - num_val

    val_filenames = train_filenames[num_train:]
    train_filenames = train_filenames[:num_train]

    return train_filenames, val_filenames, test_filenames


def get_filenames_by_patients(IMG_PATH,suffix='.npy'):
    img_dir, _, filenams = next(os.walk(IMG_PATH))
    if suffix:
        filenams = [filename for filename in filenams if filename.endswith(suffix)]
    # masks_dir, _, _ = next(os.walk(MASK_PATH))
    filenams = sort_filenames(filenams)
    return filenams


def split_to_val(filenames, val_prec, shuffle=True):
    ''' split files name list into 2 groups (training and vlidtion according to
    val_prec: pecent of data for validation'''
    unique_indcies = get_unique_indices(filenames)
    num_val = int(len(unique_indcies) * val_prec)
    num_train = len(unique_indcies) - num_val
    train_indices = np.arange(num_train)
    # fix bug:
    # train_indices = unique_indcies
    val_indices = np.arange(num_train, num_train + num_val)

    if shuffle:
        # indices = np.random.permutation(len(filenams))
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

    train_filenames = []
    val_filenames = []
    patient_idx_all = split_and_get_idx_all(filenames, 1)
    for i, filename in enumerate(filenames):
        curr_patient = patient_idx_all[i]
        if curr_patient in unique_indcies[:num_train]:
            train_filenames.append(filename)
        else:
            val_filenames.append(filename)
    return train_filenames, val_filenames

def split_to_val2(filenames, val_prec):
    ''' split files name list into 2 groups (training and vlidtion according to
    val_prec: pecent of data for validation'''
    num_val = int(len(filenames) * val_prec)
    num_train = len(filenames) - num_val

    flatten = lambda l: [item for sorted_filenames in l for item in sorted_filenames]
    train_filenames = flatten(filenames[0:num_train])
    val_filenames = flatten(filenames[num_train:])
    return train_filenames, val_filenames


def sort_filenames(filenams):
    patient_indices_unique = get_unique_indices(filenams)

    sorted_filenames = []
    for patient_idx in patient_indices_unique:
        patient_list = get_specific_patient(filenams, 1, patient_idx)
        patient_indices = []
        sorted_slices = []
        for silcename in patient_list:
            patient_indices.append(int(silcename.split('_')[2][:-4]))
        sorted_indices=sorted(range(len(patient_indices)), key=lambda k: patient_indices[k])
        for sorted_idx in sorted_indices:
            sorted_slices.append(patient_list[sorted_idx])
        sorted_filenames.append(sorted_slices)

    return sorted_filenames


def split_and_get_idx_all(list, idx):
    val_list = []
    for val in list:
        try:
            val_list.append(int(val.split('_')[idx]))
        except:
            val_list.append(int((val.split('_')[idx]).split('.')[0]))
    return val_list


def get_specific_patient(filenams, idx, patient_idx):
    patient_list = []
    for filename in filenams:
        try:
            curr_patient_idx = int(filename.split('_')[idx])
        except:
            curr_patient_idx = int((filename.split('_')[idx]).split('.')[0])
        if curr_patient_idx == patient_idx:
            patient_list.append(filename)
    return patient_list


def get_unique_indices(filenams, idx=1):
    patient_indices = split_and_get_idx_all(filenams, idx)
    return list(set(patient_indices))

def split_to_patients(filenames):
    patient_indices_unique = get_unique_indices(filenames)
    sorted_filenames = []
    for patient_idx in patient_indices_unique:
        patient_list = get_specific_patient(filenames, 1, patient_idx)
        sorted_filenames.append(patient_list)

    return (sorted_filenames)


############################################################
#  Training Utility Functions
############################################################

def step_decay(epoch):
    from liverLesionSeg2.model.ConfigClass import ConfigClass
    Config = ConfigClass.getInstance()
    initial_lrate = Config.init_lr
    drop = Config.sd_drop
    epochs_drop = Config.sd_epochs_drop
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print('learning rate = ', lrate)
    return lrate


############################################################
#  Validation Tools
############################################################

def calc_perf(mask_arr, seg, class_label=1, orig_width=512, orig_height=512):
    y_true_liver = np.where(mask_arr == class_label, 1., 0.)
    seg_liver = np.where(seg == class_label, 1., 0.)

    ground_truth_label = np.zeros((len(mask_arr), orig_width * orig_height))
    prediction_label = np.zeros((len(seg), orig_width * orig_height))

    for ii in range(len(seg)):
        ground_truth_label[ii] = np.reshape(y_true_liver[ii], orig_width * orig_height)
        prediction_label[ii] = np.reshape(seg_liver[ii], orig_width * orig_height)

    sum_ground_truth = np.sum(ground_truth_label, axis=1)  # float 64
    sum_prediction = np.sum(prediction_label, axis=1)
    sum_intersection = np.sum((prediction_label * ground_truth_label), axis=1)

    dice = 2. * sum_intersection / (sum_ground_truth + sum_prediction)  # TODO add +LossFuncs.smooth
    precision = sum_intersection / sum_prediction
    sensitivity = sum_intersection / sum_ground_truth

    return dice, precision, sensitivity

############################################################
#  Visualization Tools
############################################################
def apply_mask(image, mask, color, alpha=0.5, mask_idx=1):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == mask_idx,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def array_to_img(x, num_classes=None, scale=True, clip=False, classification=False):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A scaled image of type uint8.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if not num_classes:
        num_classes = x.max() + 1
    x = np.asarray(x, dtype=K.floatx())
    # Original Numpy array x has format (height, width, channel)
    if clip:
        x = DataClass().clip_vals(x)

    if scale:
        x = x + max(-np.min(x), 0)
        if not classification:
            x_max = np.max(x)
        else:
            x_max = num_classes - 1
        if x_max != 0:
            x = x / x_max
        x *= 255

    return x.astype('uint8')


def display_images(images1, images2, cols=4, cmap="gray", classification=False):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    cmap: Optional. Color map to use. For example, "Blues".

    """
    plt.figure(figsize=(14, 14 * (cols + 2) // cols))
    i = 1
    for image1, image2 in zip(images1, images2):
        plt.subplot(cols, 2, i)
        plt.title("H x W={}x{}".format(image1.shape[0], image1.shape[1]), fontsize=9)
        plt.axis('off')
        image1 = array_to_img(image1)
        plt.imshow(image1.astype(np.uint8), cmap=cmap)
        i += 1

        plt.subplot(cols, 2, i)
        plt.title("H x W={}x{}".format(image2.shape[0], image2.shape[1]), fontsize=9)
        plt.axis('off')
        image2 = array_to_img(image2, classification=classification)
        plt.imshow(image2.astype(np.uint8), cmap=cmap)
        i += 1


def display_imgs_gt_pred(images, labels, predictions, cols=4, cmap="gray", classification=False):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    cmap: Optional. Color map to use. For example, "Blues".

    """
    plt.figure(figsize=(14, 14 * (cols + 2) // cols))
    i = 1
    for n in range(len(images)):
        plt.subplot(cols, 3, i)
        plt.title("H x W={}x{}".format(images[n].shape[0], images[n].shape[1]), fontsize=9)
        plt.axis('off')
        image = array_to_img(images[n])
        plt.imshow(image.astype(np.uint8), cmap=cmap)
        i += 1

        plt.subplot(cols, 3, i)
        plt.title("GT: H x W={}x{}".format(labels[n].shape[0], labels[n].shape[1]), fontsize=9)
        plt.axis('off')
        label = array_to_img(labels[n], classification=classification)
        plt.imshow(label.astype(np.uint8), cmap=cmap)
        i += 1

        plt.subplot(cols, 3, i)
        plt.title("Pred: H x W={}x{}".format(predictions[n].shape[0], predictions[n].shape[1]), fontsize=9)
        plt.axis('off')
        prediction = array_to_img(predictions[n], classification=classification)
        plt.imshow(prediction.astype(np.uint8), cmap=cmap)
        i += 1


def resize_array(images, new_size=(512, 512), interpolation=cv2.INTER_CUBIC, normalize=False):
    if len(images.shape) > 3:
        images_new = np.zeros((images.shape[0], new_size[0], new_size[1], images.shape[3]))
    else:
        images_new = np.zeros((images.shape[0], new_size[0], new_size[1]))
    for i, img in enumerate(images):
        if not normalize:
            images_new[i] = cv2.resize(img, new_size, interpolation=interpolation)
        else:
            images_new[i] = DataClass().normalize_img(cv2.resize(img, new_size, interpolation=interpolation))

    return images_new

def crop_liver_from_orig(img, seg_liver, dbg=None):
    crop = crop_liver(img, dbg=dbg, orig=seg_liver)
    return crop

def get_crop_coordinates(img, pad_size=2):
    """ input: binaty @D image
        output: crop coordinates of minimal "1" area with gap padding"""
    im_h, im_w = img.shape
    liver_m, liver_n = np.where(img >= 1)
    h_min = min(liver_m) - pad_size
    h_max = max(liver_m) + pad_size
    h = h_max - h_min + 1
    w_min = min(liver_n) - pad_size
    w_max = max(liver_n) + pad_size
    w = w_max - w_min + 1
    gap = abs(h - w)
    pad_l = int(np.ceil(gap / 2.))
    pad_r = int(np.floor(gap / 2.))
    if h > w:
        w_min -= pad_l
        w_max += pad_r
        if w_min < 0:
            w_min = 0
            w_max += (0 - w_min)
        if w_max > im_w:
            w_min -= w_max - im_w
            w_max = im_w
    if h < w:
        h_min -= pad_l
        h_max += pad_r
        if h_min < 0:
            h_min = 0
            h_max += (0 - h_min)
        if h_max > im_h:
            h_min -= h_max - im_h
            h_max = im_h

    return h_min, h_max, w_min, w_max

def get_crop_coordinates_3D(img_arr, pad_size=1,dbg=False):
    """ input: binaty 3D image
        output: global crop coordinates of minimal "1" area with gap padding"""
    im_d, im_h, im_w = img_arr.shape
    liver_z, liver_h, liver_w = np.where(img_arr >= 1)
    h_min = min(liver_h) - pad_size
    h_max = max(liver_h) + pad_size
    h = h_max - h_min + 1
    w_min = min(liver_w) - pad_size
    w_max = max(liver_w) + pad_size
    w = w_max - w_min + 1
    gap = abs(h - w)
    pad_l = int(np.ceil(gap / 2.))
    pad_r = int(np.floor(gap / 2.))
    if h > w:
        w_min -= pad_l
        w_max += pad_r
        if w_min < 0:
            w_min = 0
            w_max += (0 - w_min)
        if w_max > im_w:
            w_min -= w_max - im_w
            w_max = im_w
    if h < w:
        h_min -= pad_l
        h_max += pad_r
        if h_min < 0:
            h_min = 0
            h_max += (0 - h_min)
        if h_max > im_h:
            h_min -= h_max - im_h
            h_max = im_h
    if dbg:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        num_points = int(np.round(0.02 * len(liver_z)))
        indices = np.random.permutation(len(liver_z))[0:num_points]
        # ax.scatter(liver_h, liver_w, liver_z)
        ax.scatter(liver_h[indices], liver_w[indices],liver_z[indices],s=0.8)
        ax.plot([h_min,h_max,h_max,h_min,h_min],[w_min, w_min,w_max, w_max,w_min], zs=int(im_d / 2), zdir='z',color='black')
        ax.view_init(elev=180., azim=360)
        plt.show()
        plt.ioff()
        plt.waitforbuttonpress()
        plt.close()

    return h_min, h_max, w_min, w_max

def crop_liver(img, dbg=None):
    """ crops liver segmetation image to liver ROI"""
    ''''''' ---horizontal  ,  | vertical '''
    h_min, h_max, w_min, w_max = get_crop_coordinates(img)
    crop_img = img[h_min:h_max, w_min:w_max]

    if dbg:
        crop_vis = np.zeros(crop_img.shape)
        crop_vis[crop_img==1]=255
        cv2.imshow('Image', crop_vis.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(crop_img.shape)

    return crop_img

############################################################
#  NIFTI Conversions
############################################################
def convert_volume_to_nifti(volume_arr,
                            filenames,
                            output_dir,
                            orig_volume_dir='D:/michal/Liver Data LiTS challenge/Testing Batch',
                            test_mode=True):
    import nibabel as nib

    volume_idx = get_unique_indices(filenames)[0]
    slice_indices = [int(filename.split('_')[-1][:-4]) for filename in filenames] # list of existing segmentation slices

    test_vol_filename = 'test-volume-{}.nii'.format(volume_idx)
    # test_vol_filename = 'segmentation-{}.nii'.format(volume_idx)
    if test_mode:
        test_seg_filename = 'test-segmentation-{}.nii'.format(volume_idx)
    else:
        test_seg_filename = 'segmentation-{}.nii'.format(volume_idx)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Processing test volume: ', test_vol_filename, ' --> ', test_seg_filename)

    # Load original volume
    curr_seg_data = nib.load(os.path.join(orig_volume_dir, test_vol_filename))# pointer/copy???
    # seg_vol = curr_seg_data.get_data().astype('uint8')

    seg_vol = np.zeros(curr_seg_data.shape).astype('uint8')

    for i, img in enumerate(volume_arr):
        curr_idx = slice_indices[i]
        # Convert Radio to neuro
        curr_seg = np.fliplr(np.transpose(img))
        seg_vol[:, :, curr_idx-1] = curr_seg

    new_img = nib.Nifti1Image(seg_vol, curr_seg_data.affine, curr_seg_data.header)
    nib.save(new_img, os.path.join(output_dir, test_seg_filename))

############################################################
#  Post Process Tools
############################################################
def get_largestCC(arr, dbg=False):
    if dbg:
        dbg_CC(arr,prec=0.02)

    print('Applying Connected Component')
    labels = label(arr)
    print('Found ', labels.max(), 'labels')
    max_label = 0
    max_num_bins = 0
    # Find largestCC
    for c_label in range(1, labels.max()+1):
        curr_num_bins = np.sum(np.where(labels == c_label, 1, 0))
        if curr_num_bins > max_num_bins:
            max_num_bins = curr_num_bins
            max_label = c_label
    print('Max CC label is: ', max_label)
    # largestCC = labels == np.argmax(np.bincount(labels.flat))

    print('Num liver before CC: ', np.sum(arr))
    arr = np.where(labels == max_label, 1, 0)
    print('Num liver After CC: ', np.sum(arr))
    if dbg:
        dbg_CC(arr,prec=0.02)
    return arr

def get_CC_largerThanTh(arr, thresh=8000,dbg=False):
    if dbg:
        dbg_CC(arr, prec=0.02)

    print('Applying Connected Component and take components with num pixels > max_pixels')
    labels = label(arr)
    print('Found ', labels.max(), 'labels')
    max_label = 0
    # Find largestCC
    large_labels = []
    for c_label in range(1, labels.max()+1):
        curr_num_bins = np.sum(np.where(labels == c_label, 1, 0))
        print(c_label, ':', curr_num_bins)
        if curr_num_bins > thresh:
            large_labels.append(c_label)
    print('Max CC label is: ', max_label)

    print('Num liver before CC: ', np.sum(arr))
    is_first = True
    for c_label in large_labels:
        if is_first:
            arr = np.where(labels == c_label, 1, 0)
            is_first = False
        else:
            arr[labels == c_label] = 1
    print('Num liver After CC: ',np.sum(arr) )

    if dbg:
        dbg_CC(arr,prec=0.02)
    return arr

def dbg_CC(arr, prec=0.01):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pos = np.where(arr == 1)
    num_points = int(np.round(prec * len(pos[0])))
    indices = np.random.permutation(len(pos[0]))[0:num_points]

    ax.scatter(pos[0][indices], pos[1][indices], pos[2][indices])
    ax.view_init(elev=230., azim=360)
    plt.show()
    plt.ioff()
    # plt.waitforbuttonpress()
    plt.close()


# ############################################################
#  Logs Tools
############################################################

def write_info_log(logdir, line=""):
    config = Config.getInstance()
    # save results text file
    os.makedirs(logdir)
    text_file = open(logdir + "info.txt", "a")

    text_file.write(line+"\n")

    text_file.write("type = "+str(config.backbone)+"\n")
    text_file.write("img_width = "+str(config.img_width)+"\n")
    text_file.write("img_height = "+str(config.img_height)+"\n")
    text_file.write("img_z = "+str(config.img_z)+"\n")
    text_file.write("num_classes_liver = "+str(config.num_classes_liver)+"\n")
    text_file.write("num_classes_lesion = "+str(config.num_classes_lesion)+"\n")
    text_file.write("num_classes_classification = "+str(config.num_classes_classification)+"\n")
    text_file.write("mean = "+str(config.mean)+"\n")
    text_file.write("std = "+str(config.std)+"\n")
    text_file.write("init_lr = "+str(config.init_lr)+"\n")
    text_file.write("batch_size = "+str(config.batch_size)+"\n")
    text_file.write("epochs = "+str(config.epochs)+"\n")
    text_file.write("lesion_weights = "+str(config.class_weights)+"\n")
    text_file.write("lesion_weights = "+str(config.lesion_weights)+"\n")
    text_file.write("freezing idx Threshold = "+str(config.freeze_idx_th)+"\n")
    text_file.write("\n")
    text_file.close()


# ############################################################
#  DBG Tools
############################################################

def dbg_orig_img(data_path, filename, color=(0, 255, 0), addText=True):
    # Original Image
    img = np.load(os.path.join(data_path, filename))
    img = DataClass().clip_vals(img)
    img = array_to_img(img, clip=True).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if addText:
        cv2.putText(img, 'Original Image - ' + filename[:-4], (120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)
    cv2.imshow('image', img)
    return img


def dbg_lesion_GT(img, mask, lesion_idx, lesion_type, classes_colors, alpha=0.5, color=(0, 0, 255), addText=True):
    """
    :param img:
    :param mask:
    :param lesion_idx:
    :return: Liver GT + Lesions GT (I = alpha F + (1 - alpha) B)
    """
    img = apply_mask(img, mask, color, mask_idx=lesion_idx, alpha=alpha)
    if addText:
        cv2.putText(img, '+ GT', (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color)
        cv2.putText(img, 'Lesion type :' + lesion_type, (120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, classes_colors[lesion_idx])
    cv2.imshow('image', img)

def dbg_liver_bb(img, dims, color=(0, 255, 0), addText=True):
    (w1, h1, w2, h2) = dims
    # Liver Patch on Image
    if addText:
        cv2.putText(img, '+ Liver Crop', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color)
    cv2.rectangle(img, (w1, h1), (w2, h2), color, 2)
    cv2.imshow('image', img)

def dbg_image(img_path):
    img = np.load(img_path)
    img = scale_to_img(DataClass().clip_vals(img))
    plt.imshow(img.astype('uint8'))