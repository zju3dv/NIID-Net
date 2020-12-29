# Codes are adapted from: https://github.com/zhengqili/CGIntrinsics

import numpy as np
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.measurements import label
from skimage.transform import resize

import test.saw_utils as saw_utils


def eval_on_images(shading_image_arr, pixel_labels_dir, 
    thres_list, photo_id, bl_filter_size, img_dir, mode):
    """
    This method generates a list of precision-recall pairs and confusion
    matrices for each threshold provided in ``thres_list`` for a specific
    photo.

    :param shading_image_arr: predicted shading images

    :param pixel_labels_dir: Directory which contains the SAW pixel labels for each photo.

    :param thres_list: List of shading gradient magnitude thresholds we use to
    generate points on the precision-recall curve.

    :param photo_id: ID of the photo we want to evaluate on.

    :param bl_filter_size: The size of the maximum filter used on the shading
    gradient magnitude image. We used 10 in the paper. If 0, we do not filter.
    """

    shading_image_linear_grayscale = shading_image_arr
    shading_image_linear_grayscale[shading_image_linear_grayscale < 1e-4] = 1e-4
    shading_image_linear_grayscale = np.log(shading_image_linear_grayscale)

    shading_gradmag = saw_utils.compute_gradmag(shading_image_linear_grayscale)
    shading_gradmag = np.abs(shading_gradmag)

    if bl_filter_size:
        shading_gradmag_max = maximum_filter(shading_gradmag, size=bl_filter_size)

    # We have the following ground truth labels:
    # (0) normal/depth discontinuity non-smooth shading (NS-ND)
    # (1) shadow boundary non-smooth shading (NS-SB)
    # (2) smooth shading (S)
    # (100) no data, ignored
    y_true = saw_utils.load_pixel_labels(pixel_labels_dir=pixel_labels_dir, photo_id=photo_id)
    
    img_path = img_dir+ str(photo_id) + ".png"

    # diffuclut and harder dataset
    srgb_img = saw_utils.load_img_arr(img_path)
    srgb_img = np.mean(srgb_img, axis = 2)
    img_gradmag = saw_utils.compute_gradmag(srgb_img)

    smooth_mask = (y_true == 2)
    average_gradient = np.zeros_like(img_gradmag)
    # find every connected component
    labeled_array, num_features = label(smooth_mask)
    for j in range(1, num_features+1):
        # for each connected component, compute the average image graident for the region
        avg = np.mean(img_gradmag[labeled_array == j])
        average_gradient[labeled_array == j]  = avg

    average_gradient = np.ravel(average_gradient)

    y_true = np.ravel(y_true)
    ignored_mask = y_true > 99

    # If we don't have labels for this photo (so everything is ignored), return
    # None
    if np.all(ignored_mask):
        return [None] * len(thres_list)

    ret = []
    for thres in thres_list:
        y_pred = (shading_gradmag < thres).astype(int)
        y_pred_max = (shading_gradmag_max < thres).astype(int)
        y_pred = np.ravel(y_pred)
        y_pred_max = np.ravel(y_pred_max)
        # Note: y_pred should have the same image resolution as y_true
        assert y_pred.shape == y_true.shape

        if mode < 0.1:
            confusion_matrix = saw_utils.grouped_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask], y_pred_max[~ignored_mask])
        else:
            confusion_matrix = saw_utils.grouped_weighted_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask], y_pred_max[~ignored_mask], average_gradient[~ignored_mask])
        
        ret.append(confusion_matrix)

    return ret


def get_precision_recall_list_new(predict_IID, pixel_labels_dir, thres_list, photo_ids,
                              class_weights, bl_filter_size, img_dir,
                              mode, display_process=True, samples=0, use_subset=False):

    output_count = len(thres_list)
    overall_conf_mx_list = [
        np.zeros((3, 2), dtype=int)
        for _ in range(output_count)
    ]

    if use_subset:
        photo_ids = photo_ids[::15]

    count = 0 
    total_num_img = len(photo_ids)

    sample_arr = []
    per_sample = max(len(photo_ids) // samples, 1)
    for idx, photo_id in enumerate(photo_ids):
        if display_process:
            print("photo_id ", count, photo_id, total_num_img)
        # load photo using photo id, hdf5 format
        img_path = img_dir + str(photo_id) + ".png"

        saw_img = saw_utils.load_img_arr(img_path)
        original_h, original_w = saw_img.shape[0], saw_img.shape[1]
        saw_img = saw_utils.resize_img_arr(saw_img)

        pred_N_np, pred_R_np, pred_L_np, pred_S_np, rendered_img_np = predict_IID(saw_img)
        if idx % per_sample == 0:
            sample_arr.append({
                'pred_N': pred_N_np,
                'pred_R': pred_R_np,
                'pred_L': pred_L_np,
                'pred_S': pred_S_np,
                'input_srgb': saw_img,
                'rendered_img': rendered_img_np})
        pred_S_np = np.mean(pred_S_np, axis=2)
        pred_S_np = resize(pred_S_np, (original_h, original_w), order=1, preserve_range=True,
                           mode='constant', anti_aliasing=False)

        # Save hdf5 file
        # pred_R_np = resize(pred_R_np, (original_h, original_w), order=1, preserve_range=True)
        # h5_dir = "checkpoints/test_saw/hdf5"
        # if not path.exists(h5_dir):
        #     makedirs(h5_dir)
        # hdf5_path = path.join(h5_dir, str(photo_id) + ".h5")
        # hdf5_write = h5py.File(hdf5_path, "w")
        # g = hdf5_write.create_group("prediction/")
        # g.create_dataset('R', data=pred_R_np)
        # g.create_dataset('S', data=pred_S_np)
        # hdf5_write.close()

        # pred_path = pred_dir + str(photo_id) + ".h5"
        #
        # hdf5_file_read = h5py.File(pred_path,'r')
        # pred_R = hdf5_file_read.get('/prediction/R')
        # pred_R = np.array(pred_R)
        # pred_S = hdf5_file_read.get('/prediction/S')
        # pred_S = np.array(pred_S)
        # hdf5_file_read.close()

        # compute confusion matrix
        conf_mx_list = eval_on_images(shading_image_arr=pred_S_np,
            pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
            photo_id=photo_id, bl_filter_size=bl_filter_size, img_dir=img_dir,
            mode=mode
        )

        for i, conf_mx in enumerate(conf_mx_list):
            # If this image didn't have any labels
            if conf_mx is None:
                continue
            overall_conf_mx_list[i] += conf_mx

        count += 1

        ret = []
        for i in range(output_count):
            overall_prec, overall_recall = saw_utils.get_pr_from_conf_mx(
                conf_mx=overall_conf_mx_list[i], class_weights=class_weights,
            )

            ret.append(dict(
                overall_prec=overall_prec,
                overall_recall=overall_recall,
                overall_conf_mx=overall_conf_mx_list[i],
            ))

    return ret, sample_arr


def compute_pr(predict_IID, pixel_labels_dir, splits_dir, dataset_split, class_weights, bl_filter_size, img_dir,
               mode, thres_count=400, display_process=True, samples=0, use_subset=False):
    """ Compute precision-recall

    :param predict_IID:
        the function to do IID
    :param pixel_labels_dir:
    :param splits_dir:
    :param dataset_split:
        R : train
        V : val
        E : test
    :param class_weights:
    :param bl_filter_size:
    :param img_dir:
    :param mode:
        0 : unweighted precision (P(u))
        1 : challenge precision (P(c))
    :param thres_count:
        number of shading smoothness thresholds
    :param display_process:
        print the evaluation process or not
    :param samples:
        number of samples that will be visualized and saved
    :param use_subset:
        only evaluate on a subset of SAW test set or not
    :return:
    """

    thres_list = saw_utils.gen_pr_thres_list(thres_count)
    photo_ids = saw_utils.load_photo_ids_for_split(
        splits_dir=splits_dir, dataset_split=dataset_split)

    plot_arrs = []
    line_names = []

    fn = 'pr-%s' % {'R': 'train', 'V': 'val', 'E': 'test'}[dataset_split]
    title = '%s Precision-Recall' % (
        {'R': 'Training', 'V': 'Validation', 'E': 'Test'}[dataset_split],
    )

    print("FN ", fn)
    print("title ", title)

    # compute PR 
    rdic_list, sample_arr = get_precision_recall_list_new(predict_IID=predict_IID, pixel_labels_dir=pixel_labels_dir,
                                                          thres_list=thres_list, photo_ids=photo_ids,
                                                          class_weights=class_weights, bl_filter_size=bl_filter_size,
                                                          img_dir=img_dir, mode=mode, display_process=display_process,
                                                          samples=samples, use_subset=use_subset)

    plot_arr = np.empty((len(rdic_list) + 2, 2))

    # extrapolate starting point 
    plot_arr[0, 0] = 0.0
    plot_arr[0, 1] = rdic_list[0]['overall_prec']

    for i, rdic in enumerate(rdic_list):
        plot_arr[i+1, 0] = rdic['overall_recall']
        plot_arr[i+1, 1] = rdic['overall_prec']

    # extrapolate end point
    plot_arr[-1, 0] = 1
    plot_arr[-1, 1] = 0.5

    AP = np.trapz(plot_arr[:,1], plot_arr[:,0])


    return AP, plot_arr[1:-1, :], sample_arr
