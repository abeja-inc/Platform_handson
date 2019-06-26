import numpy as np
import os


def get_bccd():
    base_path = './BCCD/'
    return base_path


def image_wise_to_instance_wise(label_img, inst_img):
    mask = []
    label = []
    inst_ids = np.unique(inst_img)
    for inst_id in inst_ids[inst_ids != -1]:
        msk = inst_img == inst_id
        lbl = np.unique(label_img[msk])[0] - 1

        assert inst_id != -1
        assert lbl != -1

        mask.append(msk)
        label.append(lbl)
    mask = np.array(mask).astype(np.bool)
    label = np.array(label).astype(np.int32)
    return mask, label


bccd_bbox_label_names = (
    'rbc',
    'wbc',
    'platelets')

bccd_semantic_segmentation_label_names = (('background',) +
                                         bccd_bbox_label_names)

bccd_instance_segmentation_label_names = bccd_bbox_label_names

# these colors are used in the original MATLAB tools
bccd_semantic_segmentation_label_colors = (
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
)
bccd_semantic_segmentation_ignore_label_color = (224, 224, 192)
