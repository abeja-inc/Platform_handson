import numpy as np
import os
import warnings
import xml.etree.ElementTree as ET
from PIL import Image

import bccd_utils


class VOCBboxDataset:

    def __init__(self, data_dir='auto', split='train', year='2012',
                 use_difficult=False, return_difficult=False):

        if data_dir == 'auto':
            data_dir = bccd_utils.get_bccd()

        if split not in ['train', 'trainval', 'val', 'test']:
            warnings.warn(
                'please pick split from \'train\', \'trainval\', \'val\', \'test\''
            )
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.use_difficult = use_difficult

        if not return_difficult:
            self.keys = ('img', 'bbox', 'label')

    def __len__(self):
        return len(self.ids)

    def get_image(self, i):
        id_ = self.ids[i]
        img_path = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        return img_path

    def get_annotations(self, i):
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(bccd_utils.bccd_bbox_label_names.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool)

        ret = []
        for b, lbl, d in zip(bbox, label, difficult):
            ymin, xmin, ymax, xmax = b.tolist()
            rect = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
            ret.append({
                'rect': rect, 
                'category_id': 0, 
                'label_id': int(lbl), 
                'label': bccd_utils.bccd_bbox_label_names[lbl], 
                'difficult': bool(d)}
            )
        return id_, ret
