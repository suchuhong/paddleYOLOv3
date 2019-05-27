from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2


class DataSetReader(object):
    """A class for parsing and read COCO dataset"""

    def __init__(self):
        self.has_parsed_categpry = False

    def _parse_dataset_catagory(self):
        self.categories = [{"id": 1, "name": "pineappletop"},
                           {"id": 2, "name": "pineapple"},
                           {"id": 3, "name": "pineapplebottom"}]
        self.num_category = len(self.categories)
        self.label_names = []
        self.label_ids = []
        for category in self.categories:
            self.label_names.append(category['name'])
            self.label_ids.append(int(category['id']))
        self.category_to_id_map = {
            v: i
            for i, v in enumerate(self.label_ids)
        }
        print("Load in {} categories.".format(self.num_category))
        self.has_parsed_categpry = True

    def get_label_infos(self):
        if not self.has_parsed_categpry:
            self._parse_dataset_catagory()
        return (self.label_names, self.label_ids)

    def get_reader(self,size=416,image=None):

        def img_reader(img, size, mean, std):
            im_path = img['image']
            im = cv2.imread(im_path).astype('float32')
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            h, w, _ = im.shape
            im_scale_x = size / float(w)
            im_scale_y = size / float(h)
            out_img = cv2.resize(im, None, None,
                                 fx=im_scale_x, fy=im_scale_y,
                                 interpolation=cv2.INTER_CUBIC)
            mean = np.array(mean).reshape((1, 1, -1))
            std = np.array(std).reshape((1, 1, -1))
            out_img = (out_img / 255.0 - mean) / std
            out_img = out_img.transpose((2, 0, 1))

            return (out_img, int(img['id']), (h, w))

        def reader():
            pixel_means = [0.485, 0.456, 0.406]
            pixel_stds = [0.229, 0.224, 0.225]
            img = {}
            img['image'] = image
            img['id'] = 0
            im, im_id, im_shape = img_reader(img, size,
                                             pixel_means,
                                             pixel_stds)
            batch_out = [(im, im_id, im_shape)]
            yield batch_out

        return reader


dsr = DataSetReader()


def infer(size=416, image=None):
    return dsr.get_reader(size, image=image)


def get_label_infos():
    return dsr.get_label_infos()