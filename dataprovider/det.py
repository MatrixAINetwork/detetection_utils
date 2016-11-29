import numpy as np
import cv2
from utils.image_processing import resize, transform
from utils.rand_sampler import RandSampler

class DetIter(object):
    """
        DetIter()
    """
    def __init__(self, imdb, batch_size, data_shape, mean_pixels=[128, 128, 128], 
            rand_samplers=[], rand_flip=False, shuffle=False, rand_seed=None):

        self._imdb = imdb
        self.batch_size = batch_size
        if isinstance(data_shape, int):
            data_shape = (data_shape, data_shape)
        self._data_shape = data_shape
        self._mean_pixels = mean_pixels
        self.is_train = self._imdb.is_train

        # image shuffle
        if rand_seed:
            np.random.seed(rand_seed)
        self._shuffle = shuffle
        self._size = self._imdb.num_images
        self._current = 0
        self._fake_index = np.arange(self._size)
        if self._shuffle:
            np.random.shuffle(self._fake_index)

        # augmentation
        self._rand_flip = rand_flip
        if not rand_samplers:
            self._rand_samplers = []
        else:
            if not isinstance(rand_samplers, list):
                rand_samplers = [rand_samplers]
            assert isinstance(rand_samplers[0], RandSampler), "Invalid rand sampler"
            self._rand_samplers = rand_samplers

    def _get_batch(self):
        batch_data = []
        batch_label = []
        indices = []
        for i in range(self.batch_size):
            if (self._current + i) >= self._size:
                if not self.is_train:
                    continue
                # use padding from middle in each epoch
                idx = (self._current + i + self._size / 2) % self._size
                fake_index = self._fake_index[idx]
            else:
                fake_index = self._fake_index[self._current + i]

            im_path = self._imdb.image_path_from_index(fake_index)
            img_id = self._imdb.image_set_index[fake_index]
            img = cv2.imread(im_path)
            gt = self._imdb.label_from_index(fake_index).copy() if self.is_train else None
            data, label = self._data_augmentation(img, gt)
            batch_data.append(data)
            if self.is_train:
                batch_label.append(label)
            indices.append(img_id)

        # pad data if not fully occupied
        for i in range(self.batch_size - len(batch_data)):
            assert len(batch_data) > 0
            batch_data.append(batch_data[0] * 0)
            indices.append(-1)

        self._current += self.batch_size
        return {
            'data': np.array(batch_data), 
            'label': np.array(batch_label) if self.is_train else None,
            'id': indices,
        }

    def _data_augmentation(self, data, label):
        if self.is_train and self._rand_samplers:
            rand_crops = []
            for rs in self._rand_samplers:
                rand_crops += rs.sample(label)
            num_rand_crops = len(rand_crops)

            if num_rand_crops > 0:
                index = int(np.random.uniform(0, 1) * num_rand_crops)
                width = data.shape[1]
                height = data.shape[0]
                crop = rand_crops[index][0]
                xmin = int(crop[0] * width)
                ymin = int(crop[1] * height)
                xmax = int(crop[2] * width)
                ymax = int(crop[3] * height)
                if xmin >= 0 and ymin >= 0 and xmax <= width and ymax <= height:
                    data = data[ymin:ymax, xmin:xmax, :]
                else:
                    # padding mode
                    new_width = xmax - xmin
                    new_height = ymax - ymin
                    offset_x = 0 - xmin
                    offset_y = 0 - ymin
                    data_bak = data
                    data = np.full((new_height, new_width, 3), 128.)
                    data[offset_y:offset_y+height, offset_x:offset_x + width, :] = data_bak
                label = rand_crops[index][1]

        if self.is_train and self._rand_flip:
            if np.random.uniform(0, 1) > 0.5:
                data = cv2.flip(data, 1)
                valid_mask = np.where(label[:, 0] > -1)[0]
                tmp = 1.0 - label[valid_mask, 1]
                label[valid_mask, 1] = 1.0 - label[valid_mask, 3]
                label[valid_mask, 3] = tmp

        if self.is_train:
            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, \
                              cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        else:
            interp_methods = [cv2.INTER_LINEAR]
        interp_method = interp_methods[int(np.random.uniform(0, 1) * len(interp_methods))]
        data = resize(data, self._data_shape, interp_method)
        data = transform(data, self._mean_pixels)
        return data, label

    def meg_get_batch(self):
        batch = self._get_batch()
        return {'img': batch['data'], 'gt_boxes': batch['label'], 'imgID': batch['id']}