import math
import torch
import random
import itertools
import pydicom
import pretrainedmodels

import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data.sampler import BatchSampler

# Get model from name
available_models = pretrainedmodels.model_names + ['efficientnet-b'+str(i) for i in range(8)]
def get_model(model_name, n_classes):
    if model_name in available_models:
        print('Loading pretrained', model_name)
        if 'efficientnet' in model_name:
            model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=n_classes)
        else:
            # Might not be 1000!
            model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
            for module in model.children():
                if isinstance(module, nn.Linear):
                    for param in module.parameters():
                        if len(param.shape) == 2:
                            num_features = param.shape[0]
                            param.data = param.data[:n_classes]
                        else:
                            param.data = param.data[:n_classes]
            dummy_input = torch.randn([2, 3, 200, 200])
            with torch.no_grad():
                assert model(dummy_input).shape[1] == n_classes
        return model 
    else:
        print('Model name', model_name, 'not in', available_models)
        raise RuntimeError

### Functions borrowed from recursion #5's discussion.
class TrainBatchSampler(BatchSampler):
    def __init__(self, dataframe, batch_size):
        self.batch_size = batch_size

        dataframe = dataframe.copy().reset_index(drop=True)
        index_groups = []
        for _, df in dataframe.groupby(['patient']):
            index_groups.append(df.index.values)
        self.group_sizes = [max(len(g) // self.batch_size, 1) for g in index_groups]

        self.index_groups = [
            self._take_every(self._cycle_with_shuffle(g), self.batch_size)
            for g in index_groups
        ]
        self.length = sum(self.group_sizes)

    def __len__(self):
        return self.length

    def __iter__(self):
        batches = []
        for size, group in zip(self.group_sizes, self.index_groups):
            for _ in range(size):
                batches.append(next(group))

        random.shuffle(batches)
        return iter(batches)

    def _cycle_with_shuffle(self, xs):
        while True:
            random.shuffle(xs)
            yield from xs

    def _take_every(self, it, n):
        while True:
            chunk = []
            for _ in range(n):
                chunk.append(next(it))
            yield chunk


class TestBatchSampler(BatchSampler):
    def __init__(self, dataframe, batch_size):
        self.batch_size = batch_size

        dataframe = dataframe.copy().reset_index(drop=True)
        index_groups = []
        for _, df in dataframe.groupby(['patient']):
            index_groups.append(df.index.values)

        self.batches = []
        for g in index_groups:
            self.batches.extend(self._split_every(g, self.batch_size))

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def _split_every(self, xs, n):
        ret = []
        chunk = []
        for x in xs:
            chunk.append(x)
            if len(chunk) == n:
                ret.append(chunk)
                chunk = []
        if chunk:
            ret.append(chunk)
        return ret

def toadaBN(net):
    for m in net.modules():
        if isinstance(m, _BatchNorm):
            m.track_running_stats = False
    return net

weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0]).cuda()
def criterion(y_pred,y_true):
    return torch.nn.functional.binary_cross_entropy_with_logits(y_pred,
                               y_true, pos_weight=weights)

# Preprocessing utilities
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def window_image(img, window_center, window_width):
    _, _, intercept, slope = get_windowing(img)
    img = img.pixel_array * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    img = (img - np.min(img)) / (np.max(img) - np.min(img)+1e-8)
    return img

def bsb_window(img):
    brain_img = window_image(img, 40, 80)
    subdural_img = window_image(img, 80, 200)
    bone_img = window_image(img, 600, 2000)
    
    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img