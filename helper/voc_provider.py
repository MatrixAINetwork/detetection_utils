import os

from utils.rand_sampler import RandCropper, RandPadder
from dataprovider.det import DetIter
from dataset.pascal_voc import PascalVoc

train_cfg = {
    "root_dir": os.path.join(os.path.dirname(__file__), '..'),
    "random_samplers": 
    [
        RandCropper(min_scale=1., max_trials=1, max_sample=1),
        RandCropper(min_scale=.3, min_aspect_ratio=.5, max_aspect_ratio=2., min_overlap=.1),
        RandCropper(min_scale=.3, min_aspect_ratio=.5, max_aspect_ratio=2., min_overlap=.3),
        RandCropper(min_scale=.3, min_aspect_ratio=.5, max_aspect_ratio=2., min_overlap=.5),
        RandCropper(min_scale=.3, min_aspect_ratio=.5, max_aspect_ratio=2., min_overlap=.7),
        RandPadder(max_scale=2., min_aspect_ratio=.5, max_aspect_ratio=2., min_gt_scale=.05),
        RandPadder(max_scale=3., min_aspect_ratio=.5, max_aspect_ratio=2., min_gt_scale=.05),
        RandPadder(max_scale=4., min_aspect_ratio=.5, max_aspect_ratio=2., min_gt_scale=.05)
    ],
    "random_flip": True,
    "shuffle": True,
    "random_seed": None
}

# # validation
# cfg.VALID = edict()
# cfg.VALID.RAND_SAMPLERS = []
# cfg.VALID.RAND_MIRROR = False
# cfg.VALID.INIT_SHUFFLE = False
# cfg.VALID.EPOCH_SHUFFLE = False
# cfg.VALID.RAND_SEED = None

ssd_prov = DetIter(
        imdb=PascalVoc("trainval", "2007", "/unsullied/sharefs/yugang/Dataset/VOC", is_train=True),
        batch_size=32, 
        data_shape=(300, 300), 
        mean_pixels=[104, 117, 123], 
        rand_samplers=train_cfg['random_samplers'],
        rand_flip=train_cfg['random_flip'],
        shuffle=train_cfg['shuffle'],
        rand_seed=train_cfg['random_seed']
    )