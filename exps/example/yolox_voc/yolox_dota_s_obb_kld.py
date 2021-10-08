# encoding: utf-8
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import ExpOBB_KLD as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.input_size = (1024, 1024) #add
        self.random_size = (28, 36)
        self.num_classes = 16
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # --------------- transform config ----------------- #
        self.degrees = 0.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mscale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = False

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 50
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.0025 / 16.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.save_interval = 1
        self.print_interval = 5
        self.eval_interval = 50

        # -----------------  testing config ------------------ #
        self.test_size = (1024, 1024)
        self.test_conf = 0.01
        self.nmsthre = 0.3 #default 0.65

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):  #  no_aug=False
        from yolox.data import (
            DOTAOBBDetection,
            TrainTransformOBB,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetectionOBB,
        )

        dataset = DOTAOBBDetection(
            #data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"), #delete
            data_dir = '/home/lyy/gxw/DOTA_OBB_1_5',
            #image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
            image_sets=[('2012', 'train')],
            img_size=self.input_size,
            preproc=TrainTransformOBB(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=150,
            ),
        )

        dataset = MosaicDetectionOBB(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransformOBB(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=400,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import DOTAOBBDetection, ValTransformOBB

        valdataset = DOTAOBBDetection(
            #data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            data_dir='/home/lyy/gxw/DOTA_OBB_1_5', #add
            image_sets=[('2012', 'val')],
            img_size=self.test_size,
            preproc=ValTransformOBB(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import DOTAEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = DOTAEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator
