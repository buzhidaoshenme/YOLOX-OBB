#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .darknet import CSPDarknet, Darknet
from .losses import IOUloss
from .KLD_loss import compute_kld_loss, KLDloss
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX


from .yolo_head_obb_kld import YOLOXHeadOBB_KLD
from .yolox_obb_kld import YOLOXOBB_KLD



