# YOLOX-OBB
YOLOX in DOTA with KLD loss. (Oriented Object Detection)（Rotated BBox）基于YOLOX的旋转目标检测
## Installation 
1. Install YOLOX-OBB(You can refer to the installation of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX))
```shell
cd YOLOX-OBB
pip3 install -r requirements.txt
pip3 install -v -e
```
2. Install pycocotools
```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
3. Install swig
```shell
sudo apt-get install swig
```
4. Create the c++ extension for python
```shell
cd DOTA_devkit_YOLO
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
cd -
```
5. Install apex
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
cd -
```
## Data Preparation
1. Split images and annotations(You can refer to [DOTA_devkit_YOLO](https://github.com/hukaixuan19970627/DOTA_devkit_YOLO))
```shell
 python DOTA_devkit_YOLO/ImgSplit_multi_process.py
 ```
 2. Transform annotations into voc-like format
 
 * `This is a object in voc-like format annotation:`
 <img src="assets/voc-like .png" width="500" >
 
 ```shell
 python custom tools/DOTA2VOC_obb.py
 ```
 3. Organize Directories(All annotations of train-images and val-images must be put into Annotations folder)
 ```
 |--your_data
     |--VOC2012
         |--Annotations
             |-- xxx.xml
                 ... 
         |--ImageSets
             |--Main
                 |--train.txt
                 |--val.txt
                 |--test.txt
         |--JPEGImages
         |--JPEGImages-val
         |-JPEGImages-test
```
## Train 
1. Modify configs

　change the data path with yours in [yolox_dota_s_obb_kld.py](https://github.com/buzhidaoshenme/YOLOX-OBB/blob/main/exps/example/yolox_voc/yolox_dota_s_obb_kld.py)
```
data_dir = 'your_data_path'
```
2. Train
```
CUDA_VISIBLE_DEVICES=0,1 python3 tools/train.py -f exps/example/yolox_voc/yolox_dota_s_obb_kld.py -d 2 -b 16 --fp16 -c weights/yolox_s.pth.tar
```
## Val
1. get results
```
CUDA_VISIBLE_DEVICES=0,1 python tools/eval.py -f exps/example/yolox_voc/yolox_dota_s_obb_kld.py -d 2 -b 16 -c YOLOX_outputs/yolox_dota_s_obb_kld/latest_ckpt.pth
```
　Results will be save to your_data/results/VOC2012/Main
 * `If test, you must comment line 151 'target = self.load_anno(index)' and uncomment line 152 'target = []' in dota_obb.py before run the above instruction. Because test-set has no annotations.`

2. Merge results(You can refer to [DOTA_devkit_YOLO](https://github.com/hukaixuan19970627/DOTA_devkit_YOLO))
```
python DOTA_devkit_YOLO/ResultMerge.py
```
3. Evaluation
```
python DOTA_devkit_YOLO/dota_v1.5_evaluation_task1.py(You can refer to [DOTA_devkit_YOLO](https://github.com/hukaixuan19970627/DOTA_devkit_YOLO))
```
 * `If test, you should upload your results to DOTA Evaluation Server.`

## Unfortunately 
This implementation only get 0.712 mAP@0.5 on DOTA v1.0.

## Reference
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

[DOTA_devkit_YOLO](https://github.com/hukaixuan19970627/DOTA_devkit_YOLO)

[YOLOv5_DOTA_OBB](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB)

[RotationDetection](https://github.com/yangxue0827/RotationDetection)
