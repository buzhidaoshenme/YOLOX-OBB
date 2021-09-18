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
 python ImgSplit_multi_process.py
 ```
 2. Transform annotations into voc-like format
 
 * `This is a object in voc-like format annotation:`
 <img src="assets/voc-like.png" width="1000" >
 
 ```shell
 python custom tools/DOTA2VOC_obb.py
 ```
 
