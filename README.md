# Improving Video Anomaly Detection Performance via Unseen Testing Data

This repo is the official open source of Improving Video Anomaly Detection Performance via Unseen Testing Data by Renzhi Wu, Shuai Li, Chenglizhao Chen, Aimin Hao.

It is implemented in tensorflow. Please follow the instructions to run the code.

## 1. Installation (Anaconda with python3.6 installation is recommended)
* Install 3rd-package dependencies of python (listed in requirements.txt)
```
numpy==1.14.1
scipy==1.0.0
matplotlib==2.1.2
tensorflow-gpu==1.4.1
tensorflow==1.4.1
Pillow==5.0.0
pypng==0.0.18
scikit_learn==0.19.1
opencv-python==3.2.0.6
```

```shell
pip install -r requirements.txt

pip install tensorflow-gpu==1.4.1
```
* Other libraries
```code
CUDA 8.0
Cudnn 6.0
Ubuntu 14.04 or 16.04, Centos 7 and other distributions.
```

## 2. Download datasets

Please manually download all datasets from [ped1.tar.gz, ped2.tar.gz, avenue.tar.gz and shanghaitech.tar.gz](https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F)
and tar each tar.gz file, and move them in to **Data** folder.

You can also download data from BaiduYun(https://pan.baidu.com/s/1fX8Lk62kw7QBfuhEDdlpNg) hn9a

## 3. Training from scratch (here we use ped2 datasets for examples)

Run `train_and_test.sh`.


## 4. Testing on saved models

Please manually download pretrained models from 
Baiduyun(https://pan.baidu.com/s/1XyJzy1SnLiToy680IduG0A) 8z49 and move them into **data/xxx/model** folder. Then, run `test.sh` (Before doing this, make sure you have extracted frame data, optical flow data, and abstract features.)

## 5. Retrain the models

Run `gen_retrain_data.sh` and move the `xxx/newdata/Test` to `xxx/Test1`, then create or modify the `test1.lst` and `all1.lst` files. Run `retrain.sh`.