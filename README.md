# Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement

This repository contains an implementation of Zero DCE in tensorflow

You can find more details here: https://li-chongyi.github.io/Proj_Zero-DCE.html.

The implementation of Zero-DCE is for non-commercial use only. 

# tensorflow 1.x
The code base uses tensorlow 1.x

## Requirements
1. Python 3.6 or above
2. tensorflow 1.6 or above
3. opencv
4. matplotlib


### Test: 
1. To test from tflite file use python3 infer_tflite.py <low_lght_image>

### Train: 
pyhton train.py <br>
N.B. executing the above command will train the model with default hyperparameters


## Bibtex

```
@inproceedings{Zero-DCE,
 author = {Guo, Chunle Guo and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
 title = {Zero-reference deep curve estimation for low-light image enhancement},
 booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)},
 pages    = {1780-1789},
 month = {June},
 year = {2020}
}
```
# References
[1] Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement - CVPR 2020 [link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)

[2] Low-light dataset - [link](https://drive.google.com/file/d/1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN/view)

# Maintainer
[1] This repo is created and maintained by Saket Mohanty
