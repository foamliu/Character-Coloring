# Character Coloring

[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg?style=flat-square)](https://github.com/OpenImageIO/oiio/blob/master/LICENSE)

This repository is to repeat Colorful Image Colorization with Color-Net.

## Dependencies
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

![image](https://github.com/foamliu/Character-Coloring/raw/master/images/dataset.png)

Follow the [instruction](http://sysu-hcp.net/lip/index.php) to download Character-Coloring dataset.

## Architecture

![image](https://github.com/foamliu/Character-Coloring/raw/master/images/color_net.png)


## Usage
### Data Pre-processing
Extract training images:
```bash
$ python pre-process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo

```bash
$ python demo.py
```

Input | Output | GT | 
|---|---|---|
|![image](https://github.com/foamliu/Character-Coloring/raw/master/images/0_image.png) | ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/0_out.png)| ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/0_gt.png)|
|![image](https://github.com/foamliu/Character-Coloring/raw/master/images/1_image.png) | ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/1_out.png)| ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/1_gt.png)|
|![image](https://github.com/foamliu/Character-Coloring/raw/master/images/2_image.png) | ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/2_out.png)| ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/2_gt.png)|
|![image](https://github.com/foamliu/Character-Coloring/raw/master/images/3_image.png) | ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/3_out.png)| ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/3_gt.png)|
|![image](https://github.com/foamliu/Character-Coloring/raw/master/images/4_image.png) | ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/4_out.png)| ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/4_gt.png)|
|![image](https://github.com/foamliu/Character-Coloring/raw/master/images/5_image.png) | ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/5_out.png)| ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/5_gt.png)|
|![image](https://github.com/foamliu/Character-Coloring/raw/master/images/6_image.png) | ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/6_out.png)| ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/6_gt.png)|
|![image](https://github.com/foamliu/Character-Coloring/raw/master/images/7_image.png) | ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/7_out.png)| ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/7_gt.png)|
|![image](https://github.com/foamliu/Character-Coloring/raw/master/images/8_image.png) | ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/8_out.png)| ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/8_gt.png)|
|![image](https://github.com/foamliu/Character-Coloring/raw/master/images/9_image.png) | ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/9_out.png)| ![image](https://github.com/foamliu/Character-Coloring/raw/master/images/9_gt.png)|
