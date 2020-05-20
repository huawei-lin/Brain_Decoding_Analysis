# 触觉神经脑功能拓扑分析平台
## 声明
v2020.05 本版本用于 2020 年计算机设计大赛。

## 功能
本平台将输入的 zip 压缩包(内含 31 项 dicom 文件，切片大小为 128\*128)对后进行解码，获得解码结果并输出解码过程的可视化图像。

## 需求分析
作为从可测量的大脑活动中读取大脑状态的技术，脑解码技术广泛应用于工业和医学领域，尤其是医学领域。神经疾病是我国乃至全球人口健康领域正面临的重大挑战，全世界大约五分之一的成年人在某一年经历过神经疾病。严重的疾病会导致大脑在处理认知状态时出现问题，比如长时间集中注意力，在难以分辨的两件事之间做出区分，以及对迅速出现的信息做出快速反应。
虽然脑解码技术目前前景可观，但是要实际应用仍面临一定障碍，主要有以下难点：
1.由于T-fMRI数据的高维度与其时空复杂性，脑解码技术无法做到准确快速的响应
2.由于血氧水平需要复杂的计算，脑解码技术需要多次重复地进行重新校准
3.大脑状态信息变化对肉眼不明显，难以观测到重要信息
4.阅读 fMRI 影像需要经年累月的长期培训和经验总结
在本作品中，我们使用了经过训练的3D卷积神经网络（CNN）的解码器，将大脑活动分为四个触觉任务类别。对于训练和测试，我们同时使用了HCP(Human Connectome Project )项目中的大型T-fMRI数据集，旨在实现更高的解码精度。
目前，该作品使用基于深度学习的计算机视觉技术与传统医疗影像相结合的功能磁共振成像的普适性辅助诊断方法，将输入受试者在接受刺激时大脑活动的医疗影像进行大脑激活热区分区以及多任务刺激类型分类。简而言之，我们提供了一个新的有希望的平台将人脑作为功能相互作用的大脑区域的集成网络进行检查，从而对外源性或内源性大脑进行高精度高分辨率的解码。

## 业务流程
步骤1、归档有 31 项 128\*128 的大脑切片图像的 zip 压缩文件被拖拽到 index.html 的拖拽框中。
步骤2、index.html 调用 './html/upload.php' 将 zip 压缩文件保存至 './drag_uploads/'，并命名为 'data.zip'，即路径为 './drag_uploads/data.zip'。
步骤3、保存 zip 文件后，上述 php 文件调用 './drag_uploads/fMRI/Network.py' 文件。
步骤4、上述 python 文件将压缩包解压至 './drag_uploads/fMRI/data/' 并使用 './drag_uploads/fMRI/model/' 中的 tensorflow 网络模型进行解码并获得可视化图像，最终将解码结果保存为 './drag_uploads/fMRI/result.re'，解码可视化图像保存在 './html/fmri_images/' 并命名为 '{\0-30}.jpg'，另有一张 '31.jpg' 为纯色图像。
步骤5、等待上述步骤 4 运行结束后，php 文件将读取 './drag_uploads/fMRI/result.re' 的解码结果，跳转至相应的 html 页面，并加载相应的解码可视化图像。

## 主要目录
项目存放位置: /var/www/ (在本项目中写作 './')
web主页： ./html/index.html 
压缩包存放位置： ./drag_uploads/
数据存放位置： ./drag_uploads/fMRI/data/
输出图片存放位置： ./html/fmri_images/
模型存放位置：./drag_uploads/fMRI/model/
解码结果：./drag_uploads/fMRI/result.re


## 运行环境
Ubuntu 16.04
Apache 2
python 3.7
php 7.0
tensorflow 1.13.1
cuda 10.0


## 目录
```python
./
├── drag_uploads
│   ├── data.zip 拖拽上传的 zip 文件
│   ├── fMRI     
│   │   ├── data  拖拽上传的 zip 文件解压目录
│   │   ├── logs  保存运行日志
│   │   ├── model 网络框架模型
│   │   ├── Network.py 
│   │   └── result.re 解码结果
│   └── logs 保存运行日志
├── html
│   ├── css 
│   ├── fmri_images 可视化图像
│   ├── images
│   ├── js
│   ├── logs
│   ├── nuget
│   ├── sass
│   ├── themes
│   ├── upload.php
│   ├── Social.html
│   ├── index.html
│   ├── static-state.html
│   ├── Relation.html
│   ├── Language.html
│   ├── Gambling.html
│   ├── hu-kou.html
│   ├── lips.html
│   ├── Motor.html
│   ├── emotion.html
│   ├── Anticapate.html
│   ├── figer.html
│   └── WM.html
├── model_creator 网络框架模型训练代码
└── README.md
```

## 实例
测试数据: ./data/
将压缩包拖拽进主页的文件框中。

## 注意事项
1. 注意文件夹权限问题。
2. 注意系统环境版本问题。

## ABOUT ME
陕西科技大学   电子信息与人工智能学院
指导老师：齐勇 qiyong@sust.edu.cn
 
v2020.05
林华伟   winsoullin@gmail.com
李艳平   1761708364@qq.com


v2019.05
曹静    onetwothreeamon@gmail.com
林华伟   winsoullin@gmail.com
陈嘉树   1416228373@qq.com

## 修改时间
2020-05-21 00:49:24