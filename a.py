import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import torchvision.transforms as transforms
import argparse
from collections import OrderedDict
from PIL import Image
import os
import yolo

# 定义图片预处理函数
def preprocess_image(image_path, resize, crop):
    img_mat = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
    ])
    img = transform(img_mat)
    im_tensor = torch.unsqueeze(img, 0).float()
    return im_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    
    # 获取yolov5网络文件
    net = yolo.get_model(opt)
    
    # 配置量化参数，这里假设我们有N张图片用于量化
    N = 10  # 举例，实际使用时应根据图片数量进行设置
    qconfig = {'iteration': N, 'use_avg': False, 'data_scale': 1.0, 'firstconv': False, 'per_channel': False}
    
    # 调用量化接口
    quantized_net = mlu_quantize.quantize_dynamic_mlu(net.float(), qconfig_spec=qconfig, dtype='int8', gen_quant=True)
    
    # 设置为推理模式
    quantized_net.eval()
    
    # 假设我们有一个包含所有图片路径的列表
    image_paths = ["./images/image1.jpg", "./images/image2.jpg", ...]  # 这里需要填入所有图片的路径
    
    # 循环读取图片执行量化模型的推理
    for image_path in image_paths:
        im_tensor = preprocess_image(image_path, resize=640, crop=640)
        quantized_net(im_tensor)  # 执行推理，这里不需要计算损失，因此不使用反向传播
    
    # 保存量化模型
    torch.save(quantized_net.state_dict(), './yolov5s_int8.pt')