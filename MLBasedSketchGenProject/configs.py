import torch
import os

# HuggingFace 镜像（加速下载）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 模型配置（CPU 友好）
MODEL_CONFIG = {
    'clip': 'openai/clip-vit-base-patch32',
    # 'diffusion': 'OFA-Sys/small-stable-diffusion-v0',
    'diffusion': 'CompVis/stable-diffusion-v1-4',
    # 'diffusion': 'hakurei/waifu-diffusion',
}

# 设备配置
DEVICE = 'cpu'
DTYPE = torch.float32

# 基础生成参数
IMAGE_SIZE = 512
GUIDANCE_SCALE = 7.5

# Stroke 复杂度控制（Fine-grained Controllability 核心）
STROKE_CONFIG = {
    1: {'steps': 10, 'desc': 'minimal sketch, 1-3 simple lines, basic outline'},
    2: {'steps': 20, 'desc': 'simple sketch, clean lines, basic details'},
    3: {'steps': 40, 'desc': 'detailed sketch, clear outlines, moderate details'},
    4: {'steps': 60, 'desc': 'complex sketch, many details, intricate shapes'},
    5: {'steps': 80, 'desc': 'highly detailed sketch, dense lines, complex structure'},
}

# 风格控制
STYLE_PROMPTS = {
    'sketch': 'line art, black and white sketch, outline drawing, sketch style, monochrome, high contrast',
    'minimal': 'minimalist line sketch, single line drawing, abstract outline, simple contour',
    'cartoon': 'cartoon style, simple shapes, flat colors, bold outlines',
}

# 线条风格
LINE_STYLE_CONFIG = {
    'thin': 'thin lines, delicate strokes, fine lines',
    'thick': 'bold lines, thick strokes, strong outlines',
    'hatch': 'hatching, cross-hatching, technical drawing style',
    'dotted': 'dotted lines, stippling, pointillism style',
    'sketchy': 'sketchy lines, rough outlines, hand-drawn feel',
}

# 负面提示
NEGATIVE_PROMPTS = {
    'sketch': 'photo, realistic, 3d, rendering, shading, texture, color, background, detailed, complex, painting',
    'minimal': 'detailed, complex, shading, texture, background, colors, realistic',
}

# 实验用提示词
TEST_PROMPTS = ['cat', 'tree', 'house', 'airplane', 'flower']