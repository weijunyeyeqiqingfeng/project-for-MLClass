import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(image, Image.Image):
        image.save(path)
    else:
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        Image.fromarray(image).save(path)
    print(f'✓ Saved: {path}')

def clip_similarity(clip_model, processor, image, text):
    inputs = processor(
        text=[text],
        images=image,
        return_tensors='pt',
        padding=True
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
        return outputs.logits_per_image.item()

def edge_density(image):
    """
    Measure structural complexity of a sketch using edge density.
    Higher value indicates denser stroke structures.
    """
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges.mean()

def stroke_countestimation(image):
    """估计笔画数量（基于连通组件分析）"""
    img = np.array(image.convert('L'))
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels = cv2.connectedComponents(binary)
    return num_labels - 1  # 减去背景

def structural_complexity(image):
    """基于傅里叶变换的结构复杂度"""
    img = np.array(image.convert('L'))
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return np.std(magnitude_spectrum)