import os
import cv2
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt


def psnr(target, ref):
    target_data = target.astype(float)
    ref_data = ref.astype(float)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(255. / rmse)


def mse(target, ref):
    err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
    err /= float(target.shape[0] * target.shape[1])
    return err

def compare_images(target, ref):
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(ssim(target, ref, multichannel =True))
    return scores


target = cv2.imread('hr.png')
ref = cv2.imread('re.png')
scores = compare_images(target, ref)
print("VAE : ", scores)
"""

target = cv2.imread('EDSR.png')
ref = cv2.imread('HR.png')
scores = compare_images(target, ref)
print("EDSR : ", scores)


target = cv2.imread('SRGAN.png')
ref = cv2.imread('HR.png')
scores = compare_images(target, ref)
print("SRGAN : ", scores)


target = cv2.imread('ERCA.png')
ref = cv2.imread('HR.png')
scores = compare_images(target, ref)
print("ERCA : ", scores)


target = cv2.imread('ESRGAN.png')
ref = cv2.imread('HR.png')
scores = compare_images(target, ref)
print("ESRGAN : ", scores)


target = cv2.imread('SRFEAT.png')
ref = cv2.imread('HR.png')
scores = compare_images(target, ref)
print("SRFEAT : ", scores)

"""
