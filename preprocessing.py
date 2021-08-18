import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy as ent
from utils import entropy, cross_entropy, generate_x
import argparse

parser = argparse.ArgumentParser(description='Image preprocessing for CEIQ training')
parser.add_argument('-i', '--input_csv', default='contrast_distorted_dmos.csv', type=str,
                    help='csv that stores dmos of training images')
parser.add_argument('-f', '--folder_path', default='dataset/contrast_distorted', type=str, 
                    help='relative path to folder contains training images')

args = parser.parse_args()

df = pd.read_csv(args.input_csv)
Xs = []
ys = []
for i in range(df.shape[0]):
    img_path = os.path.join(args.folder_path, df.iloc[i]['img_name'])
    Xs.append(generate_x(img_path))
    ys.append(df.iloc[i]['dmos'])
Xs = np.array(Xs)
ys = np.array(ys)
print('Training data shape:')
print(Xs.shape)
print(ys.shape)

print('Saving training sets into pickle files...')
with open('Xs.pickle', 'wb') as f:
    pickle.dump(Xs, f)

with open('ys.pickle', 'wb') as f:
    pickle.dump(ys, f)
print('Saved successfully')