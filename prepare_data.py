import os
import shutil

import cv2
import numpy as np
import pandas as pd

import tools

import matplotlib.pyplot as plt

radius = 3
n_points = 8 * radius

dataset_folder = 'out/flir_analysis'
df = pd.read_pickle(os.path.join(dataset_folder, 'dataset.pkl'))
selected = df.query("file_name.str.contains('user_001')").sort_values('attack')

out_folder = os.path.join(dataset_folder, "doc")

for row in selected.iterrows():
    file_name = row[1].file_name
    file_path = row[1].file_path
    attack = row[1].attack
    file = os.path.join(file_path, file_name)

    # copy image file
    name, ext = os.path.splitext(file_name)
    if attack == -1:
        out_file_name = name
    else:
        out_file_name = f"{name}-attack_{attack}"

    shutil.copy(file, os.path.join(out_folder, f"{out_file_name}{ext}"))

    image = cv2.imread(file)
    histogram, lbp = tools.compute_multichannel_lbp_histogram(image, n_points, radius)

    # lpb image
    lbp_data = np.moveaxis(lbp, (0, 1, 2), (2, 0, 1)).astype(int)
    lbp_data = (lbp_data - lbp_data.min()) / (lbp_data.max() - lbp_data.min())
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 6.4))
    fig.tight_layout()
    ax.imshow(lbp_data)
    fig.savefig(os.path.join(out_folder, f"{out_file_name}-lbp.jpg"))

    # lbp histograms
    hist = histogram.ravel()
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    fig.tight_layout()
    ax.bar(np.arange(hist.shape[0]), hist)
    fig.savefig(os.path.join(out_folder, f"{out_file_name}-histogram.jpg"))
    pass

