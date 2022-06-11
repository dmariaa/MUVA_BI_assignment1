import glob
import os.path
import sys

import numpy as np
import cv2
import pandas as pd
import seaborn
from skimage import feature
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from tools import compute_multichannel_lbp_histogram, roc_pr_curves, DET_curve


def get_image_data(image_path):
    path, file = os.path.split(image_path)
    base_dir = path.split(os.path.sep)[-1]

    if base_dir == 'users':
        return_data = {
            'file_path': path,
            'file_name': file,
            'attack': -1,
        }
    else:
        attack_number = int(base_dir.split('_')[-1])
        return_data = {
            'file_path': path,
            'file_name': file,
            'attack': attack_number,
        }

    return return_data


def generate_dataframe(folder):
    images = glob.glob(os.path.join(images_path, "**/*.jpg"), recursive=True)
    radius = 3
    n_points = 8 * radius

    data = []

    for i, image_path in enumerate(images):
        print(f"Processing image {i} of {len(images)}", end='\r')
        image_data = get_image_data(image_path)
        image = cv2.imread(image_path)
        lbp_histogram = compute_multichannel_lbp_histogram(image, n_points, radius)
        image_data['features'] = lbp_histogram.flatten()
        data.append(image_data)

    df = pd.DataFrame(data)
    return df


def split_dataset(df):
    X = np.vstack(df['features'].values)
    Y = (df['attack'].values != -1).astype(float)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=.25, random_state=12345)
    split_ix = splitter.split(X, Y)

    for train_ix, test_ix in split_ix:
        X_train = X[train_ix]
        Y_train = Y[train_ix]
        X_test = X[test_ix]
        Y_test = Y[test_ix]

    return X_train, Y_train, X_test, Y_test


def fit_clf(dataset):
    X_train, Y_train, X_test, Y_test = split_dataset(dataset)
    classifier = SVC(kernel='sigmoid', C=100.0, gamma='auto', random_state=12345, probability=True)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train[:, :26])
    classifier.fit(X_train, Y_train)

    X_test = scaler.transform(X_test[:, :26])
    prediction = classifier.predict(X_test)
    prediction_proba = classifier.predict_proba(X_test)[:, 1]

    return X_test, Y_test, prediction, prediction_proba


if __name__ == "__main__":
    import argparse

    plt.rcParams.update({'font.size': 7})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    def options():
        parser = argparse.ArgumentParser()
        parser.add_argument('-g', '--generate', help='Regenerate dataset', action='store_true')
        parser.add_argument('-o', '--output', help='Output folder for images and pkl files',
                            default='out/flir_analysis')
        return parser

    args = options().parse_args()
    generate = args.generate
    output_folder = args.output

    if generate:
        images_path = "data/flir"
        df = generate_dataframe(images_path)
        df.to_pickle(os.path.join(output_folder, 'dataset.pkl'))
    else:
        df = pd.read_pickle(os.path.join(output_folder, 'dataset.pkl'))

    X_test, Y_test, prediction, prediction_proba = fit_clf(df)

    roc_pr_curves(Y_test, prediction_proba, save_name=os.path.join(output_folder, "flir_roc"))
    APCER_1, BPCER_1, ths_1 = DET_curve(Y_test, prediction_proba, save_name=os.path.join(output_folder, "flir_det"))

    cm = metrics.confusion_matrix(Y_test, prediction)
    accuracy = metrics.accuracy_score(Y_test, prediction)
    tn, fp, fn, tp = cm.flatten()

    cm_df = pd.DataFrame(cm, index=['live', 'fake'], columns=['live', 'fake'])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    seaborn.heatmap(cm_df, annot=True, ax=ax)
    plt.show()

    apcer = fn / (tp + fn)
    npcer = fp / (fp + tn)
    print(f"Accuracy: {accuracy}")
    print(f"APCER: {apcer}")
    print(f"NPCER: {npcer}")
    print(f"ACER: {(apcer + npcer) / 2}")
    print(f"FRR: {fn / (tp + fn)}")
    print(f"FAR: {fp / (fp + tn)}")
