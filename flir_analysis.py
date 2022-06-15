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

from tools import compute_multichannel_lbp_histogram, roc_pr_curves, DET_curve, confusion_matrix


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
        lbp_histogram, _ = compute_multichannel_lbp_histogram(image, n_points, radius)
        image_data['features'] = lbp_histogram.flatten()
        data.append(image_data)

    df = pd.DataFrame(data)
    return df


def split_dataset(df):
    X = np.vstack(df['features'].values)
    Y = (df['attack'].values != -1).astype(float)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=.25, random_state=12345)
    split_ix = splitter.split(X, df['attack'].values)

    for train_ix, test_ix in split_ix:
        X_train = X[train_ix]
        Y_train = Y[train_ix]
        X_test = X[test_ix]
        Y_test = Y[test_ix]

    return X_train, Y_train, X_test, Y_test


def fit_clf(X_train, Y_train):
    classifier = SVC(kernel='sigmoid', C=100.0, gamma='auto', random_state=12345, probability=True)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    classifier.fit(X_train, Y_train)

    return classifier, scaler


def predict(X, classifier, scaler):
    X = scaler.transform(X)
    prediction = classifier.predict(X)
    prediction_proba = classifier.predict_proba(X)[:, 1]
    return prediction, prediction_proba


if __name__ == "__main__":
    import argparse

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

    X_train, Y_train, X_test, Y_test = split_dataset(df)
    classifier, scaler = fit_clf(X_train, Y_train)

    # Stats around train dataset
    prediction, prediction_proba = predict(X_train, classifier, scaler)
    roc_pr_curves(Y_train, prediction_proba, save_name=os.path.join(output_folder, "flir_roc_train"))
    APCER_train, BPCER_train, ths_train = DET_curve(Y_train, prediction_proba,
                                                    save_name=os.path.join(output_folder, "flir_det_train"))
    accuracy, tn, fp, fn, tp = confusion_matrix(Y_train, prediction,
                                                save_name=os.path.join(output_folder, "confusion_matrix_train"))
    # Stats around test dataset
    prediction, prediction_proba = predict(X_test, classifier, scaler)
    roc_pr_curves(Y_test, prediction_proba, save_name=os.path.join(output_folder, "flir_roc_test"))
    APCER_test, BPCER_test, ths_test = DET_curve(Y_test, prediction_proba,
                                                 save_name=os.path.join(output_folder, "flir_det_test"))

    accuracy, tn, fp, fn, tp = confusion_matrix(Y_test, prediction,
                                                save_name=os.path.join(output_folder, "confusion_matrix_test"))

    # Generate excel table for DET
    # out_DATA = {
    #     "APCER_train": APCER_train,
    #     "BPCER_train": BPCER_train,
    #     "APCER_test": APCER_test,
    #     "BPCER_test": BPCER_test,
    # }
    #
    # out_DATA_df = pd.DataFrame(out_DATA)
    # out_DATA_df.to_excel(os.path.join(output_folder, "DET_flir.xlsx"))
