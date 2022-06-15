import argparse
import glob
import numbers
import os

import cv2
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler

from tools import get_split_dataset_idx, set_split_dataset_idx, compute_lbp_histogram, roc_pr_curves, DET_curve, cm2inch


def face_segmentator(image):
    # Resize image
    factor = 10
    img_aux = cv2.resize(image, (int(image.shape[1] / factor), int(image.shape[0] / factor)))

    # Convert to grayscale
    img_aux_gray = cv2.cvtColor(img_aux, cv2.COLOR_BGR2GRAY)

    # Viola-Jones
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detectamos las caras de las personas en los frames.
    rects = faceCascade.detectMultiScale(img_aux_gray, 1.1, 4)

    # Se dibujan los bounding box de las caras en verde.
    rect_array = np.array(rects)
    (x, y, w, h) = rects[np.argmax(rect_array[:, 2] * rect_array[:, 3])] * factor

    # Extract face
    img_face = image[y:y + h, x:x + w].copy()

    return img_face


def feature_extractor(img_face, mode=4):
    features = np.array([])
    img_face_gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)

    # Edges
    if mode == 1 or mode == 4:
        img_edge = cv2.Canny(img_face_gray, 50, 150)
        features = np.hstack([features, np.sum(img_edge) / (255 * img_edge.shape[0] * img_edge.shape[1])])

    # Lab color space
    if mode == 2 or mode == 4:
        img_face_lab = cv2.cvtColor(img_face, cv2.COLOR_BGR2LAB)
        hist_lab = np.squeeze(cv2.calcHist([img_face_lab], [0], None, [12], [0, 256]))
        features = np.hstack([features, hist_lab])

    # LBP
    if mode == 3 or mode == 4:
        hist_lbp, lbp = compute_lbp_histogram(img_face_gray, 6, 2)
        features = np.hstack([features, hist_lbp])

    return features


def get_image_data(image_path):
    path, file = os.path.split(image_path)
    base_dir = path.split(os.path.sep)[-1]
    file_name, ext = os.path.splitext(file)

    if base_dir == 'USER':
        return_data = {
            'file_path': path,
            'file_name': file_name,
            'attack': -1,
        }
    else:
        attack_number = int(base_dir.split('_')[-1])
        return_data = {
            'file_path': path,
            'file_name': file_name,
            'attack': attack_number,
        }

    return return_data


def extract_features(base_path):
    paths = glob.glob(os.path.join(base_path, "**/*.jpg"), recursive=True)

    features = []

    for mode in range(1, 5):
        for file_path in paths:
            data = get_image_data(file_path)
            image = cv2.imread(file_path)
            img_face = face_segmentator(image)
            data['mode'] = mode
            data['features'] = feature_extractor(img_face, mode)
            features.append(data)

    df = pd.DataFrame(features)
    return df


def fit_clf(X_train, X_test, Y_train, Y_test):
    classifier = LDA()

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    classifier.fit(X_train, Y_train)

    X_test = scaler.transform(X_test)
    prediction = classifier.predict(X_test)
    prediction_proba = classifier.predict_proba(X_test)[:, 1]

    return prediction, prediction_proba


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


    def options():
        parser = argparse.ArgumentParser()
        parser.add_argument('-g', '--generate', help='Regenerate dataset', action='store_true')
        parser.add_argument('-o', '--output', help='Output folder for images and pkl files',
                            default='out/image_analysis')
        return parser


    args = options().parse_args()
    generate = args.generate
    output_folder = args.output

    if generate:
        dataset = extract_features("./data/sony")
        dataset.to_pickle(os.path.join(output_folder, 'dataset.pkl'))
    else:
        dataset = pd.read_pickle(os.path.join(output_folder, 'dataset.pkl'))

    train_idx, test_idx = get_split_dataset_idx(dataset.query('mode==1'))

    # Test

    X_train_1, X_test_1, Y_train_1, Y_test_1 = set_split_dataset_idx(dataset.query('mode==1'), train_idx, test_idx)
    pred_1, pred_proba_1 = fit_clf(X_train_1, X_test_1, Y_train_1, Y_test_1)
    roc_pr_curves(Y_test_1, pred_proba_1, save_name=os.path.join(output_folder, "mode1_roc_test"))
    APCER_1_test, BPCER_1_test, ths_1_test = DET_curve(Y_test_1, pred_proba_1,
                                                       save_name=os.path.join(output_folder, "mode1_det_test"))

    X_train_2, X_test_2, Y_train_2, Y_test_2 = set_split_dataset_idx(dataset.query('mode==2'), train_idx, test_idx)
    pred_2, pred_proba_2 = fit_clf(X_train_2, X_test_2, Y_train_2, Y_test_2)
    roc_pr_curves(Y_test_2, pred_proba_2, save_name=os.path.join(output_folder, "mode2_roc_test"))
    APCER_2_test, BPCER_2_test, ths_2_test = DET_curve(Y_test_2, pred_proba_2,
                                                       save_name=os.path.join(output_folder, "mode2_det_test"))

    X_train_3, X_test_3, Y_train_3, Y_test_3 = set_split_dataset_idx(dataset.query('mode==3'), train_idx, test_idx)
    pred_3, pred_proba_3 = fit_clf(X_train_3, X_test_3, Y_train_3, Y_test_3)
    roc_pr_curves(Y_test_3, pred_proba_3, save_name=os.path.join(output_folder, "mode3_roc_test"))
    APCER_3_test, BPCER_3_test, ths_3_test = DET_curve(Y_test_3, pred_proba_3,
                                                       save_name=os.path.join(output_folder, "mode3_det_test"))

    X_train_4, X_test_4, Y_train_4, Y_test_4 = set_split_dataset_idx(dataset.query('mode==4'), train_idx, test_idx)
    pred_4, pred_proba_4 = fit_clf(X_train_4, X_test_4, Y_train_4, Y_test_4)
    roc_pr_curves(Y_test_4, pred_proba_4, save_name=os.path.join(output_folder, "mode4_roc_test"))
    APCER_4_test, BPCER_4_test, ths_4_test = DET_curve(Y_test_4, pred_proba_4,
                                                       save_name=os.path.join(output_folder, "mode4_det_test"))

    # Train
    X_train_1, X_test_1, Y_train_1, Y_test_1 = set_split_dataset_idx(dataset.query('mode==1'), train_idx, test_idx)
    pred_1, pred_proba_1 = fit_clf(X_train_1, X_train_1, Y_train_1, Y_train_1)
    roc_pr_curves(Y_train_1, pred_proba_1, save_name=os.path.join(output_folder, "mode1_roc_train"))
    APCER_1_train, BPCER_1_train, ths_1_train = DET_curve(Y_train_1, pred_proba_1,
                                                          save_name=os.path.join(output_folder, "mode1_det_train"))

    X_train_2, X_test_2, Y_train_2, Y_test_2 = set_split_dataset_idx(dataset.query('mode==2'), train_idx, test_idx)
    pred_2, pred_proba_2 = fit_clf(X_train_2, X_train_2, Y_train_2, Y_train_2)
    roc_pr_curves(Y_train_2, pred_proba_2, save_name=os.path.join(output_folder, "mode2_roc_train"))
    APCER_2_train, BPCER_2_train, ths_2_train = DET_curve(Y_train_2, pred_proba_2,
                                                          save_name=os.path.join(output_folder, "mode2_det_train"))

    X_train_3, X_test_3, Y_train_3, Y_test_3 = set_split_dataset_idx(dataset.query('mode==3'), train_idx, test_idx)
    pred_3, pred_proba_3 = fit_clf(X_train_3, X_train_3, Y_train_3, Y_train_3)
    roc_pr_curves(Y_train_3, pred_proba_3, save_name=os.path.join(output_folder, "mode3_roc_train"))
    APCER_3_train, BPCER_3_train, ths_3_train = DET_curve(Y_train_3, pred_proba_3,
                                                          save_name=os.path.join(output_folder, "mode3_det_train"))

    X_train_4, X_test_4, Y_train_4, Y_test_4 = set_split_dataset_idx(dataset.query('mode==4'), train_idx, test_idx)
    pred_4, pred_proba_4 = fit_clf(X_train_4, X_train_4, Y_train_4, Y_train_4)
    roc_pr_curves(Y_train_4, pred_proba_4, save_name=os.path.join(output_folder, "mode4_roc_train"))
    APCER_4_train, BPCER_4_train, ths_4_train = DET_curve(Y_train_4, pred_proba_4,
                                                          save_name=os.path.join(output_folder, "mode4_det_train"),
                                                          log_scale=False)
