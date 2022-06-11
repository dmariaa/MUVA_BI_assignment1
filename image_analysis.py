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

from tools import compute_lbp_histogram, roc_pr_curves, DET_curve


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


def fit_clf(dataset):
    X = np.vstack(dataset['features'].values)
    y = (dataset['attack'].values != -1).astype(float)

    # LDA
    clf = LDA()
    clf.fit(X, y)
    pred = clf.predict(X)
    pred_proba = clf.predict_proba(X)[:, 1]

    return X, y, pred, pred_proba


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

    X_1, y_1, pred_1, pred_proba_1 = fit_clf(dataset.query('attack!=2 and mode==1'))
    roc_pr_curves(y_1, pred_proba_1, save_name=os.path.join(output_folder, "mode1_roc"))
    APCER_1, BPCER_1, ths_1 = DET_curve(y_1, pred_proba_1, save_name=os.path.join(output_folder, "mode1_det"))

    X_2, y_2, pred_2, pred_proba_2 = fit_clf(dataset.query('attack!=2 and mode==2'))
    roc_pr_curves(y_2, pred_proba_2, save_name=os.path.join(output_folder, "mode2_roc"))
    APCER_2, BPCER_2, ths_2 = DET_curve(y_2, pred_proba_2, save_name=os.path.join(output_folder, "mode2_det"))

    X_3, y_3, pred_3, pred_proba_3 = fit_clf(dataset.query('attack!=2 and mode==3'))
    roc_pr_curves(y_3, pred_proba_3, save_name=os.path.join(output_folder, "mode3_roc"))
    APCER_3, BPCER_3, ths_3 = DET_curve(y_3, pred_proba_3, save_name=os.path.join(output_folder, "mode3_det"))

    X_4, y_4, pred_4, pred_proba_4 = fit_clf(dataset.query('attack!=2 and mode==4'))
    roc_pr_curves(y_4, pred_proba_4, save_name=os.path.join(output_folder, "mode4_roc"))
    APCER_4, BPCER_4, ths_4 = DET_curve(y_4, pred_proba_4, save_name=os.path.join(output_folder, "mode4_det"), log_scale=False)
