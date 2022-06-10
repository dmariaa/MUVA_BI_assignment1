import glob

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from image_analysis import face_segmentator, feature_extractor
from tools import roc_pr_curves, DET_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def extract_features_ddbb(all_users_path, all_spoofs_path, modo=4):
    # Users
    users_paths = glob.glob(all_users_path + "./**/*.jpg", recursive=True)

    features_users = []
    user_names = []
    for user_path in users_paths:
        image = cv2.imread(user_path)
        img_face = face_segmentator(image)
        user_names.append(user_path.split("/")[-1].split(".jpg")[0])
        features_users.append(feature_extractor(img_face, modo))

    features_users = np.array(features_users)
    features_users = pd.DataFrame(features_users)
    features_users.index = user_names

    # attack_01
    spoofs_paths = glob.glob(all_spoofs_path + "./**/*.jpg", recursive=True)

    features_spoofs = []
    spoof_names = []
    for spoof_path in spoofs_paths:
        image = cv2.imread(spoof_path)
        img_face = face_segmentator(image)
        spoof_names.append(spoof_path.split("/")[-1].split(".jpg")[0])
        features_spoofs.append(feature_extractor(img_face, modo))

    features_spoofs = np.array(features_spoofs)
    features_spoofs = pd.DataFrame(features_spoofs)
    features_spoofs.index = spoof_names

    return features_users, features_spoofs


def fit_clf(features_users, features_spoofs):
    X = np.vstack((features_users.values, features_spoofs.values))
    y = np.squeeze(np.vstack((np.zeros((features_users.shape[0], 1)), np.ones((features_users.shape[0], 1)))))

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

    all_users_path = "./data/SONY/USER/"
    all_spoofs_path = "./data/SONY/attack_01/"

    features_users, features_spoofs = extract_features_ddbb(all_users_path, all_spoofs_path, modo=1)
    X_1, y_1, pred_1, pred_proba_1 = fit_clf(features_users, features_spoofs)

    features_users, features_spoofs = extract_features_ddbb(all_users_path, all_spoofs_path, modo=2)
    X_2, y_2, pred_2, pred_proba_2 = fit_clf(features_users, features_spoofs)

    features_users, features_spoofs = extract_features_ddbb(all_users_path, all_spoofs_path, modo=3)
    X_3, y_3, pred_3, pred_proba_3 = fit_clf(features_users, features_spoofs)

    features_users, features_spoofs = extract_features_ddbb(all_users_path, all_spoofs_path, modo=4)
    X_4, y_4, pred_4, pred_proba_4 = fit_clf(features_users, features_spoofs)

    roc_pr_curves(y_1, pred_proba_1, save_name="1")
    roc_pr_curves(y_2, pred_proba_2, save_name="2")
    roc_pr_curves(y_3, pred_proba_3, save_name="3")
    roc_pr_curves(y_4, pred_proba_4, save_name="4")

    APCER_1, BPCER_1, ths_1 = DET_curve(y_1, pred_proba_1, save_name="1")
    APCER_2, BPCER_2, ths_2 = DET_curve(y_2, pred_proba_2, save_name="2")
    APCER_3, BPCER_3, ths_3 = DET_curve(y_3, pred_proba_3, save_name="3")
    APCER_4, BPCER_4, ths_4 = DET_curve(y_4, pred_proba_4, save_name="4", log_scale=False)
