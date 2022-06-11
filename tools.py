import glob
import numbers
import os.path
import shutil
import sys

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from skimage import feature
from sklearn import metrics
from scipy.interpolate import interp1d

eps = sys.float_info.epsilon


def compute_multichannel_lbp_histogram(image, number_of_points, radius, method='uniform'):
    histograms = []

    for channel in range(image.shape[-1]):
        histogram, lbp = compute_lbp_histogram(image[..., channel], number_of_points, radius, method)
        histograms.append(histogram)

    return np.array(histograms)


def compute_lbp_histogram(image, number_of_points, radius, method='uniform'):
    lbp = feature.local_binary_pattern(image, number_of_points, radius, method)

    n_bins = int(lbp.max() + 1)
    histogram, _ = np.histogram(lbp.ravel(),
                                bins=n_bins,
                                range=(0, n_bins))

    histogram = histogram / (np.sum(histogram) + eps)

    return histogram, lbp


def DET_curve(y, pred_proba, log_scale=True, save_name=None):
    y_users = y[np.where(y == 1)]
    y_spoofs = y[np.where(y == 0)]

    pred_proba_users = pred_proba[np.where(y == 1)]
    pred_proba_spoofs = pred_proba[np.where(y == 0)]

    ths = []
    APCER = []
    BPCER = []

    for th in np.arange(0, 1, 0.001):
        NPAIS = y_spoofs.shape[0]
        RESi_apcer = pred_proba_spoofs < th
        APCER.append((1 - (1 / NPAIS) * np.sum(RESi_apcer)) * 100)

        NBF = y_users.shape[0]
        RESi_bpcer = pred_proba_users < th
        BPCER.append((1 / NBF) * np.sum(RESi_bpcer) * 100)

        ths.append(th)

    ths = np.array(ths)
    APCER = np.array(APCER)
    BPCER = np.array(BPCER)

    f, (ax1) = plt.subplots(1, 1, figsize=cm2inch(9, 7))

    lc = colorline(APCER, BPCER, ths)
    plt.colorbar(lc)
    ax1.set_ylabel('BPCER (%)')
    ax1.set_xlabel('APCER (%)')

    if log_scale:
        ax1.set_xscale("log")
        ax1.set_xticks([0.01, 0.1, 1, 10, 100])
        ax1.set_xticklabels(['0.01', '0.1', '1', '10', '100'])
        ax1.set_yscale("log")
        ax1.set_yticks([0.01, 0.1, 1, 10, 100])
        ax1.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
        ax1.set_xlim([0.01, 120]), ax1.set_ylim([0.01, 120])
    else:
        ax1.set_xticks([0, 100])
        ax1.set_xticklabels(['0', '100'])
        ax1.set_yticks([0, 100])
        ax1.set_yticklabels(['0', '100'])
        ax1.set_xlim([-5, 105]), ax1.set_ylim([-5, 105])

    if save_name is not None:
        plt.savefig(save_name + ".pdf", format='pdf', bbox_inches='tight')
        plt.savefig(save_name + ".svg", format='svg', bbox_inches='tight')

    f.tight_layout()
    f.show()

    return APCER, BPCER, ths


def roc_pr_curves(y, pred_proba, save_name=None):
    fpr, tpr, thresholds_roc = metrics.roc_curve(y, pred_proba)
    precision, recall, thresholds_pr = metrics.precision_recall_curve(y, pred_proba)

    fpr, tpr, thresholds_roc = resample(fpr, tpr, thresholds_roc)
    recall, precision, thresholds_pr = resample(np.flip(recall), np.flip(precision),
                                                np.flip(np.hstack((thresholds_pr, 1))))

    gs_kw = dict(width_ratios=[2.5, 3])
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=cm2inch(15, 7), gridspec_kw=gs_kw)

    lc = colorline(fpr, tpr, thresholds_roc, ax1)
    # ax1.plot(fpr, tpr, label='a' , linewidth=1, color = [0,0,0])
    ax1.set_ylabel('True Positive Rate (recall)')
    ax1.set_xlabel('False Positive Rate (1 - specificity)')
    ax1.xaxis.set_label_coords(0.5, -0.165)
    ax1.plot([0, 1], ls=':', c="0.5", linewidth=1, label="Baseline")
    ax1.set_xlim([-0.05, 1.05]), ax1.set_ylim([-0.05, 1.05])
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1]), ax2.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1]), ax2.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])

    lc = colorline(recall, precision, thresholds_pr, ax2)
    # ax2.plot(recall, precision, label='a' , linewidth=1, color = [0,0,0])
    ax2.set_ylabel('Precision')
    ax2.set_xlabel('Recall')
    ax2.xaxis.set_label_coords(0.5, -0.165)
    bl = np.sum(y) / y.shape
    ax2.plot([0, 1], [bl, bl], ls=':', c="0.5", linewidth=1, label='Baseline')
    ax2.set_xlim([-0.05, 1.05]), ax2.set_ylim([-0.05, 1.05])
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1]), ax2.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1]), ax2.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])

    plt.colorbar(lc)

    if save_name is not None:
        plt.savefig(save_name + ".pdf", format='pdf', bbox_inches='tight')
        plt.savefig(save_name + ".svg", format='svg', bbox_inches='tight')

    f.tight_layout()
    f.show()


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates,
    in the correct format for LineCollection:
    an array of the form
    numlines x (points per line) x 2 (x and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(x, y, z=None, axes=None,
              cmap=plt.get_cmap('viridis'),
              norm=plt.Normalize(0.0, 1.0), linewidth=4, alpha=1.0,
              **kwargs):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if isinstance(z, numbers.Real):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = matplotlib.collections.LineCollection(
        segments, array=z, cmap=cmap, norm=norm,
        linewidth=linewidth, alpha=alpha, **kwargs
    )

    if axes is None:
        axes = plt.gca()

    axes.add_collection(lc)
    axes.autoscale()

    return lc


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def resample(x, y, z):
    delta_x = np.linspace(0, 0.000001, num=len(x), endpoint=True)
    x = x + delta_x
    f = interp1d(x, y)
    xnew = np.linspace(0, 1, num=1000, endpoint=True)
    ynew = f(xnew)

    delta_x = np.linspace(0, 0.000001, num=len(x), endpoint=True)
    x = x + delta_x
    f = interp1d(x, z)
    xnew = np.linspace(0, 1, num=1000, endpoint=True)
    znew = f(xnew)

    return xnew, ynew, znew


def change_files_case(folder):
    files = glob.glob(os.path.join(folder, "**/*.*"), recursive=True)
    for file in files:
        if os.path.isfile(file):
            new_name = file.lower()
            # os.rename(file, new_name)
            shutil.move(file, new_name)


if __name__ == "__main__":
    """ uncomment next line to change files case sensitivity to make life easier... """
    # change_files_case('./data')
