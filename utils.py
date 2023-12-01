import os
import numpy as np
import pickle as pk
from skimage.transform import resize
from skimage.io import imread, imshow, show
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import RocCurveDisplay, accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_images(dataset_path):
    X = []
    Y = []
    classes = []
    for subdir in os.listdir(dataset_path):
        classes.append(subdir)
        cnt = 0
        for file in os.listdir(dataset_path + subdir):
            img = imread(dataset_path + subdir + '/' + file)
            X.append(img)
            Y.append(classes.index(subdir))
            cnt += 1
        print(f'Class {subdir} loaded. Instances: {cnt}')
    return np.array(X), np.array(Y), classes

def grayscale_transform(X):
    return rgb2gray(X)

def PCA_fit(X, n_components=65):
    pca = PCA(n_components=n_components)
    X = X.reshape(len(X), -1)
    pca.fit(X)
    return pca

def save_obj(pca, filename):
    pk.dump(pca, open(filename, 'wb'))

def load_obj(filename):
    if os.path.exists(filename):
        pca = pk.load(open(filename, 'rb'))
        return pca

def PCA_transform(X, pca):
    X = X.reshape(len(X), -1)
    X = pca.transform(X)
    return X

def channel_transform(X):
    # separate the channels for each image
    X_r = X[:,:,:,0]
    X_g = X[:,:,:,1]
    X_b = X[:,:,:,2]
    return X_r, X_g, X_b

def channel_wise_PCA(X_train, X_val, X_test, standardize_features=True, with_hog=False):
    # extract channels
    X_r, X_g, X_b = channel_transform(X_train)

    if with_hog:
        X_train_hog, X_val_hog, X_test_hog = hog_transform(X_train, X_val, X_test, standardize_features=standardize_features)

    # PCA on each channel
    pca_r_path = 'pca_r.npy'
    pca_g_path = 'pca_g.npy'
    pca_b_path = 'pca_b.npy'

    pca_r = load_obj(pca_r_path)
    if pca_r is None:
        pca_r = PCA_fit(X_r)
        save_obj(pca_r, pca_r_path)
    print(f'PCA red explained variance ratio: {pca_r.explained_variance_ratio_}')

    pca_g = load_obj(pca_g_path)
    if pca_g is None:
        pca_g = PCA_fit(X_g)
        save_obj(pca_g, pca_g_path)
    print(f'PCA green explained variance ratio: {pca_g.explained_variance_ratio_}')

    pca_b = load_obj(pca_b_path)
    if pca_b is None:
        pca_b = PCA_fit(X_b)
        save_obj(pca_b, pca_b_path)
    print(f'PCA blue explained variance ratio: {pca_b.explained_variance_ratio_}')

    # transform each channel
    X_r = PCA_transform(X_r, pca_r)
    X_g = PCA_transform(X_g, pca_g)
    X_b = PCA_transform(X_b, pca_b)

    # concatenate the channels
    X_train = np.concatenate((X_r, X_g, X_b), axis=1)

    # transform validation set
    X_r, X_g, X_b = channel_transform(X_val)

    X_r = PCA_transform(X_r, pca_r)
    X_g = PCA_transform(X_g, pca_g)
    X_b = PCA_transform(X_b, pca_b)

    X_val = np.concatenate((X_r, X_g, X_b), axis=1)

    # transform test set
    X_r, X_g, X_b = channel_transform(X_test)
    
    X_r = PCA_transform(X_r, pca_r)
    X_g = PCA_transform(X_g, pca_g)
    X_b = PCA_transform(X_b, pca_b)

    X_test = np.concatenate((X_r, X_g, X_b), axis=1)

    # standardize the data
    if standardize_features:
        X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    if with_hog:
        # concatenate hog features
        X_train = np.concatenate((X_train, X_train_hog), axis=1)
        X_val = np.concatenate((X_val, X_val_hog), axis=1)
        X_test = np.concatenate((X_test, X_test_hog), axis=1)

    return X_train, X_val, X_test

def grayscale_PCA(X_train, X_val, X_test, standardize_features=True):
    # transform training set
    X_train = grayscale_transform(X_train)

    pca_gs_path = 'pca_gs.npy'
    pca = load_obj(pca_gs_path)
    if pca is None:
        pca = PCA_fit(X_train)
        save_obj(pca, pca_gs_path)
    print(f'PCA grayscale explained variance ratio: {pca.explained_variance_ratio_}')

    # transform training set
    X_train = PCA_transform(X_train, pca)

    # transform validation set
    X_val = grayscale_transform(X_val)
    X_val = PCA_transform(X_val, pca)

    # transform test set
    X_test = grayscale_transform(X_test)
    X_test = PCA_transform(X_test, pca)

    # standardize the data
    if standardize_features:
        X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    return X_train, X_val, X_test

# extract hog feature vectors from images
def hog_transform(X_train, X_val, X_test, standardize_features=True):
    # transform training set
    X_train_path = 'X_train_hog.npy'
    X_train_temp = load_obj(X_train_path)
    if X_train_temp is None:
        X_train_temp = np.array([hog(img, channel_axis=-1) for img in tqdm(X_train)])
        save_obj(X_train_temp, X_train_path)
    X_train = X_train_temp

    pca_hog_path = 'pca_hog.npy'
    pca = load_obj(pca_hog_path)
    if pca is None:
        pca = PCA_fit(X_train, n_components=100)
        save_obj(pca, pca_hog_path)
    print(f'PCA hog explained variance ratio: {pca.explained_variance_ratio_}')

    # transform training set
    X_train = PCA_transform(X_train, pca)

    # transform validation set
    X_val = np.array([hog(img, channel_axis=-1) for img in tqdm(X_val)])
    X_val = PCA_transform(X_val, pca)

    # transform test set
    X_test = np.array([hog(img, channel_axis=-1) for img in tqdm(X_test)])
    X_test = PCA_transform(X_test, pca)

    # standardize the data
    if standardize_features:
        X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    return X_train, X_val, X_test

def standardize(X_train, X_val, X_test):
    # standardize training set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # standardize validation set
    X_val = scaler.transform(X_val)

    # standardize test set
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test
