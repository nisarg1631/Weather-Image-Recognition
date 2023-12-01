# Weather Image Classification using ML

## Dataset

The dataset used for this project is the [Weather Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) from Kaggle. This dataset contains 6862 images of different types of weather, it can be used to implement weather classification based on the photo. The pictures are divided into 11 classes: dew, fog/smog, frost, glaze, hail, lightning , rain, rainbow, rime, sandstorm and snow.

## Model

The models trained for this project are:

1. Naive Bayes
2. Decision Tree
3. Support Vector Machine

## Feature Extraction Techniques

1. Histogram of Oriented Gradients (HOG)
2. Principal Component Analysis (PCA)

## File Structure

```
dataset/
    dew/
    fogsmog/
    frost/
    glaze/
    hail/
    lightning/
    rain/
    rainbow/
    rime/
    sandstorm/
    snow/
resized/ [same as dataset/ but resized to 200x200x3]
results/ [results of the models]
    results_dt.txt [results of Decision Tree]
    results_nb.txt [results of Naive Bayes]
    results_svm.txt [results of Support Vector Machine]
    dt_roc.png [ROC curve of Decision Tree]
    nb_roc.png [ROC curve of Naive Bayes]
    svm_roc.png [ROC curve of Support Vector Machine]
utils.py [utility functions for feature extraction]
preprocess.py [preprocess the dataset, resize the images]
classifier_dt.py [Decision Tree classifier]
classifier_nb.py [Naive Bayes classifier]
classifier_svm.py [Support Vector Machine classifier]
M*_dt.pkl [trained Decision Tree models]
M*_nb.pkl [trained Naive Bayes models]
M*_svm.pkl [trained Support Vector Machine models]
pca_r.npy [PCA components - red channel]
pca_g.npy [PCA components - green channel]
pca_b.npy [PCA components - blue channel]
pca_gs.npy [PCA components - grayscale]
pca_hog.npy [PCA components - HOG]
X_train_hog.npy [HOG features of training set]
README.md [this file]
```

## How to run

First download the dataset from [here](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) and extract it to the current directory (like the above file structure).

With the above file structure, run the following commands:

```
python3 preprocess.py [if the dataset is not resized, this creates resized/ directory]
python3 classifier_dt.py [train Decision Tree classifier]
python3 classifier_nb.py [train Naive Bayes classifier]
python3 classifier_svm.py [train Support Vector Machine classifier]
```

** if the trained models are not provided, the above commands will train the models and save them in the current directory, else the trained models will be loaded from the current directory
