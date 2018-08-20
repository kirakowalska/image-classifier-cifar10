from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
from load_cifar import load_batch
import pickle

image_size = 224
model = VGG16(weights='imagenet', include_top=False)
model.summary()

def compute_features(images):
    features = []
    for image in images:
        features.append(cv2.resize(image, (image_size, image_size)))

    features = preprocess_input(np.array(features))
    features = model.predict(features)
    features = np.array([feature.flatten() for feature in features])
    return features

if __name__ == '__main__':

    ### Import data
    X_train, y_train = load_batch()
    X_test, y_test = load_batch(test=True)

    ### Subset data to 1/10th
    n_train = X_train.shape[0] // 2
    X_train = X_train[0:n_train]
    y_train = y_train[0:n_train]
    print("Number of training images:", n_train)
    # Number of training images: 5000

    n_test = X_test.shape[0] // 10
    X_test = X_test[0:n_test]
    y_test = y_test[0:n_test]
    print("Number of testing images:", n_test)
    # Number of testing images: 1000

    ### Compute features
    print("Computing train features...")
    features_train = compute_features(X_train)
    print("Computing test features...")
    features_test = compute_features(X_test)

    # Dump features for later reuse
    pickle.dump({"features_train":features_train,"y_train":y_train}, open("cnn_train.pickle", "wb"))
    pickle.dump({"features_test": features_test, "y_test": y_test}, open("cnn_test.pickle", "wb"))
