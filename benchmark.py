from skimage.feature import hog
import numpy as np
def compute_hog_features(images):
    features = []
    hog_images = []
    for image in images:
        fd,hog_image = hog(image,visualize=True)
        features.append(fd)
        hog_images.append(hog_image)
    return np.array(features), hog_images

if __name__ == '__main__':

    # Import data
    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = y_train.flatten()
    y_test  = y_test.flatten()

    # Subset data to 1/10th
    n_train = X_train.shape[0] // 10
    X_train = X_train[0:n_train]
    y_train = y_train[0:n_train]
    print("Number of training images:", n_train)
    # Number of training images: 5000

    n_test = X_test.shape[0] // 10
    X_test = X_test[0:n_test]
    y_test = y_test[0:n_test]
    print("Number of testing images:", n_test)
    # Number of testing images: 1000

    # Get HOG features
    print("Computing train features...")
    features_train, _ = compute_hog_features(X_train)
    print("Computing test features...")
    features_test, _ = compute_hog_features(X_test)

    # Train SVM
    from sklearn.svm import LinearSVC
    clf = LinearSVC()

    print("Training...")
    clf.fit(features_train, y_train)
    print(clf.score(features_train, y_train))

    print("Testing...")
    print(clf.score(features_test, y_test))

    # Training...
    # 0.513
    # Testing...
    # 0.464
