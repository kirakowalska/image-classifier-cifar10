# CIFAR 10 classification

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes `['plane','auto','bird','cat','deer','dog','frog','horse','ship','truck']`, with 6000 images per class.
There are 50000 training images and 10000 test images. 

### Load the data

You can use the following scipt to download and process the data.

```
from load_cifar import load_batch
X_train, y_train = load_batch()
X_test, y_test = load_batch(test=True)
```

### Images in each class

Exemplary images in each class can be plotted by running `implot.py`.
<img src="https://github.com/kirakowalska/image-classifier-cifar10/blob/master/cifar10.png" width="600">

### Train a shallow classifier to serve as a benchmark

We use the [HoG descriptor](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html) for feature extraction from images and linear SVM for classification. The classifier is trained and tested by running `benchmark.py`.

Train accuracy: 51.3%

Test accuracy: **46.4%**

### Extract visual features using a pre-trained CNN network.

Extract visual features from images using the VGG-16 network pretrained on Imagenet. We use the penultimate leayer of VGG-16 for feature computation. We call these features CNN codes.

```
from cnn import compute_features
features_train = compute_features(X_train)
```
We use Principal Component Analysis (PCA) to visualise the CNN nodes in two dimensions (see `experiments.ipynb`). Each colour corresponds to a different class in the CIFAR-10 dataset. You can see that the CNN nodes can separate the image classes into (overlapping) clusters.

<img src="https://github.com/kirakowalska/image-classifier-cifar10/blob/master/cifar10_pca.png" width="600">

### Train a SVM classifier on top of the CNN Codes

We then use the CNN nodes as inputs to a linear SVM classifier*. We can see an improved classification performance on test images.

Train accuracy: 100%

Test accuracy: **81.5%**

*Note that the hyperparameter C of the SVM classifier has been optimised using cross-validation (see `experiments.ipynb`) but the classifier has shown no sensitivity in the tested C range.

### Boost classification accuracy using [voting ensemble](https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/)

Finally, we improve the classification accuracy by classification using voting from three classifiers (logistic regression, decision tree classifier, linear SVM). The ensemble further boosts the classification accuracy. Again, see `experiments.ipynb` for more details.

Test accuracy: **81.9%**

# Voilà! 

Our classifier can now distinguish cats from horses from planes etc. with quite a good accuracy (81.9% to be exact)! Further work could make it even better using data augmentation (blurring, rotating), bagging and other ensemble learning approaches.
