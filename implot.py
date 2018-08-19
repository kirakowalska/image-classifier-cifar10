import matplotlib.pyplot as plt
import numpy as np

# Import data
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test  = y_test.flatten()

n=0
fig = plt.figure()
for c in range(0,10):
    # Find images in class c
    idxs = np.where(y_train == c)[0][0:10]

    # Add them to the plot
    for i in idxs:
        a = fig.add_subplot(10, 10, n + 1)
        plt.imshow(X_train[i,:,:,:])
        plt.axis("off")
        n+=1

fig.set_size_inches(12,12)
plt.savefig('cifar10.png')

