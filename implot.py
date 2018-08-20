from load_cifar import load_batch
import numpy as np
import matplotlib.pyplot as plt

X_train, y_train = load_batch()

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
plt.savefig('cifar10test.png')