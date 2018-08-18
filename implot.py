from load_cifar import load_batch
import numpy as np
import matplotlib.pyplot as plt

images , labels = load_batch()
images = np.rollaxis(np.reshape(images, (10000, 3, 32, 32)), 1, 4)

n=0
fig = plt.figure()
for c in range(0,10):
    # Find images in class c
    idxs = np.where(labels == c)[0][0:10]

    # Add them to the plot
    for i in idxs:
        a = fig.add_subplot(10, 10, n + 1)
        plt.imshow(images[i,:,:,:])
        plt.axis("off")
        n+=1

fig.set_size_inches(12,12)
plt.savefig('cifar10.png')

