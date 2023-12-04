import matplotlib.pyplot as plt
from skimage import io


def get_image(path):
    return io.imread(path)

def get_shape(path):
    shape = get_image(path).shape
    height = shape[0]
    width = shape[1]
    layers = shape[-1] if len(shape) == 3 else 1
    return height, width, layers

def show_images(images, columns=3):
    rows = (len(images) - 1) // columns + 1
    fig, ax = plt.subplots(rows, columns, figsize=(8, 2 * rows))
    axs = ax.flatten()

    for i, path in enumerate(images['path']):
        img = io.imread(path)
        axs[i].imshow(img)
        axs[i].axis('Off')

    plt.tight_layout()
    plt.show()

def show_class_samples(df, classes):
    for c in classes:
        fig, ax = plt.subplots(2, 4, figsize=(8, 4))
        fig.suptitle(c)
        axs = ax.flatten()

        samples = df[df['label'] == c].sample(8)

        for i, idx in enumerate(samples.index):
            img = io.imread(df.loc[idx, 'path'])
            axs[i].imshow(img)
            axs[i].axis('Off')

        plt.tight_layout()
        plt.show()