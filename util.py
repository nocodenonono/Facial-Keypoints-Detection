import numpy as np
import matplotlib.pyplot as plt


def rotate_pt(x, y, degree, origin):
    """
    Rotate a point around a given point.

    Reference: https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
    """
    ox, oy = origin
    radians = -degree * np.pi / 180
    xx = ox + np.cos(radians) * (x - ox) - np.sin(radians) * (y - oy)
    yy = oy + np.sin(radians) * (x - ox) + np.cos(radians) * (y - oy)

    return xx, yy


def show_image(tensor, anno):
    """
    plot annotated image

    Input:
        tensor -> Batch_size x channel_num x H x W
        anno -> Batch_size x 136
    """
    tensor = tensor.cpu().clone()
    tensor = tensor[0].squeeze(0)
    tensor = tensor.squeeze(0)

    anno = anno.cpu().clone()
    anno = anno[0].squeeze(0)
    plt.imshow(tensor, cmap='gray')
    plt.scatter(anno[:68], anno[68:])
    plt.show()


def plot_loss(train_loss, val_loss):
    """
    Plot training loss and validation loss history
    """
    plt.plot(train_loss, 'g', label='train loss')
    plt.plot(val_loss, 'b', label='val loss')
    plt.legend(loc="upper right")
    plt.ylabel('Loss')
    plt.xlabel('Batch number')
    plt.show()
