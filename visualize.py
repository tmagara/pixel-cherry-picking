import pathlib

import chainer
from matplotlib import pyplot


class Visualize(chainer.training.Extension):
    def __init__(self, iterator, target, samples=6, converter=chainer.dataset.convert.concat_examples, device=None):
        self.iterator = iterator
        self.target = target
        self.samples = samples
        self.converter = converter
        self.device = device

    def __call__(self, trainer):
        xp = self.target.xp
        x, labels = self.converter(self.iterator.next(), self.device)
        images_list = self.target.show(x, labels)
        for (path, images) in images_list:
            images = images.data[0:self.samples]
            images = xp.clip(images, 0.0, 1.0)
            rows, columns, c, h, w = images.shape
            images = xp.transpose(images, (0, 3, 1, 4, 2))
            images = xp.reshape(images, (rows * h, columns * w, c))
            images = chainer.backends.cuda.to_cpu(images)
            path_object = pathlib.Path(trainer.out) / path.format(epoch=trainer.updater.epoch)
            save_vhc(images, str(path_object))


def save_vhc(images, filepath, dpi=100):
    figure = pyplot.figure(figsize=(images.shape[1] / dpi, images.shape[0] / dpi), dpi=dpi)
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    axes = figure.add_subplot(1, 1, 1)
    axes.axis('off')
    axes.imshow(images, interpolation="nearest")
    figure.savefig(filepath, dpi=dpi, facecolor='black')
    pyplot.close(figure)
