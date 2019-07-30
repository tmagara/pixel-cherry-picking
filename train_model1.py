import argparse

import chainer

import net.autoencoder
import visualize


@chainer.training.make_extension(priority=chainer.training.PRIORITY_WRITER)
def observe_weight(trainer):
    model = trainer.updater.get_optimizer('main').target
    xp = model.xp
    ww_sum = 0
    ww_size = 0
    for p in model.params(False):
        ww_sum += xp.sum(xp.square(p.data))
        ww_size += p.size
    trainer.observation['weight_std'] = xp.sqrt(ww_sum / ww_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=128, help='Number of images in each mini-batch')
    parser.add_argument('--dim-z', '-z', default=64, type=int, help='dimension of encoded vector')
    parser.add_argument('--masks', '-m', default=4, type=int, help='Number of mask images')
    parser.add_argument('--epoch', '-e', default=20, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.config.autotune = True
        chainer.backends.cuda.get_device_from_id(args.gpu).use()

    model = net.autoencoder.Model1(
        [64, 128, 256, args.dim_z],
        [args.dim_z, 256, 128, 64],
        3,
        args.masks
    )
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_cifar10()
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, False, False)
    dump_iter = chainer.iterators.SerialIterator(test, args.batchsize)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.trainer.Trainer(updater, (args.epoch, 'epoch'), args.out)

    trainer.extend(chainer.training.extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(chainer.training.extensions.FailOnNonNumber())
    trainer.extend(observe_weight, trigger=(1, 'epoch'))
    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.PrintReport([
        'epoch',
        'main/mse', 'validation/main/mse',
        'main/z_loss', 'validation/main/z_loss',
        'weight_std',
        'elapsed_time',
    ]))
    trainer.extend(visualize.Visualize(dump_iter, model, device=args.gpu), trigger=(1, 'epoch'))

    trainer.run()


if __name__ == '__main__':
    main()
