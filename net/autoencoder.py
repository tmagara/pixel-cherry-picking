import chainer


class Encoder(chainer.Chain):
    def __init__(self, units):
        super().__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.c0 = chainer.links.Convolution2D(units[0], units[1], 4, 2, 1, True, w)
            self.c1 = chainer.links.Convolution2D(units[1], units[2], 4, 2, 1, True, w)
            self.c2 = chainer.links.Convolution2D(units[2], units[3], 4, 2, 1, True, w)
            self.c3 = chainer.links.Convolution2D(units[3], units[4] * 2, 4, 2, 1, False, w)

            self.bn0 = chainer.links.BatchNormalization(units[1])
            self.bn1 = chainer.links.BatchNormalization(units[2])
            self.bn2 = chainer.links.BatchNormalization(units[3])

    def __call__(self, x):
        h = x * 2 - 1

        h = chainer.functions.relu(self.bn0(self.c0(h)))
        h = chainer.functions.relu(self.bn1(self.c1(h)))
        h = chainer.functions.relu(self.bn2(self.c2(h)))
        h = self.c3(h)
        h = chainer.functions.average_pooling_2d(h, h.shape[2:])
        h1, h2 = chainer.functions.split_axis(h, 2, 1)

        return h1, h2


class Decoder(chainer.Chain):
    def __init__(self, units, bottom_width=4):
        super().__init__()

        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.dc0 = chainer.links.Deconvolution2D(units[0], units[1], bottom_width, 1, 0, True, initialW=w)
            self.dc1 = chainer.links.Deconvolution2D(units[1], units[2], 4, 2, 1, True, initialW=w)
            self.dc2 = chainer.links.Deconvolution2D(units[2], units[3], 4, 2, 1, True, initialW=w)
            self.dc3 = chainer.links.Deconvolution2D(units[3], units[4], 4, 2, 1, False, initialW=w)

            self.bn0 = chainer.links.BatchNormalization(units[1])
            self.bn1 = chainer.links.BatchNormalization(units[2])
            self.bn2 = chainer.links.BatchNormalization(units[3])

    def __call__(self, z):
        h = z

        h = chainer.functions.relu(self.bn0(self.dc0(h)))
        h = chainer.functions.relu(self.bn1(self.dc1(h)))
        h = chainer.functions.relu(self.bn2(self.dc2(h)))
        h = self.dc3(h)

        return h


class Model1(chainer.Chain):
    def __init__(self, units_e, units_d, out_channels, out_candidates):
        super().__init__()
        self.out_channels = out_channels
        self.out_candidates = out_candidates

        with self.init_scope():
            self.encoder = Encoder([out_channels] + units_e)
            self.decoder = Decoder(units_d + [out_candidates * out_channels])

    def process(self, x, labels, vae=True):
        xp = self.xp

        z_mu, z_ln_var = self.encoder(x)
        z = z_mu
        if vae:
            z_std = chainer.functions.exp(0.5 * z_ln_var)
            r = xp.random.normal(size=z_ln_var.shape).astype(z_ln_var.dtype)
            z += z_std * r

        y = self.decoder(z)
        y = chainer.functions.sigmoid(y)
        N, _, H, W = y.shape
        y = chainer.functions.reshape(y, (N, self.out_channels, self.out_candidates, H, W))
        return y, z_mu, z_ln_var

    def __call__(self, x, labels):
        y, z_mu, z_ln_var = self.process(x, labels, True)

        mse, log_mse = self.calculate_loss(x[:, :, None], y)
        mse = chainer.functions.mean(mse)
        log_mse = chainer.functions.mean(log_mse)

        z_loss = chainer.functions.gaussian_kl_divergence(z_mu, z_ln_var, 'no')
        z_loss = chainer.functions.mean(z_loss)

        chainer.report({
            'mse': mse,
            'log_mse': log_mse,
            'z_loss': z_loss,
        }, self)

        loss = mse * (x.shape[1] * x.shape[2] * x.shape[3])
        loss += z_loss

        return loss

    def calculate_loss(self, x, y):
        mse = chainer.functions.square(x - y)
        mse = chainer.functions.mean(mse, (1,), keepdims=True)
        mse = chainer.functions.min(mse, (2,), keepdims=True)
        mse = chainer.functions.mean(mse, (3, 4), keepdims=True)
        log_mse = 0.5 * chainer.functions.log(mse)
        return mse, log_mse

    def show(self, x, labels):
        y, _, _ = self.process(x, labels, True)

        dump = self._dump(x, y)
        sorted_y = self.sort_y(y)
        dump_sorted = self._dump(x, sorted_y)

        return [
            ('dump_recent.png', dump),
            ('dump_sorted_recent.png', dump_sorted),
            ('dump_{epoch}.png', dump),
            ('dump_sorted_{epoch}.png', dump_sorted),
        ]

    def sort_y(self, y):
        xp = self.xp
        N, C, M, H, W = y.shape

        rgby = xp.array([0.2126, 0.7152, 0.0722], y.dtype)[None, :, None, None, None]
        yy = xp.power(y.data, 2.2)
        yy = xp.sum(rgby * yy, (0, 1,), keepdims=True)

        indices = xp.argsort(yy, 2)
        onehot = xp.eye(M)[indices]
        onehot = xp.transpose(onehot, (0, 1, 2, 5, 3, 4))

        sorted = chainer.functions.sum(y[:, :, None] * onehot, (3,))
        return sorted

    def _dump(self, x, y):
        x = x[:, :, None]
        _, best_mask = self.cherry_pick(x, y.data)
        y, best_mask = chainer.functions.broadcast(y, best_mask)
        y_best = chainer.functions.sum(y * best_mask, (2,))
        y_best = y_best[:, :, None]

        if y.shape[2] > 4:
            even = chainer.functions.concat((x, y_best, y), 2)
            odd = chainer.functions.concat((x * 0, y_best * 0, best_mask), 2)
            output = chainer.functions.stack((even, odd), 1)
            output = chainer.functions.reshape(output, (-1, ) + output.shape[2:])
        else:
            output = chainer.functions.concat((x, y_best, y,  best_mask), 2)

        return output

    def cherry_pick(self, x, y):
        xp = self.xp
        pixel_loss = (y - x) ** 2
        pixel_loss = xp.sum(pixel_loss, (1,), keepdims=True)
        min_indices = xp.argmin(pixel_loss, 2)
        mask = xp.eye(y.shape[2])[min_indices]
        mask = xp.transpose(mask, (0, 1, 4, 2, 3))
        mask = mask.astype(x.dtype)
        return min_indices, mask
