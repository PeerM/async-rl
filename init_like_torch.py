from chainer import links as L
import numpy as np


def init_like_torch(link):
    # Mimic torch's default parameter initialization
    # TODO(muupan): Use chainer's initializers when it is merged
    for l in link.links():
        if isinstance(l, L.Linear):
            out_channels, in_channels = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels)

            # Initialize weights to the standard deviation.
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape)

            # If there are biases, do the same.
            if l.b is not None:
                l.b.data[:] = np.random.uniform(-stdv, stdv,
                                                size=l.b.data.shape)
        elif isinstance(l, L.Convolution2D):
            out_channels, in_channels, kh, kw = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels * kh * kw)

            # Initialize the weights to +/- the standard deviation of
            # the number of weights.
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape)

            # If there are biases, do the same thing.
            if l.b is not None:
                l.b.data[:] = np.random.uniform(-stdv, stdv,
                                                size=l.b.data.shape)



