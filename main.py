import numpy as np

import network

# https://github.com/niderhoff/nlp-datasets
# https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/

nrinner = 2048
nrout = 128


def main():
    nw = network.CustomNetwork(layers=(2, 10, 10, 1),
                               interconnects=10,
                               activation=network.tanh)

    # Number of iterations for training
    num_iterations = 1000

    # AND
    data = [[[1, 1], [1]],
            [[1, 0], [0]],
            [[0, 1], [0]],
            [[0, 0], [0]]]

    nw.train(num_iterations, data)

    print('Training completed.')


if __name__ == '__main__':
    main()