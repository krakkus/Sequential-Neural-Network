import numpy as np

import network

# https://github.com/niderhoff/nlp-datasets
# https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/

nrinner = 2048
nrout = 128

def runit(nw, data):
    for i in range(0, len(data) - 1):
        pass


def main():
    nw = network.CustomNetwork(layers=(128, 1000, 1000, 128),
                               interconnects=100,
                               activation=network.tanh)

    with open('sample.txt', 'r') as file:
        data = file.read()

    while True:
        d = [0] * 128
        o = nw.forward(d)
        err = nw.calculate_error(d)
        nw.backpropagate(d)
        print('.', end='')
        pass

        #runit(nw, data)


if __name__ == '__main__':
    main()