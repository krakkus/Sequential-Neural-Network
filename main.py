import numpy as np

import network

# https://github.com/niderhoff/nlp-datasets
# https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/

nrinner = 2048
nrout = 128


def main():
    nw = network.CustomNetwork(layers=(2,25,25,25,25,2),
                               interconnects=10,
                               activation=network.tanh)

    # Number of iterations for training
    num_iterations = 1000
    inp = [0,1]
    oup = [1,0]


    for _ in range(num_iterations):
        # Forward pass
        o = nw.forward(inp)

        # Calculate error
        err = nw.calculate_error(oup)
        print(f'Error: {err}')

        # Backpropagation
        nw.backpropagate(oup)

    print('Training completed.')


if __name__ == '__main__':
    main()