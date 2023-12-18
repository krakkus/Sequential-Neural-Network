import numpy as np
import cv2
import network

# https://github.com/niderhoff/nlp-datasets
# https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/


def draw_points(points, margin=0.1):
    # Create a blank 500x500 black image
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    # Convert points from range [-1-margin, 1+margin] to [0, 500]
    points = [(int(((x+1)*(1-margin)+margin)*250), int(((y+1)*(1-margin)+margin)*250)) for x, y in points]

    # Draw each point
    for x, y in points:
        cv2.circle(img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    # Display the image
    cv2.imshow('Image', img)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()

def generate_points_on_circle(n_points, radius=1):
    # Generate an array of angles evenly spaced around the circle
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    # Calculate the x and y coordinates of the points
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    # Combine the x and y coordinates into a list of points
    points = list(zip(x, y))

    return points

def generate_points_on_sinus(n_points, qwerty):
    # Generate an array of x values evenly spaced from -1 to +1
    x = np.linspace(-1, 1, n_points)

    # Calculate the y coordinates of the points
    y = np.sin(np.pi * x)

    # Combine the x and y coordinates into a list of points
    points = list(zip(x, y))

    return points

def main():
    nw = network.CustomNetwork(layers=(3, 64, 64, 64, 1),
                               activation=network.tanh)

    circle = generate_points_on_sinus(100, 1)
    m = len(circle)

    i = 0
    buff = []
    learning_rate = 0.1
    final_learning_rate = 0.01
    while True:
        i += 1

        idx_1 = (i + 0) % m
        idx_2 = (i + 1) % m
        idx_3 = (i + 2) % m
        idx_4 = (i + 3) % m

        data_in = [circle[idx_1][1],
                   circle[idx_2][1],
                   circle[idx_3][1]]
        data_out = [circle[idx_4][1]]

        o = nw.train_one((data_in, data_out), learning_rate)
        buff.append((circle[idx_4][0], o))
        if len(buff) > 100:
            buff.pop(0)

        if i % 1000 == 0:
            learning_rate *= 0.99

        if i % 100 == 0:
            print(i, nw.calculate_error(data_out), learning_rate)
            draw_points(buff)

        if learning_rate < final_learning_rate:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            exit()

if __name__ == '__main__':
    main()