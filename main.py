import numpy as np
import cv2
import network

# https://github.com/niderhoff/nlp-datasets
# https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/


img = None

def draw_clear():
    global img
    img = np.zeros((500, 500, 3), dtype=np.uint8)

def draw_points(points, clr, margin=0.1):
    global img
    # Create a blank 500x500 black image
    if img is None:
        img = np.zeros((500, 500, 3), dtype=np.uint8)

    # Convert points from range [-1-margin, 1+margin] to [0, 500]
    points = [(int(((x+1)*(1-margin)+margin)*250), int(((y+1)*(1-margin)+margin)*250)) for x, y in points]

    # Draw each point
    for x, y in points:
        cv2.circle(img, (x, y), radius=5, color=clr, thickness=-1)

    # Display the image
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
    nw = network.CustomNetwork(layers=(1, 20, 20, 20, 1),
                               activation=network.tanh,
                               magic=12)

    circle = generate_points_on_sinus(50, 1)[:-1]
    m = len(circle)

    i = 0
    o = [0]
    buff = []
    learning_rate = 0.1
    final_learning_rate = 0.0001
    while True:
        i += 1

        idx_1 = (i + 0) % m
        idx_2 = (i + 1) % m

        data_in = [circle[idx_1][1]]
        data_out = [circle[idx_2][1]]

        o = nw.train_one((data_in, data_out), learning_rate)
        buff.append((circle[idx_2][0], o))
        if len(buff) > m:
            buff.pop(0)

        if i % 1000 == 0:
            learning_rate *= 0.99

        if i % 1000 == 0:
            print(i, nw.calculate_error(data_out), learning_rate)
            draw_clear()
            draw_points(circle, (255, 0, 0))
            draw_points(buff, (0, 255, 0))
            cv2.imshow('Image', img)
            cv2.waitKey(1)

        if learning_rate < final_learning_rate:
            cv2.waitKey(0)
            break

    buff = []
    i = 0
    while True:
        i += 1

        idx_1 = (i + 0) % m
        idx_2 = (i + 1) % m

        data_in = o
        o = nw.forward(data_in)

        buff.append((circle[idx_2][0], o))
        if len(buff) > m:
            buff.pop(0)

        if i % m == 0:
            print(i)
            draw_clear()
            draw_points(buff, (255, 255, 255))
            cv2.imshow('Image', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()