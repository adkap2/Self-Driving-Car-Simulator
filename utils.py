import cv2, os
import numpy as np
import matplotlib.image as mpimp

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def load_image(data_dir, image_file):

    return mpimp.imread(os.path.join(data_dir, image_file.strip()))

def crop(image):

    return image[60:-25, :, :]

def resize(image):
    
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):

    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess(image):

    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

def choose_image(data_dir, center, left, right, steering_angle):

    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle

def random_flip(image, steering_angle):

    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

# def random_translate(image, steering_angle, range_x, range_y):

#     x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
#     x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
#     xm, ym = np.mgrid[0:IMAGE_HEIGHT. 0:IMAGE_WIDTH]

def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):

    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)

    return image, steering_angle

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):

    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)

    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]

            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augment(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
            images[i] = preprocess(image)
            i+=1
            if i == batch_size:
                break
            yield images, steers

