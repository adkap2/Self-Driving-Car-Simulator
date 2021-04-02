import cv2, os
import numpy as np
import matplotlib.image as mpimp

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def load_image(data_dir, image_file):
    """Loads image in from directory where images are stored"""

    return mpimp.imread(os.path.join(data_dir, image_file.strip()))

def crop(image):
    """Crop shrink height dimension down to consistent size"""
    return image[60:-25, :, :]

def resize(image):
    """Resize image to consient height, width to be readable"""
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
    """Convert image from RGB to Y U V channels 
    Luminance, chroma blue and chroma red for readability
    Better to train model than standard human perception colors"""

    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess(image):
    """Calls processing functions and returns a
    processed image"""

    image = crop(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = resize(image)
    image = rgb2yuv(image)
    return image

def choose_image(data_dir, center, left, right, steering_angle):
    """ Random picks an image to be either center, left or right
    from given point in time"""

    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle

def random_flip(image, steering_angle):
    """ Randomly flips image 
    to extra more information from data set"""
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """ Randomly translates image to extract additional information
    from dataset
    returns and image translatation as well as an updated steering angle"""

    # x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    # x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    # xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # Store height and width of the image
    transx, transy = range_x * (np.random.rand() - 0.5), range_y 
    * (np.random.rand() - 0.5)
    steering_angle += transx * 0.002
    height, width = image.shape[:2]
    #quarter_height, quarter_width = height / 4, width / 4
    T = np.float32([[1, 0, transx], [0, 1, transy]])
    # We use warpAffine to transform
    # the image using the matrix, T
    img_translation = cv2.warpAffine(image, T, (width, height))
    return img_translation, steering_angle


def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """ Augment function takes passed in image and does random
     shifts and flips on image to gain more usable data"""
    image, steering_angle = choose_image(data_dir, center, left,
     right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle,
     range_x, range_y)

    return image, steering_angle

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):

    #batch_size = image_paths.shape[0] # Batchsize size of sample THIS IS NOT SPLITTING INTO TRAINING/TEST

    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)

    i = 0
    for index in np.random.permutation(batch_size):
        center, left, right = image_paths[index]
        steering_angle = steering_angles[index]

        if is_training and np.random.rand() < 0.6:
            image, steering_angle = augment(data_dir, center, left, right, steering_angle)
        else:
            image = load_image(data_dir, center)
        images[i] = preprocess(image)
        steers[i] = steering_angle
        i += 1
        yield images , steers

    #return images, steers

