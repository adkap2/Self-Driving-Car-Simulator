# Self Driving Car Simulator

## Proposition


**Self Driving Car Simulator** is a deep learning neural network project with a mission to emulate human driving behaivior within a simulator
Using collected driving data from the [Udacity Self-Driving Simulator](https://github.com/udacity/self-driving-car-sim), a vehicle is trained 
to autonomously steer itself based on a set of input images. The deep convolutional neural network used performs well with image regression and classification problems. The initial model was inspired by the [NVIDIA](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) Convolutional Neural Network implemented for their autonomous vehicle.

## Data Collection

- Training data was collected by driving five laps around the simulator track while recording 
    images as well as steering angle.

- The images were taken from three camera angles (center, left, right) at a rate of 
    14 images per second. 


![Alt Text](figures/ezgif.com-video-to-gif.gif)



## EDA

### Steering

- Steering angle was normalized to the range of (-1.5,1.5)
- Steering angle data resembles disproportionate data at angle 0 with next most frequent angles being -1.5 and 1.5
- Training data for steering was done using a keyboard which likely caused abrupt changes in angles. If data was collected on joystick, t
    data would transition more smoothly from 0.

<img src="figures/Distr_Steering.png" alt="alt text" width=400 height=300>

### Image Processing

- Images collected were resized and reshaped to a standard size of 66 x 200 pixels with 3 channels
- Images converted from rgb to yuv for better processing in neural network
- For each data point image from center, left or right was randomly chosen and augmented using random flip
    and random translate to gain more usage from data collected
- Image processing took approximately 27,000 data points and converted it to 9000 data points
- After completing an 80% train test split, training data contained 7396 trainable images

<img src="figures/Image_comparison.png" alt="alt text" width=600 height=300>

## Running the code
**All code is stored in src directory**
- To train the model, simply run
    ```
    python model.py
    ```
- Custom hyperparameters may be specified in the main function of model.py

## Training

- Training the model required taking the processed camera angle and inputting it into a CNN where it was evaluated against the recorded steering angle
    at that given state to produce an error. This error was then placed back into the CNN with backpropagation weight adjustment. 

<img src="figures/block_diagram.png" alt="alt text" width=600 height=300>


- I used a convolulational 2d neural network to train the model. 
- Started with an initial layer of the size of the input shape, then slowly increased number of filters while decreasing kernal size.
- Model used multiple 50% dropout layers for regularization in between the dense layers.
- Full neural network yielded 250,000 parameters

<img src="figures/hyperparameters.png" alt="alt text" width=300 height=300>





## Results


## Moving Forward

## Technologies Used
* [Matplotlib](https://matplotlib.org)
* [Pandas](https://pandas.pydata.org)
* [Tensorflow Keras](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
* [Cv2](https://pypi.org/project/opencv-python/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [NumPy](https://numpy.org)
* [Autonomous Vehicle Simulator](https://github.com/udacity/self-driving-car-sim)


### Citations:
1. “End-to-End Deep Learning for Self-Driving Cars.” NVIDIA Developer Blog, 25 Aug. 2020, developer.nvidia.com/blog/deep-learning-self-driving-cars/.


2. Sagar, Abhinav. “How to Train Your Self Driving Car Using Deep Learning.” Medium, Towards Data Science, 26 Nov. 2019, towardsdatascience.com/how-to-train-your-self-driving-car-using-deep-learning-ce8ff76119cb.


3. Udacity. “Udacity/Self-Driving-Car-Sim.” GitHub, github.com/udacity/self-driving-car-sim. 
