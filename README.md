# Self Driving Car Simulator

## Proposition


**Self Driving Car Simulator** is a deep learning neural network project with a mission to emulate human driving behaivior within a simulator
Using collected driving data from the [Udacity Self-Driving Simulator](https://github.com/udacity/self-driving-car-sim), a vehicle is trained 
to autonomously steer itself based on a set of input images.

## Data Collection

- Training data was collected by driving five laps around the simulator track while recording 
    images as well as steering angle.

- The images were taken from three camera angles (center, left, right) at a rate of 
    14 images per second. 


![Alt Text](ezgif.com-video-to-gif.gif)



## EDA

### Steering

- Steering angle was normalized to the range of (-1,1)
- Steering angle data resembles disproportionate data at angle 0 with next most frequent angles being -1 and 1
- Training data for steering was done using a keyboard which likely caused abrupt changes in angles. If data was collected on joystick, t
    data would transition more smoothly from 0.

<img src="figures/Distr_Steering.png" alt="alt text" width=400 height=300>

### Image Processing

- Images collected were resized and reshaped to a standard size of 66 x 200 pixels with 3 channels
- Images converted from rgb to yuv for better processing in neural network
- For each data point image from center, left or right was randomly chosen and augmented using random flip
    and random translate to gain more usage from data collected

<img src="figures/Image_comparison.png" alt="alt text" width=400 height=300>

## Running the code
**All code is stored in src directory**
1.  **Initialization** 



## Plots


### GME
<img src="figures/GME_Mentions_Price.png" alt="alt text" width=400 height=300>

## Results

###  Running The One Way Anova Test with gradient of daily number of stock mentions, daily change in stock price



## Moving Forward

## Technologies Used
* Matplotlib
* Pandas


### Citations:
