# **Traffic Sign Recognition** 

## Introduction

In this project, we will use deep neural networks and convolutional neural networks to classify traffic signs. We will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, we will try out your model on images of German traffic signs found on the web.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization1.png "Visualization: Plotting the count of each sign"
[image2]: ./examples/visualization2.png "Visualization: Plotting distribution of classes in the training and validation sets"
[image3]: ./examples/rgb2yuv.png "RGB to YUV Conversion"
[image4]: ./data/Sign1.tiff "Traffic Sign 1"
[image5]: ./data/Sign2.tiff "Traffic Sign 2"
[image6]: ./data/Sign3.tiff "Traffic Sign 3"
[image7]: ./data/Sign4.tiff "Traffic Sign 4"
[image8]: ./data/Sign5.tiff "Traffic Sign 5"
[image9]: ./data/Sign6.tiff "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sameha/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 samples
* The size of the validation set is 4,410 samples
* The size of test set is 12,630 samples
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43 classes, 0 -> 42

#### 2. Include an exploratory visualization of the dataset.

For exploratory visualization of the data set. We plotted the count of each sign as well as the distribution of classes in the training and validation sets.

##### **Plotting the count of each sign**

The following graph shows the count for each sign in the German Traffic Signs Database. It clearly shows that the distribution for each class is not equal, and that some classes (like class #1 or #2) are more than **ten times** the frequency of other classes (like class #0 or #19).

![alt text][image1]

##### **Plotting distribution of classes in the training and validation sets**

The following graph shows the normalized histograms for the training, validation, and test sets as well as all data combined. As shown from before, it is clear that the frequency of each class is not equal to other classes.

![alt text][image2]

In addition, the graph shows that the training, validation, and test sets have been properly created, as their distribution looks similar.


### Design and Test a Model Architecture

#### 1. Data Preprocessing

As a first step, I decided to convert the images to the YUV scale as it has shown to produce better results than RGB in my experiments, as well as in a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

Here is an example of four traffic sign images before and after converting.

![alt text][image3]

As a last step, I normalized the image data because having features within -1 --> 1 range produces better results for deep learning.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 14x14x6 				    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 5x5x6 				    |
| Flatten   	      	| Input = 5x5x16, outputs = 400  			    |
| Fully connected		| Fully Connected. input = 400, outputs = 120	|
| RELU					|												|
| Fully connected		| Fully Connected. input = 120, outputs = 84	|
| RELU					|												|
| Fully connected		| Fully Connected. input = 84, outputs = 43 	|
| Softmax				|           									|
|						|												|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters:

* Learning Rate: 0.001
* Optimizer: Adam Optimizer
* Loss: Cross Entropy
* Number of Epochs = 979 (Found from training runs)
* Batch Size = 128

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* training set accuracy of: 100.0%
* validation set accuracy of: 95.62%
* test set accuracy of: 93.17%

The archicteure used was the well known LeNet CNN architecutre, it has been shown to produce excellent results on image recognition problems. My results of 100% on the train set, 95.62% on the validation set, and 93.17% on the test set provide evidence that the model is working well.
 

### Test a Model on New Images

#### 1. Choose fsix German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

The images have different sizes and resolution, the first image might be difficult to classify because of its bad resolution, the second image might be difficult to classify because of a weatermark on the image and because its sign (No Passing) is similair to another sign (End of No Passing).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work      		| Road Work 									| 
| No Passing   			| End of No Passing     						|
| Yield					| Yield											|
| Wild Animals Crossing	| Wild Animals Crossing			 				|
| Stop		            | Stop  		    							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This compares favorably to the accuracy on the test set of 93.17%

#### 3. Describe how certain the model is when predicting on each of the six new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

