#**Traffic Sign Recognition** 

##Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./images/image1.png "Traffic Sign 1"
[image5]: ./images/image2.png "Traffic Sign 2"
[image6]: ./images/image3.png "Traffic Sign 3"
[image7]: ./images/image4.png "Traffic Sign 4"
[image8]: ./images/image5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the gray image can ignore the more additional infomation of the image.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the value of pixel is 0-255 and the shape is 32x32x3. The normalization operation is more efficiant one in image data handling. Because it only handles the samples to the range of 0-1.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Flatten					|		outputs 400										|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 1x1x400     									|
| RELU					|										|
| Flatten					|		outputs 400	|
| Concat					|		outputs 800	|
| Dropout					|		|
| Fully connected		| outputs 43     |
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an CNN architecture. The best value of epochs I tested is about 50-70, so I choose 64. And the best batch size is 128 after tested the different values. For getting the better prediction of image, I decided the value of learning rate to be 0.0009.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.934
* test set accuracy of 0.922

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen? Maybe I will choose the LeNet-5 because of its accuracy.
* What were some problems with the initial architecture? For training dataset it maybe can got a high accuracy but act not well for vaild dataset.
* How was the architecture adjusted and why was it adjusted? There are some methods for this, like 1.convert the RGB image to gray image; 2. normalized the image; 3. apply the new LeNet architecture from [pdf](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf); 4. adjust the learning-rate and batch-size for getting high performance
* Which parameters were tuned? How were they adjusted and why? The parameters that should be tuned is like epochs, size of batch, learning rate, dropout value.
* What are some of the important design choices and why were they chosen? The layer of CNN is important, we should adjust them for participate situation. Of course, the other method like dropout should be applied for big dataset and it will perform well for big dataset.

If a well known architecture was chosen:

* What architecture was chosen?  LeNet
* Why did you believe it would be relevant to the traffic sign application? For classifing model, it can perform well like human.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? For test dataset, we got the accuracy of 0.922. It is high for prediction of image.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 
![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

the first and fourth maybe difficult because they have some other words on the images。
####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)      		| Speed limit (50km/h)   | 
| Stop     			| Stop   |
| General caution					| General caution											|
| Road work	      		| Road work					 				|
| Turn left ahead			| Turn left ahead      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image,  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.071743         			| End of speed limit (80km/h)   									| 
| 0.05285687     				| Slippery road 										|
| 0.05154734					| Keep left											|
| 0.05106123	      			| Speed limit (50km/h)					 				|
| 0.0465317				    | No passing for vehicles over 3.5 metric tons      							|


For the second image,  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.10543087         			| Priority road   									| 
| 0.10525934     				| End of speed limit (80km/h) 										|
| 0.08112243					| Keep left											|
| 0.05417077	      			| Slippery road					 				|
| 0.04826771				    | Speed limit (80km/h)      							|

For the third image,  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.06330259         			| Priority road   									| 
| 0.060952     				| Speed limit (50km/h) 										|
| 0.0591356					| Slippery road											|
| 0.05471617	      			| Go straight or left					 				|
| 0.05259452				    | Ahead only      							|

For the fourth image,  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.07733472         			| Speed limit (50km/h)   									| 
| 0.06139544     				| Slippery road 										|
| 0.05471208					| End of speed limit (80km/h)										|
| 0.04894884	      			| No passing					 				|
| 0.04378645				    | Speed limit (100km/h)      							|

For the fifth image,  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.08724929         			| End of all speed and passing limits   									| 
| 0.08393034     				| Go straight or left 										|
| 0.05755705					| Ahead only										|
| 0.05325893	      			| Yield					 				|
| 0.04107359				    | Keep left     							|

the  images is almost the same ，uncertain predict those images.s 

