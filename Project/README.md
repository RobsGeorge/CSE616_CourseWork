# ImageColorization-Deep-CNN
# Introduction:

In image colorization, our goal is to produce a colored image given a greyscale input image. The problem in this task is that a single grayscale image may correspond to many colored images. As a result, traditional models often relied on significant user input alongside a grayscale image. 
Recently, deep neural networks have shown remarkable success in automatic image colorization going from grayscale to color with no additional human input. This success may in part be due to their ability to capture and use semantic information in colorization.

This project implements a deep convolutional neural network for automatic colorization, the problem of converting grayscale input images into colored images. The model is based on the ResNet-18 classifier and trained on the MIT Places database of landscapes and scenes.

# The Problem:
We aim to infer a full-coloured image, which has 3 values per pixel (lightness, saturation, and hue), from a grayscale image, which has only 1 value per pixel (lightness only). For simplicity, we will only work with images of size 256×256, so our inputs are of size 256×256×1 (the lightness channel) and our outputs are of size 256×256×2 (the other two channels). 
Rather than work with images in the RGB format, as people usually do, we will work with them in the LAB colour space (Lightness, A, and B). This colour space contains exactly the same information as RGB, but it will make it easier for us to separate out the lightness channel from the other two (which we call A and B). We'll make a helper function to do this conversion.
We'll try to predict the colour values of the input image directly (we call this regression).

# Papers used:
Colorful Image Colorization - https://arxiv.org/abs/1603.08511
Let there be color! - https://www.semanticscholar.org/paper/Let-there-be-color!-Iizuka-Simo-Serra/ec3453b0892ed8bf0531ffb5370c0159597eec11
Also, I found this paper very useful to take its work into consideration: Combining Deep Convolutional Neural Networks with Markov Random Fields for Image Colorization - https://drive.google.com/file/d/12GfrYOoHUIyUd7gQlLRoe-DdkjhvMOId/view?usp=sharing

# Dataset:
We are going to use subset of MIT Places dataset (containing places, landscapes, and buildings). You can download this dataset from this link: http://data.csail.mit.edu/places/places205/testSetPlaces205_resize.tar.gz 

Dataset is split into 90% training, and 10% test (validation) ... 40k images for training and 1k for validation.

# Preparation and Steps of Work:
First, downloading the dataset.
Then, moving data into training and validation folders
using this code:
![image](https://user-images.githubusercontent.com/33100615/167317035-1fe5e919-5959-4741-bd1a-bbca16d6cbdb.png)

After that, downloading and importing libraries needed: 
![image](https://user-images.githubusercontent.com/33100615/167317066-a1e98e33-ccc5-404e-8a3b-9c0d74f32260.png)

# Tools:
We'll build and train our model with PyTorch. We'll also use torchvision, a helpful set of tools for working with images and videos in PyTorch, and scikit-learn for converting between RGB and LAB colorspces.

# Model:
More info about our model can be found in the PDF documentation. 
Here is a figure showing the architecture of the model:
![model](https://user-images.githubusercontent.com/33100615/167316005-35e05795-fca3-497d-ba87-7f62d3be8d3a.jpg)

Model is defined inside the code file called model.py

# Helper Functions: 
Before we train, we define helper functions for tracking the training loss and converting images back to RGB: def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None)

# Training:
We train our model using the train function: def train(train_loader, model, criterion, optimizer, epoch)

We trained our model on Google Co-lab session for 100 epochs.

![image](https://user-images.githubusercontent.com/33100615/167317185-63cd83bc-6dd7-4c07-93cd-f8611e10947a.png)

We define a training loop:

![image](https://user-images.githubusercontent.com/33100615/167317198-297ca951-048f-433c-a22b-0d991492b31a.png)

#Validation:
Using validate function: def validate(val_loader, model, criterion, save_images, epoch)

# Results:
We showed the results using the following code: 
![image](https://user-images.githubusercontent.com/33100615/167317277-ea7bd58f-9788-469c-9614-72ad4ad51799.png)

A sample image result found in the "outputs" directory.

Another results were shown in the documentation/report as follows:
![image](https://user-images.githubusercontent.com/33100615/167316119-e89343a2-f2f8-4c80-bf4b-06bc0d815f57.png)



