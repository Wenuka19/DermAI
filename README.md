# DermAI

## Introduction
 DermAI is a fully offline skin disease identifying device which will provide accurate predictions to users when they input an image of their affected area. This is built using a Raspberry PI computer<br>
 <img src="/assets/images/product_img.jpg" width="200">
 
_(source: [Microsoft Machine Learning Kit for Lobe](https://www.pi-shop.ch/microsoft-machine-learning-kit-for-lobe))_

### Problem Motivation
Skin diseases are reported to be the 4th leading cause of disability worldwide, but with the limited number and distribution of dermatologists, it is nearly impossible to provide adequate care to everyone in need using traditional methods.
Thus, the implementation of long- range technological support tools has been growing in an effort to provide quality dermatology care to even the most remote settings globally. DermAI is an effort to provide this long-range support for skin diseases.

### Main Components

**Raspberry Pi Camera** <br>
This is used to take a picture of the infected skin area and send it to the Raspberry pi computer.

**Raspberry Pi Computer** <br>
This unit contains a trained Machine Learning Model that can classify skin disease images. It will receive the image and display the identified result along with the reasoning for the prediction.

## Related Work
I was able to find following researches and studies related to my project.

* [Discriminative Feature Learning for Skin Disease Classification Using Deep Convolutional Neural Network](https://ieeexplore.ieee.org/document/9007648)<br>
  B. Ahmad, M. Usama, C. -M. Huang, K. Hwang, M. S. Hossain and G. Muhammad, "Discriminative Feature Learning for Skin Disease Classification Using Deep Convolutional   Neural Network," in IEEE Access, vol. 8, pp. 39025-39033, 2020, doi: 10.1109/ACCESS.2020.2975198.
* [Deep Learning and Machine Learning Techniques of Diagnosis Dermoscopy Images for Early Detection of Skin Diseases](https://www.mdpi.com/2079-9292/10/24/3158)<br>
  Abunadi I, Senan EM. Deep Learning and Machine Learning Techniques of Diagnosis Dermoscopy Images for Early Detection of Skin Diseases. Electronics. 2021;             10(24):3158. https://doi.org/10.3390/electronics10243158
* [A Smartphone-Based Skin Disease Classification Using MobileNet CNN](https://arxiv.org/abs/1911.07929)
* [Google's AI Tool (Not launched yet)](https://blog.google/technology/health/ai-dermatology-preview-io-2021/)
  

## Project Plan
Note that the following plan is fairly optimistic since there can be unexpected delays and changes to the initial plan. There are several weeks left towards the end to account for delays and provide time for thorough testing

| Week | Tasks |
|:---:|-----------|
| Week 01| * Learn the fundamentals<br>* Create a GitHub Repository<br>* Get the workflow sorted out for training a model, testing a model and loading it to the Raspberry Pi for testing<br> |
| Week 02| * Obtain all the required components<br>* Assemble the system and test for proper functionality<br>* Find avaialable datasets and finalize a dataset for training<br>* Develop a GUI for Raspberry PI to capture the images and to do the preditions|
| Week 03,04 & 05| * Train several models on PC<br>* Load and test it on the Raspberry Pi<br>* Evaluate the performance<br>|
| Week 06| * Finalize a model for the system<br>* Fine tune it for the skin tone bias<br>* Improve the test set accuracy<br>|
| Week 07| * Integrate the system to be used as a standalone device by setting up a power unit and a display unit<br>* Test the system with real time data<br>|
| Remaining Weeks| * Test and further develop the model<br>* Catchup with any delays in prior weeks<br>|

## Weekly Reports

### Week 01 (13<sup>th</sup> September to 20<sup>th</sup> September)
<p> I dedicated the first week to learn all the fundamentals required to start this project and to finalize the workflow of training a model, converting a model, loading it to Raspberry Pi and running inference on it.</p>

#### Key Learnings
* The Computer Science problem I'm trying to solve in this project is an image classification problem. To solve these problems there is a wide variety of techniques avaialable. Support Vector Machines(SVM), Convolution Neural Networks(CNN) & K-NN Classifiers are some of them. Out of them CNN seems to provide a very high accuracy. For my application the model needs to identify fine-grained patterns of the images. For these applications the convolution filters get fined tuned in the training process. So I decided to use a CNN for my application

* To implement the above technique Python language will be used since Python is the go-to language used for Machine Learning and Deep Learning. And there are machine learning frameworks avaialable which deals with the complex fundamentals and provides easy to use syntax for the programmer. PyTorch, Tensorflow & Keras are the most popular machine learning frameworks available for Python language. Out of these 3 PyTorch provides an easy to use, flexible and a native environment for the programmer. But Tensorflow seems to have comprehensive ecosystem of community resources, libraries and it has a wide range of tools which are useful for my application. Keras was adopted and integrated into TensorFlow in mid-2017. Also, almost all the resouces I found have used Tensorflow so I decided to use that framework.

* Developing a CNN from scratch for a complex image classification problem like this is difficult when you have a limited amount of computational resources and training data. In such cases a widely used technique is [Transfer Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/). In my project also I will use the transfer learning technique to train already available well-trained models for my application domain.

#### Finalized Workflow
##### Loading a pre trained model and Training it 
In order to learn and create a workflow for training and testing, I loaded a InceptionV3 model to [Kaggle](https://www.kaggle.com) and trained it for the [DermNet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) dataset using transfer learning. Here my main goal was to get familiarized with the syntax and process so I did not worry about the accuracy of the predictions.

##### Converting the model to Tensorflow Lite model
[Tensorflow Lite](https://www.tensorflow.org/lite) is a lightweight framework introduced by Tensforflow to deploy machine learning models on mobile, micrcontrollers and other edge devices. After building and training the Tensorflow model on Google Colab we can use the following code snippet to convert it into a Tensorflow Lite model.
```
 # convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tf_lite_model = converter.convert()

# save the model
tf_lite_model_file = pathlib.Path('tf_lite01.tflite')
tf_lite_model_file.write_bytes(tf_lite_model)
```
#### Running Inference on Raspberry Pi
After getting the Tensorflow Lite model I needed to find out how to load it into Raspberry Pi and run inference on it. For this I found [the following](https://www.tensorflow.org/lite/examples/image_classification/overview) resources. With this I was able to run inference on the Raspberry Pi.

### Week 02 & 03 (20<sup>th</sup> September to 4<sup>th</sup> October)

There was an unexpected delay in obtaining the required parts for my project. By the mid of 2<sup>nd</sup> week I was able to obtain the Raspberry Pi, Pi Camera and all the required cables. After assembling it took a while to setup the PiCamera because there were some packages missing in the system. After going through several online resources I was able to write a simple python programme that opens up the camera view finder and captures an image. 
The next challenge was to develop a GUI to do the predictions. Since I'm already using Python language to do the predictions I decided to use [Tkinter](https://docs.python.org/3/library/tkinter.html) which is a popular Python package for GUI development. It is very beginner friendly and provides only basic functionalities of a GUI. After making the initial frames I couldn't find a way to place the camera-view inside one window and let the user capture an image with a button click. Even in [Raspberry Pi camera documentation](https://www.raspberrypi.com/documentation/accessories/camera.html) they have not mentioned a way to place the camera view inside a Tkinter window. Finally I came across [this](https://pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/) tutorial which implements a workaround to do what I needed. Finally I was able to develop the GUI for the app. Then I used a simple CNN trained on the dataset and converted it into a Tensorflow lite model and tried to make predictions on the images captured by the Raspberry Pi camera. This shows the top 5 predictions along with the probabilities. Since the model is not properly tuned the predictions are not accurate but now the skeleton of my project is complete.

#### Demo(As of 4<sup>th</sup> October)
