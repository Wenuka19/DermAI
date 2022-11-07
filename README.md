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
| Week 02 & 03| * Obtain all the required components<br>* Assemble the system and test for proper functionality<br>* Develop a GUI for Raspberry PI to capture the images and to do the preditions|
| Week 04 | * Exploring about CNN models available<br>* Finalize the set of steps to follow when training the model<br>|
| Week 05 | * Set up a full training + evaluation skeleton and gain trust in its correctness<br>|
| Week 06 | * Finalize the dataset<br>* Verify accuracy, loss and other metrics for the particular dataset<br>|
| Week 07 & 08 | * Train a vgg-16 model for the finalized dataset<br>* Fine-tune the model<br>|

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
Since there's no real skin disease images avaiable to capture I used a google image pointed at the camera for demonstration. Now in the coming weeks I can fine tune the Tensorflow model to get a higher accuracy.

[Week02.webm](https://user-images.githubusercontent.com/89344987/194615943-785368d4-1ea5-48d2-aee0-7a020114649e.webm)

### Week 04 (04<sup>th</sup> October to 11<sup>th</sup> October)
This week was totally dedicated to make the design decisions of the deep learning model of my project. The main decision I had to take was the architecture of the model. And the most recommended approach for this was to read some research papers in the same problem domain and start with a model architecture implemented by them. I came across these research papers and decided to analyze them to make this design decision. 
* Vipin Venugopal, Justin Joseph, M. Vipin Das, Malaya Kumar Nath,
An EfficientNet-based modified sigmoid transform for enhancing dermatological macro-images of melanoma and nevi skin lesions,
Computer Methods and Programs in Biomedicine, Volume 222, 2022, 106935, ISSN 0169-2607, https://doi.org/10.1016/j.cmpb.2022.106935.
* [Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset](https://arxiv.org/pdf/2104.09957v1.pdf)
* [Dermatologist-level classification of skin cancer with deep neural networks](https://www.nature.com/articles/nature21056.epdf?author_access_token=8oxIcYWf5UNrNpHsUHd2StRgN0jAjWel9jnR3ZoTv0NXpMHRAJy8Qn10ys2O4tuPakXos4UhQAFZ750CsBNMMsISFHIKinKDMKjShCpHIlYPYUHhNzkn6pSnOCt0Ftf6)

These papers have implemented **VGG-16, EfficientNet, InceptionV3** models. Also a popular CNN model used in embedded devices is MobileNet architecture. So I decided to use these models and select the best one for my use case.
Another problem I had was deciding the next steps in the training process. As I found out from the internet, there's no straightforward set of steps we should follow when training a ML model. But I came across [a very informative article](http://karpathy.github.io/2019/04/25/recipe/) written by an expert in deep learning, explaining his experience and the method he use when training a model. I decided to refer to his method when training my model.

### Week 05 (11<sup>th</sup> October to 18<sup>th</sup> October)
From this week onwards my main focus is to train and fine tune the model on PC for the datasets available. As I read in the tutorial article, the first step is to set up a full training + evaluation skeleton and gain trust in its correctness. For this I use a simple toy model. 
And I verifed that the code I wrote to load, input, normalize, train, and predict the model are correct. Verification of the inputs was done by vizualizing the image data right before the input to the model. 
After that I overfitted the model to training data, to make sure that I can obtain the maxiumum attainable accuracy for this dataset. After that I did some predictions to verify that the accuracy value aligns with the predictions. Further I visualized my predictions using a Confusion matrix to identify which labels are misclassified. By the end of the 5<sup>th</sup> week I was able to verify my training and evaluation skeleton.

### Week 06 (18<sup>th</sup> October to 25<sup>th</sup> October)
Now that the training, evaluating skeleton is finalized I tried to finalize a dataset to train my model. So far I was able to find 3 datasets. Below is a small summary about each dataset.

| Dataset 01 [Dermnet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) | Dataset 02 [DDI Dataset](https://ddi-dataset.github.io/) | Dataset 03 [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) |
|:---:|:---:|:---:|
| Contains around 19,500, out of which approximately 15,500 have been split in the training set and the remaining in the test set. | Containts around 656 images representing 570 unique patients. | Contains 16,577 clinical images |
| Data consists of images of 23 types of skin diseases | Data consists of images of 2 main skin conditions(benign  & malignant) | Data consists of 114 skin conditions with at least 53 images per skin condition |
| Dataset does not contain any skin condition label | Dataset contains skin condition labels and skin type labels based on the [Fitzpatrick](https://dermnetnz.org/topics/skin-phototype) scoring system | Dataset contains skin condition labels and skin type labels based on the [Fitzpatrick](https://dermnetnz.org/topics/skin-phototype) scoring system |
<br>
So this shows that inorder to build a multiclass classifier which tries to overcome the skin tone bias in predictions, I will have to use the Fitzpatrick17k dataset. 
I had to spend the rest of the week to sort out the dataset and train it for a simple model to verify that accuracy, loss metrics are correct for this particular dataset.

### Week 07 & 08 (25<sup>th</sup> October to 8<sup>th</sup> November)

Now I had to decide model to use for transfer learning this particular dataset. 
The [research](https://arxiv.org/pdf/2104.09957v1.pdf) paper which used the Fitzpatrick17k dataset suggests a VGG-16 model to train. So I decided to use that model as a starting process for my implementation aswell.
The model specified in that research paper the last fully connected 1000 unit layer with the following sequence of layers: a fully connected 256 unit layer, a ReLU layer, dropout layer with a 40% change of dropping, a layer with the number of predicted categories, and finally a softmax layer. As a result, the model has *135,338,674 parameters* of which *1,078,130 are trainable*.[Figure of the model](/assets/images/vgg_model1.png)<br>
I used 90% of the dataset for training, 5% for validation and 5% for testing. After training for 25 epochs I was able to obtain following results.<br>
* Training Accuracy - 40.19%
* Validation Accuracy - 33.62%
* Test Set Accuracy - 29%<br>
#### Training Graphs and the Heatmaps for the first attempt 
<div align = "center">
<p float="middle">
  <img src="/assets/images/vgg_model1_graphs.png" width="400" />
  <img src="/assets/images/vgg_model1_heatmap.png" width="400" /> 
</p>
</div>

To save and load models, weights and other details I use [Wandb](https://wandb.ai) platform. This can also be used to visualize and compare training between different models.
