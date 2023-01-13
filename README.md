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
| Week 09 | * Fixed the errors in the training process<br>|
| Week 10 | * Fine tuning the model<br>|
| Week 11 | * Obtain the maximum possible accuracy<br>|

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

### Week 09 (08<sup>th</sup> November to 15<sup>th</sup> November)
As I was training the model I noticed that the accuracy of the model keeps increasing in an unusual manner. But the predictions of it to external images was not accurate. This was because when I was creating the datasets from the image directory, it kept getting different sets of images for test,validation and train datasets. As a result I have trained the model to the whole dataset. This lead to the model overfitting to the whole dataset. I used the following method to create datasets from the image directory.

```
#Load data as numpy arrays
train_data = tf.keras.utils.image_dataset_from_directory(train_data_dir,label_mode='categorical',  image_size=IMG_SIZE,batch_size=2,)
```
So op the 09<sup>th</sup> week I had to redo all the work I did on the previous week. This time I uploaded the dataset in 3 separate folders called train,test and validation. This way there won't be a mixup between the datasets.
```
#Load data as numpy arrays
train_data = tf.keras.utils.image_dataset_from_directory(train_data_dir,label_mode='categorical',  image_size=IMG_SIZE,batch_size=2,)
val_data = tf.keras.utils.image_dataset_from_directory(val_data_dir,label_mode='categorical',  image_size=IMG_SIZE,batch_size=2,)
test_data = tf.keras.utils.image_dataset_from_directory(test_data_dir,label_mode='categorical',  image_size=IMG_SIZE,batch_size=2,)
```
And then I retrained the same model to obtain an accuracy value close to what I got on the previous week.

### Week 10 (15<sup>th</sup> November to 22<sup>nd</sup> November)
After getting a considerable accuracy for the base model, I tried to do some fine-tuning to the model and improve the accuracy. First I unfroze the last 15 layers of the model and retrained the model with a learning rate of 10<sup>-4</sup>. This improved the accuracy of the slightly. The values I obtained are as follows.
* Training Accuracy - 43%
* Validation Accuracy - 35%<br>

The other adjustments I tried were unfreezing more layers, adjusting the learning rate, changing the architecture of the model and trying a different optimizer. But none of them seem to improve the accuracy more.

### Week 11 (22<sup>nd</sup> November to 29<sup>th</sup> November)
Unfortunately, my Raspberry PI malfunctioned. It doesn't bootup when it is powered up. I tried several debugging methods methods suggested in online forums but none of them seem to work for me. As it is suggested in https://forums.raspberrypi.com/viewtopic.php?t=58151 the main indicator that there exists an issue is that the ACT LED does not repeatedly blink after the Rpi is powered up. According to the online forums this is happening when the RPI is unable to read from the SD card. I checked it with a different SD card but that also didn't seem to work. I cannot recall anything I did out of ordinary which could cause the RPI to malfunction. There were no physical damages to the board aswell. After many attempts, I decided to look for another Rpi board. And since the end semester exams were coming up I had to pause the project work for some time. 

#### (30<sup>th</sup> November to 18<sup>th</sup> December)
There were no significant progress in the project work due to end-semester examination. I was still looking for an alternative Raspberry PI and was also trying a few debugging methods.


### Week 12 (19<sup>th</sup> December to 26<sup>th</sup> December)
I was able to find a Raspberry Pi model 3B and I tried to setup my project in that. But it did not support the camera I already had. The camera interface is not shown in the Raspberry Pi configuration menu. It doesn't recognize the camera when connected. At this point I decided not to use a Rasberry Pi. Instead I decided to develop an android app which can be used as an image classifier. This meant there would be some restrictions in the computing power but since almost every smartphone in the market market today has a very quality camera I can use high quality images for predictions. Also the setup and demonstration will be much easier.
With the help of many online resources I was able to build the android app, put a multi class classification model and do some predictions.

#### Demo(As of 26<sup>th</sup> December)
The app looks like below in the emulator. It displays the highest confident result along with the percentage. Note that the model used in this is a custom model used for different use case.

[Mobile_App.mp4](https://user-images.githubusercontent.com/89344987/210223484-44402679-ed00-46f1-bd63-ace5d8d21cf4.mp4)

### Week 13 (26<sup>th</sup> December to 2<sup>nd</sup> January)
Now that the mobile app is finalized I tried to convert the vgg_16 model I trained earlier to a `.tflite` model. This caused some unexpected issues. When I run the code snippet which converts the model to a `.tflite` model the colab runs out of RAM. I was using the free version of colab which gives 12GB memory. I looked for solutions in online forums and tried limiting the memory growth, converting using a saved model, converting after freeing up the memory by deleting some data and reducing the batch size as mentioned in the given forums. [solution 01](https://github.com/tensorflow/models/issues/1817), [solution 02](https://github.com/tensorflow/tensorflow/issues/40760). But none of them seem to work. Also I was not able to find a project done by converting a vgg_16 model to a `.tflite` model. And it was mentioned in the [official documentation of TensorFlow lite](https://www.tensorflow.org/lite/guide/ops_compatibility) that certain types of models cannot be converted to `.tflite` format. So I decided that this model may not be usable in my project. Then I tried using a [MobileNetV3](https://paperswithcode.com/method/mobilenetv3) and a [InceptionV3](https://keras.io/api/applications/inceptionv3/) and I was able to convert both of them to `.tflite` models. Out of the two InceptionV3 started to overfit very quickly. The complexity of the model architecture and the lack of data in the dataset wee the main reasons for this. But the MobileNet model didn't overfit as quickly as the Inception model. 
So I decided to go ahead with that architecture. Since I used transfer learning method to train my model, I replaced the last layer of the InceptionV3 model with the following layers.
* Dense Layer with 1024 nodes.<br>
* Dense Layer with `no. of classes` nodes.<br>

As I found in various online resources `Adam` optimizer with the learning rate of 0.0001 seem to give very good results for the `MobileNet` model. So I decided to use the same optimizer for my use case. [Figure of the model](/assets/images/InceptionV3_model.png)<br>

Right before the val_loss start to increase when training, I was able to obtain the following accuracy values for the MobileNet model.
* Training Accuracy - 95.02%
* Validation Accuracy - 37%
* Test Set Accuracy - 39%<br>

### Week 14 (02<sup>nd</sup> January to 09<sup>th</sup> January)
Even after converting the model to `.tflie` I was able to obtain same accuracy, f1-score values for the test dataset. This verified that the conversion process doesn't reduce accuracy.
This week and the rest of the time will be mainly focused towards the generalizing the model to avoid overfitting. As I found in many online resources the first step to reduce overfitting is to get more data. But in my case it is a bit difficult to get more data from a reliable resource. 
When I observed the number of samples per class I saw that the dataset is heavily unbalanced.
This is shown in the graph below.
![train_dataset](https://user-images.githubusercontent.com/89344987/211035094-8aa24cde-01fd-412b-a3b1-021f0354a7be.png)
To overcome this imbalance I tried implementing [Data Augmentation.](https://www.analyticsvidhya.com/blog/2021/05/image-classification-with-tensorflow-data-augmentation-on-streaming-data-part-2/#:~:text=What%20is%20Data%20Augmentation%3F,Image%20flip%2C%20and%20many%20more.). I used the Tensorflow's [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) API to do image augmentations while training.
I applied the following augmentations to the datasets.
```
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip = True,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rotation_range=15,
    fill_mode='nearest',
    zoom_range=1.6,
    brightness_range = [1,1.5] 
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input   # InceptionV3 image preprocessing
)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input   # InceptionV3 image preprocessing
)
```
Then I oversampled the classes with few number of images to get an average of about 180 images per class. After training for this dataset, I got the following accuracy values.
* Training Accuracy - 93%
* Validation Accuracy - 40%
* Precision - 0.41
* Recall - 0.41
* F1 Score - 0.40<br> 

Then I tried to implement [class_weights](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) when training the model. This uses a weight value when calculating the loss function. This weight value is calculated according to the number of samples per class so that the loss function gets optimized more for data with low no. of samples per class. This didn't seem to improve the accuracy of the model. So I decided to go finalize this model.<br>
Even though this accuracy may seem pretty low, average accuracy of a dermalogists' prediction for a disease just by visual inspection is also about 65%. So I think my model performs moderately well for the dataset.

#### Final Demonstration
[Final_Demo.mp4](https://user-images.githubusercontent.com/89344987/212349741-c80d3bd2-94c7-421c-ad6a-b6030a8ffee6.mp4)

