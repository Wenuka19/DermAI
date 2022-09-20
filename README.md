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
| Week 01| * Obtain all the required components<br>* Learn the fundamentals<br>* Create a GitHub Repository<br>|
| Week 02| * Assemble the system and test for proper functionality<br>* Find avaialable datasets and finalize a dataset for training<br>* Get the workflow sorted out for training a model, testing a model and loading it to the Raspberry Pi for testing<br>* Develop a GUI for Raspberry PI to capture the images and to do the preditions|
| Week 03,04 & 05| * Train several models on PC<br>* Load and test it on the Raspberry Pi<br>* Evaluate the performance<br>|
| Week 06| * Finalize a model for the system<br>* Fine tune it for the skin tone bias<br>* Improve the test set accuracy<br>|
| Week 07| * Integrate the system to be used as a standalone device by setting up a power unit and a display unit<br>* Test the system with real time data<br>|
| Remaining Weeks| * Test and further develop the model<br>* Catchup with any delays in prior weeks<br>|
