<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Classification of Flowers using VGG19</div>
<div align="center"><img src="https://github.com/Pradnya1208/Classification-of-flowers-using-VGG19/blob/main/output/intro2.gif?raw=true"></div>


## Overview:
VGG-19 is a convolutional neural network that is 19 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database [1]. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224.
<br>Through this project we'll understand and built a classification model for flower's dataset.

## Dataset:
[Flower Recognition](https://www.kaggle.com/alxmamaev/flowers-recognition)
This dataset contains 4242 images of flowers.
The data collection is based on the data flicr, google images, yandex images.
<br>
The pictures are divided into five classes: chamomile, tulip, rose, sunflower, dandelion.
For each class there are about 800 photos. Photos are not high resolution, about 320x240 pixels. Photos are not reduced to a single size, they have different proportions


## Implementation:

**Libraries:**  `NumPy` `pandas` `sklearn` `tensorflow` `seaborn` `keras`
## Data Exploration:
<img src="https://github.com/Pradnya1208/Classification-of-flowers-using-VGG19/blob/main/output/count.PNG?raw=true">
<br>
<img src="https://github.com/Pradnya1208/Classification-of-flowers-using-VGG19/blob/main/output/pic.PNG?raw=true">
<br>

### Convolutional Neural Networks
A Convolutional Neural Network is a special type of an Artificial Intelligence implementation which uses a special mathematical matrix manipulation called the convolution operation to process data from the images.
- A convolution does this by multiplying two matrices and yielding a third, smaller matrix.
- The Network takes an input image, and uses a filter (or kernel) to create a feature map describing the image.
- In the convolution operation, we take a filter (usually 2x2 or 3x3 matrix ) and slide it over the image matrix. The coresponding numbers in both matrices are multiplied and and added to yield a single number describing that input space. This process is repeated all over the image.
<br>
<img src="https://github.com/Pradnya1208/Classification-of-flowers-using-VGG19/blob/main/output/cnn1.gif?raw=true">
<img src="https://github.com/Pradnya1208/Classification-of-flowers-using-VGG19/blob/main/output/2d-3d.PNG?raw=true">
<br>
We use different filters to pass over our inputs, and take all the feature maps, put them together as the final output of the convolutional layer.
We then pass the output of this layer through a non-linear activation function. The most commonly used one is ReLU.
The next step of our process involves further reducing the dimensionality of the data which will lower the computation power required for training this model. This is achieved by using a Pooling Layer. The most commonly used one is max pooling which takes the maximum value in the window created by a filter. This significantly reduces the training time and preserves significant information.
<br>
<img src="https://github.com/Pradnya1208/Classification-of-flowers-using-VGG19/blob/main/output/pooling.PNG?raw=true">
<br>

#### STRIDE: 

Stride just means the amount a filter moves during a covolution operation. So, a stride of 1 means that the filter will slide 1 pixel after each covolution operation.<br>
<img src="https://github.com/Pradnya1208/Classification-of-flowers-using-VGG19/blob/main/output/cnn2.gif?raw=true">
<br>

#### PADDING: 
Padding is just zero value pixels that surround the input image. This protects the loss of any valuable information since the feature map is ever shrinking.<br>
### Training the Model
```
with strategy.scope():
    pre_trained_model = VGG19(input_shape=(224,224,3), include_top=False, weights="imagenet")

    for layer in pre_trained_model.layers[:19]:
        layer.trainable = False

    model = Sequential([
        pre_trained_model,
        MaxPool2D((2,2) , strides = 2),
        Flatten(),
        Dense(5 , activation='softmax')])
    model.compile(optimizer = "adam" , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg19 (Model)                (None, 7, 7, 512)         20024384  
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 3, 3, 512)         0         
_________________________________________________________________
flatten_5 (Flatten)          (None, 4608)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 5)                 23045     
=================================================================
Total params: 20,047,429
Trainable params: 4,742,661
Non-trainable params: 15,304,768
_________________________________________________________________
```

```
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
history = model.fit(x_train,y_train, batch_size = 64 , epochs = 12 , validation_data = (x_test, y_test),callbacks = [learning_rate_reduction])
```
```
Accuracy of the model is -  88.09248208999634 %
```

### Analysis of the Model:

<img src="https://github.com/Pradnya1208/Classification-of-flowers-using-VGG19/blob/main/output/modelanalysis.PNG?raw=true">
<br>

```
              precision    recall  f1-score   support

   dandelion       0.92      0.91      0.92       210
       daisy       0.92      0.87      0.89       154
       tulip       0.83      0.87      0.85       197
   sunflower       0.88      0.95      0.91       147
        rose       0.86      0.80      0.83       157

    accuracy                           0.88       865
   macro avg       0.88      0.88      0.88       865
weighted avg       0.88      0.88      0.88       865
```
<br>
<img src="https://github.com/Pradnya1208/Classification-of-flowers-using-VGG19/blob/main/output/cm.PNG?raw=true">
<br>

### Correctly classified images:
<img src="https://github.com/Pradnya1208/Classification-of-flowers-using-VGG19/blob/main/output/results.PNG?raw=true">
<br>

### Incorrectly classified images:
<img src="https://github.com/Pradnya1208/Classification-of-flowers-using-VGG19/blob/main/output/incorrect.PNG?raw=true">
<br>

### Learnings:
`Convolutional Neural Networks`
`VGG19`






## References:
[VGG16 and VGG19](https://keras.io/api/applications/vgg/)

### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner

[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]


