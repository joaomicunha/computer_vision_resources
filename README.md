# **Computer Vision**

This repo documents my journey from being a Data Scientist with basic knowledge of image data and cv techniques, to hopefully build up more expertise. I have some experience with images related projects and have studied Deep Learning through courses and literature over the last year and a half.

On the 2nd of December 2019 I started this learning process. The progress will be:

- Quickly review Andrew's Ng Convolutional Neural Networks chapter to put my self in the right mindset and vocabulary space. 
- Complete PyTorch course (ideally with Computer Vision orientation). During the PyData Conference in London 2019, I had the opportunity to attend multiple presentations using PyTorch and found the framework very intuitive. This helped me decide to learn PyTorch as my main Deep Learning framework. 
- Find a away to explore recent literature on the field and improve on the habit of reading more papers.
- Implement an object detection framework to detect/localise interesting information on Bulb's bills/documents. 

## Super summarised notes on Andrew's Ng Convolutional Neural Networks chapter:

**Week 1: Overview**

- Convolve a n x n image with a f x f filter (often called kernel) and the result is a f-n+1 by f-n+1 image. We get the resulted matrix by doing the element-wise product sliding the filter.
- Convolution filters are useful to detect abstractions on images (edges for instance).
- Padding is used to expand the image with 0 (black) pixels around. Padding 1 means add 1 pixel around the image.
- Padding can be used to allow multiple convolutions to be applied without shrinking the size of the image (same padding). For instance by padding a 6x6 image with 2 (we get a 8x8 image) allows us to convolve with a 3x3 filter and get a 6x6 image still (8-3+1). 
- The formula to find out what padding to apply in order to keep original size is p = (f-1)/2 (given a predefined f shape filter).
- The content of the filter is normally treated as hyperparameters of the network and tuned during training. 
- Strided convolutions move by s pixels (instead of 1).
- floor((n + 2p - f)/s) + 1) is the final shape of a 1D matrix convolved by a f x f filter with padding p and stride s.
- Cross-correlation in traditional math books will rotate the kernel matrix first but in deep-learning is called convolution and it doesn't rotate.
- For RGB images, we convolve over volume with filters of f x f x n_channels and the number of channels in the image should match the number of channels in the filter.  a 6x6x3 image convolved with a 3x3x3 filter still returns a 3D matrix of 4x4 because we multiply and sum all the pixels of the filter with the section of the image. 
- We can convolve the same image simultaneously by 2 filters (example 3x3x3 and 3x3x3). This will return a 4x4x2 matrix (where 2 is the number of filters). 
- A great thing about convolutional layers is that the size and number of the filters is agnostic of the size of the input. This makes the process quite scalable and is less prone to overfitting, as the number of parameters is kept relatively small. so a conv layer with 4 3x3x3 parameters will have 27 + 1 (bias) * 4 parameters.
- Conv nets use also Pooling layers (together with convolutional layers).
- In Pooling we also have stride and filter size and based on that, we take an operation in a region. MaxPooling will take tthe max value on the fxf region. 
- The intuition is that the MaxPooling will preserve a relevant feature from in a filter region. 
- Pooling doesn't have parameters to learn (weights) only hyperparameters stride and filter shape f. 
- In volume (for RGB images) the pooling operation is made independent of channel. So the output will be as deep as the input. 
- Another less popular pooling operation is AveragePooling.
- Output of a Pooling layer is given by the same formula as the output of a convolutional layer (without p as pooling normally doesn't use pooling).
- Conv layers are followed by Pooling normally.
- As we go deeper in a CNN, nH and nW tend to decrease and nC (channels/volume) tend to increase.
- Convolutions are useful due to 'parameter sharing' (vertical edges can be useful in different places of the image) and 'sparsity of connections' (each output pixel is only dependent on a small number of pixels).

**Week 2: Overview**

- In this week we explore common architectures etc. 
- Potential for transferring useful architectures is huge in computer vision.
- LeNet-5, AlexNet and VGG are classic CNNs. ResNet (152 layer NN). Inception NN.
- LeNet-5 is from 1998 paper trained in 1D grayscale images (conv avgPool conv avgPool fc fc output). Interesting that older CNNs were using Sigmoid/Tanh (not ReLus activation functions).
- AlexNet (2012). Much bigger than LeNet (60M parameters) but similar architecture. This paper added ReLu and an innovative idea around training in two GPUs. This paper also introduced the idea of Local Response Normalisation (LRN) which is not super popular this days. Given an input block, it aims at normalising on the channel inputs. 
- VGG-16 (just conv layers with 3x3 filters with stride 1 and MaxPool layers with 2x2 filters and stride of 2). Large network 138M parameters but appealing simple architecture. There is also VGG-19 a larger version.
- ResNets (2015) are made of Residual blocks. A residual block uses shortcuts (or skip connections) to pass information deeper in NN. This allows to train much deeper networks (faster?). Empirically we see that for traditional plain networks the training error plateous after some layers being added, but with ResNet this doesn't happen.
- Adding ResNet block should not heart performance because we are adding the output of previous layers.
- ResNet assumes that when we add previous outtputs the dimensions are the same. That is why we add conv layers with same output shape (He et al. 2015 Deep residual networks for image recognition).
- 1 by 1 convolutions are interesting. Also called Network in Network. This is very useful to shrink the number of channels in the output. For example if we have an input of 28x28x192 and want to get the channels to 28 we can convolve with a 1x1x28 filter. On top of this advantage, 1x1 convolutions add non-linearities allowing us to map more complex functions. 

- Inception Networks. The idea is to apply multiple operations (pooling, conv, 1x1 convs) in the same layers and stack the output before being parsed to the next output. This allows us to say start with a 28x28x192 input and go tot 28x28x256.
- We stack the multiple outputs of the same layer by channel (concat by channel).
- The problem of Inception Networks is the computation cost.
- Inception Network has intermediary softmax outputs to make sure that features being computed during the process are relevant too.
- googleLeNet is the name of the network trained on the Going Deeper with Convolutions 2014 paper.
- DONT UNDERSTAND HOW DO WE GET SAME width and depth outputs by convolving with small filters.

- Transfer Learning is what we need almost always!
- Different deep learning frameworks allow us to do transfer learning with trainbaleParameter = 0 or freeze = 1
- A good trick is to pre-compute the frozen layers for all the examples in the training set and save those weights to disk. Then all we have to do is train a network on the feed-forward layers feeding the frozen weights as input.
- The idea is that the more training data we have the less layers we have to freeze from previously trained models.
- We can initialise our application with the downloaded weights and then train for a certain number of epochs.

- Data augmentation is a core idea to expand image datasets (often hard to label). The core idea is that transformations that are invariant to the thing we are trying to classify, should be considered. 
- Shearing, mirroring, random cropping, local wraping are examples.
- Color shifting is also commonly used (adding, subtracting to the RGB pixels). 

- Andrew Ng describes two sources of knowledge in an ML problem (labeled data and hand engeneering,model features, hacks etc). Due to how difficult is to get labellled data for computeer vision tasks, means the field has been evolving more on the later. 

- Multi-Crop at test time means for each test image with crop n different instances of the image and run predictions on each crop. The final prediction is then the average of the n cropped predictions. This is costly and so not super popular in production systems. 

**Week 3: Detection Algorithms**

- Sliding window detection we slide a fixed size window throguh the entire image covering all possible pixels. This is computationally expensive. There is ways to implemet this algorithm convolutional way without having to run it n times for each sliding window.
- This is done by transforming FC layers to CONV layers by applying n conv of size of the previous output.
- YOLO - You Only Look Once algo. Split the image into n equal section grid. And for each grid, prepare a label of lengh 8 including pC (probability of including object), bXm bY, bH, bW (bounding box if grid contains object), and 3 flags for each object being detected. The label is then n x n x 8 (for 3 classes). We then use back-propagation to map the image x to the output n x n x 8. This allows us to detect multiple objects in different locations in the same image with one network with convolution layers. 
- YOLO is also computationally attractive and suitable for live video streaming.
- In YOLO, bXm bY, bH, bW are specified relative to the grid cell. 
- 19 x 19 is classic grid size

- IoU (intersection of union) is used to evaluate how well the localisation of the bounding box is done. If the prediction is perfect the IoU is 1, and normally the threshold used is 0.5 to counting how many times the object is correctly localised.
- Non-max supression simply outputs the predicted bounding-box with higher IoU against the labelled/actual bounding-box (and drop the remaining ones). For each grid slot, two anchor boxes are predicted (some with really low pC)/
- Anchor-boxes is this idea of including multiple labels/bounding-boxes for the same grid. So a NN trained on this framework would be trained agains a n x n (number of grids) x 8 (5 + number of classes) x 2 (2 anchor boxes). 

- Scanning algortihms are a set of object detection methods. Region Proposed (R-CNN) is another. This methods identify segmented reagions and then classifies each region (outputing the label and bounding box for each region).

## Cool articles and resources:

https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_convolutional_neuralnetwork/
https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/