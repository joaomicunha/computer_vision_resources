# **Computer Vision**

This repo documents my journey from being a Data Scientist with basic knowledge of image data and cv techniques, to hopefully build up more expertise. I have some experience with images related projects and have studied Deep Learning through courses and literature over the last year and a half.

On the 2nd of December 2019 I started this learning process. The progress will be:

- Quickly review Andrew's Ng Convolutional Neural Networks chapter to put my self in the right mindset and vocabulary space. 
- Complete PyTorch course (ideally with Computer Vision orientation). During the PyData Conference in London 2019, I had the opportunity to attend multiple presentations using PyTorch and found the framework very intuitive. This helped me decide to learn PyTorch as my main Deep Learning framework. 
- Find a away to explore recent literature on the field and improve on the habit of reading more papers.

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
- The problem is that is quite slow to run. Fast R-CNN and Faster R-CNN are newer implementations using convolution layers to acelarate the training.

**Week 4 **




** Pytorch for Computer Vision Course (Udacity) **

Day 1 Notes: 

- torch.FloatTensor -> to create a tensor float
- torch.tensor -> to create a general tensor
- tensor.view(rows,cols) -> to reshape a tensor
- tensor.view(rows, -1) -> reshapes the tensor to rows number of rows and infers the necessary number of colums to be able to match. for instance if we have a length 6 1d tensor we can shape to 3 rows and 2 columns as 3 x 2 is 6. We can achieve this by tensor.view(3,2) or tensor.view(3,-1)
- torch.from_numpy(a_numpy_array) -> to convert numpy array to tensor
- torch.dot(tensor1, tensor2) -> dot product
- torch.linspace(0,10,5) -> useful to plot a function where the function can be a function of this. 
- greyscale images are 2D matrixes with values from 1 (black) to 255 (white)
- a_tensor.dim() -> number of dimensions i.e. 2 for 2D matrix
- torch.arrange(18).view(3,3,2) -> generates a tensor with 18 values and shape 3 (rows), 3 (columns) and 2 (channels).
- torch.matmul(tensor1, tensor2) -> obviously only 2 matrixes where the columns of the first and the rows of the second are the same, can be multiplied.
- torch.tensor(2.0m requires_grad = True) -> to define a tensor that requires a derivative to be done.
- For instance if we have a function y = x**2 + z**3 and want to compute the partial derivatives of x and z respectively, we would create to x and z tensors (e.g. x = torch.tensor(1.0, requires_grad=True)) and then call y.backward() (to compute the derivatives). We can access these by x.grad or z.grad.

Day 2 Notes: 

- We need to define a forward() function to be able to compute the output of a function given a set of inputs.
- To predict one  output we do forward(torch.tensor([2])) to predict multiple we do forward(torch.tensor([[2],  [7]]))
- from torch.nn import Linear  -> model = Linear(in_features = 1, out_features=1) instanciates a Linear model with 1 set of inputs and ouputs (one model.bias one model.weight).
- nn.Linear() -> initiates random weights
- torch.manual_seed(1)  -> set seed
- we can also build custom Models class -> traditionally PyTorch custom classes inherit from nn.Module (import torch.nn as nn). We can start any custom PyTorch model by using:

```python
class LR(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.linear = nn.Linear(input_size, output_size)
  def forward(self, x):
    pred = self.linear(x)
    return(pred)
```

- if we instante this class model = LR(1,1) we can get our params as model.parameters().
- subtract the derivative of the partial derivative on the point to move  towards the minima as the partial derivative of a point points up. this is gradient descent where we update w1 = w0 - learn_rate * derivative_of_loss_function_to_x.
- nn.MSELoss() defines a choise for loss function
- we need to pick an optimizer too. torch.optim.SGD(model.paramters(), lr = 0.01) -> stochastic gradient descent is the same concept of gradient descent but for the latter we optimise for all the samples simultaneously which makes the process slow. 
- we train our model for a certain number of epochs - > an epoch is the number of times we pass the dataset back and forth trough the model.
- tensor.item() converts the tensor to a number.
- We then iterate through each epoch:


```python
criterion = nn.MSELoss()
optimiser = torch.optim.SGD(model.paramters(), lr = 0.01)
losses = []
epochs = 100
model = Model()

for i in range(epochs):
    #for each epoch
    #feed x through the model and compute the first vector of predictions
    y_pred = model.forward(x)
    #calculate the loss
    loss = criterion(y_pred, y)
    print("epoch: {} Loss: {}".format(str(i), loss.item()))
    #Store the loss values
    losses.append(loss)
    #Since the gradients acucmulate, we have to always set it to zero
    optimiser.zero_grad()
    #compute the partial derivatives
    loss.backward()
    #update the parameters
    optimiser.step()
```

Day 3 Notes:

- Create a Pytorch Dataset - pytorch has access to multiple pre loaded dataset.
- Cross-entropy - because classification models predict probabilities, we can take the sumation of the log of each probability. we want to minimise the cross-entropy (nn.BCELoss()).
- For classification, we pass the forward output to a sigmoid function:

```python
class Model(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.linear = nn.Linear(input_size, output_size)
  def forward(self, x):
    pred = torch.sigmoid(self.linear(x))
    return(pred)

#Instantiate:
model = Model(2,1)
```

Day 4 Notes:

- For deep NN we can create a new base class that supports a number of hiden layers in the init method.

```python
class Model(nn.Module):
  def __init__(self, input_size, H1, output_size):
    super().__init__()
    self.linear = nn.Linear(input_size, H1)
    self.linear2 = nn.Linear(H1, output_size)
  def forward(self, x):
    x = torch.sigmoid(self.linear(x))
    x = torch.sigmoid(self.linear2(x))

    return(x)
```


- AdamOptimiser - adapting learning algorithms. combines rmsprop and adagrad. it covers the limitation of SGD which is the need to pick a learning rate. Adam picks learning rates for each weight. 

Day 5 Notes:

-  we install both torch and torchvision (this package contains transformations and commonly used datasets).
- Often we compose transformers to be applied when we download the data. Multiple sequential transformers can be applied. The normalize() one aimss to center the pixel features to 0.5 mean and 0.5 standard deviation.
- The DataLoader defines how we will feed the data to the network for training. In the case below we will use batches of size 100 using shuffling. 

```python
transform = transforms.Compose([transforms.Resize((28,28)), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, ), (0.5, ))])


train_data = datasets.MNIST(root = "./data", train=True, download=True, transform=transform)

training_loader = torch.utils.data.DataLoader(dataset = training_dataset, batch_size=100, shuffle=True)
```

- Function to convert transformed tensor to both numpy array for plotting (transpose to go to h,w, c and invert the normalisation and clipping)

```python
def im_convert(tensor):
  image = tensor.clone().detach().numpy()
  image = image.transpose(1,2,0)
  #inverts the normalisation (x - mean)/std.var
  image = image*np.array((0.5, 0.5, 0.5)) + image*np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return(image)
```

- nn.CrossEntropyLoss() (combinartion of LogSoftmax() and NLLLoss()) is generaly used for multiclass classification as a cost function. Very interesting that we don't apply an activation function in the final linear layer and just output the raw final output (with no softmax).

```python
from torch import nn
import torch

class Classifier(nn.Module):
    
    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    def forward(self, x):
        x = F.relu(self.linear1(x))  
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

model = Classifier(784, 125, 65, 10)
criterion = nn.CrossEntropyLoss()
#sometimes need to tune this learning rate:
optimizer = torch.optim.Adam(model.parameters(). lr = 0.01)

```

- to train with epoch and batches, we need to feed each batch n epoch times.

```python 
epochs = 12
running_loss_history = []

for e in range(epochs):
    running_loss = 0.0
    running_corrects = 0.0
    for inputs, labels in training_loader:
      #flat the input images:
      inputs = inputs.view(784, -1)
      outputs = model.forward(inputs)
      loss = criterion(outputs, label)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      #torch.max() gives the index of the top n max values in the tensor.
      _, preds = torch.max(outputs, 1)
      #compare predictions with labels from the loader:
      running_corrects += torch.sum(preds == labels.data)
      running_loss += loss.item()

    epoch_loss = running_loss/len(train_loader)
    epoch_acc = running_corrects.float()/len(training_loader)
    running_loss_history.append(epoch_loss)
```

- to estimate validation performance we run with with torch.no_grad() to optimise memory as we are only doing inference and don't need the derivatives.

Day 6 notes:

- convolutions solve the computanional problem of using FF NN for image classification as some images are high dimentional (72x72 rgb images is 15552 pixels which means we would have to use 15552 nodes in the first hidden layer).
- convolutions combined with pooling layers are great to combat overfitting. 
- receptive field is the area where the conv kernel performs and a feature map is the output of a series of convolution layers including image features.
- The author of the course suggest Relu activation function is more biologically sound as other activation functions as neurons activate at a minimum value oof zero. ReLu is also more robust to vanishing gradient problem. 
- the filter's output is a measure of similarity of the feature in the filter and the same feature in the image.
- pooling layers aree designed to reduce the complexity of the model keeping the essential features. This reduces the computational cost and the risk of overfitting. Most importantly, it's scale invariant!
- as we go deeped in the conv network, we start encoding specific abstract features. 
- fully connected layers are solely responsible for classification task whilst the conv and pooling layers will extract features. 

- we can use a similar code structure to implement a convolutional network.
- The first value of nn.Conv2d is the number of channels of the input (1 for greyscale) and the second is the number of filters.

- In google colab we can add GPUs in Runtime. To use Cuda GPUs we need to send the model and the inputs to the gpu using device inputs = inputs.to(device) where device is specified before. this has to be done to all inputs and labels and models to be used in the training process (training and val sets).

- we can use dropout layer to reduce overfitting. See below.
```python
from torch import nn
import torch

class LeNet(nn.Module):
    
    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        #2 by 2 max pooling kernel
        x = F.max_pool_2d(x, 2, 2) 
        x = F.relu(self.conv2(x)) 
        x = F.max_pool_2d(x, 2, 2) 
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = LeNet(784, 125, 65, 10)
criterion = nn.CrossEntropyLoss()
#sometimes need to tune this learning rate:
optimizer = torch.optim.Adam(model.parameters(). lr = 0.01)

```



## Cool articles and resources:

https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_convolutional_neuralnetwork/
https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/