# **Computer Vision**

This repo documents my journey from being a Data Scientist with basic knowledge of image data and cv techniques, to hopefully build up more expertise. 

On the 2nd of December 2019 I started this learning process. The progress will be:

- Quickly review Andrew's Ng Convolutional Neural Networks chapter to put my self in the right mindset and vocabulary space. 
- Complete PyTorch course (ideally with Computer Vision orientation). During the PyData Conference in London 2019, I had the opportunity to attend multiple presentations using PyTorch and found the framework very intuitive. This helped me decide to learn PyTorch as my main Deep Learning framework. 
- Find a away to explore recent literature on the field and improve on the habit of reading more papers.
- Implement an object detection framework to detect/localise interesting information on Bulb's bills/documents. 

## Notes on Andrew's Ng Convolutional Neural Networks chapter:

- Convolve a n x n image with a f x f filter (often called kernel) and the result is a f-n+1 by f-n+1 image. We get the resulted matrix by doing the element-wise product sliding the filter.
- Convolution filters are useful to detect abstractions on images (edges for instance).
- Padding is used to expand the image with 0 (black) pixels around. Padding 1 means add 1 pixel around the image.
- Padding can be used to allow multiple convolutions to be applied without shrinking the size of the image (same padding). For instance by padding a 6x6 image with 2 (we get a 8x8 image) allows us to convolve with a 3x3 filter and get a 6x6 image still (8-3+1). 
- The formula to find out what padding to apply in order to keep original size is p = (f-1)/2 (given a predefined f shape filter).
- The content of the filter is normally treated as hyperparameters of the network and tuned during training. 
- Strided convolutions move by s pixels (instead of 1).
- floor((n + 2p - f)/s) + 1) is the final shape of a 1D matrix convolved by a f x f filter with padding p and stride s.


## Cool articles and resources:

https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_convolutional_neuralnetwork/