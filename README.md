# End-to-end-compression

A PyTorch code to implement end to end image compression framework. 

Reference : https://arxiv.org/abs/1708.00838



 This framework is consists of a compression layer called comCNN and reconstruction layer called recCNN. Both networks are fully convolutional neural networks. 
This architecture is implemented in PyTorch framework. Google Colaboratory was used for its execution. The implementation was tested on 2 datasets - CIFAR and STL10. Both datasets were available in torchvision module.
Since code is written in jupyter notebook, It can be executed by selecting 'run all'. This code uses GPU if available. The flag cuda is set by using a function is_available: torch.cuda.is_available(). 
Adam optimizer is used.
Since there is no high level function for interpolation in pytorch, a wrapper class is used which encapsulates interpolate function in nn.functional module. 
For ease of implementation, bilinear interpolation is used instead of bicubic interpolation.


The results are as below : 
