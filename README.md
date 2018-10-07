# End-to-end-compression

This is my implementation of end to end image compression framework[1] .

You need to have following libraries installed in order to run this code : <br />
image==1.5.25<br />
numpy==1.14.5<br />
Pillow==5.2.0<br />
python-dateutil==2.7.3<br />
tensorboard==1.10.0<br />
tensorflow==1.10.1
<br /><br /><br />

Description of python files : 
<br />
end_to_end_net.py : This file contains the tensorflow code for ComCNN and RecCNN as described in the reference paper[1]. This code uses tensorflow layers API. <br />
utils.py : This file contains the python code for utilities used in main code to read / write / process image. <br />
end_to_end.py : This file contains tensorflow graph as described in the paper[1]. It contains the definition of hyperparameters, loss functions and optimizer used. Run this file as :
```python end_to_end.py``` 
to run this code. 

Below are the results obtained after training the network on Caltech 101[2] aeroplane dataset. I had used 200 epochs with the batch size of 128. 


<br /><br />


Inference results are as below : 

| Original Image  | Compact Representation | Reconstructed Image |
| ------------- | ------------- | ------------- |
| ![Original image of an aeroplane](https://github.com/kunalrdeshmukh/End-to-end-compression/blob/master/results/airplane_001/resized_img.jpg) Size : 29KB| ![Compact representation of aeroplane](https://github.com/kunalrdeshmukh/End-to-end-compression/blob/master/results/airplane_001/mid_im.jpg) Size : 16KB| ![Reconstructed image of an aeroplane](https://github.com/kunalrdeshmukh/End-to-end-compression/blob/master/results/airplane_001/final_im.jpg) Size:29KB|
| ![Original image of an stonehenge](https://github.com/kunalrdeshmukh/End-to-end-compression/blob/master/results/Random/resized_img.jpg) Size: 37KB| ![compact image of an stonehenge](https://github.com/kunalrdeshmukh/End-to-end-compression/blob/master/results/Random/mid_im.jpg) Size: 17KB |![Reconstructed image of an stonehenge](https://github.com/kunalrdeshmukh/End-to-end-compression/blob/master/results/Random/final_im.jpg) Size: 37KB|

<br />

All input images are of size 180 x 180. The compact representation of an image is in jpg format and it can be transferred through a channel which supports JPEG format. 

![End to end framework diagram ](https://github.com/kunalrdeshmukh/End-to-end-compression/blob/master/results/End_to_end_framework_diagram.png)



<br /><br />

References : 

[1] Jiang Feng, Et al. (August 2017). An End-to-End Compression Framework Based on Convolutional Neural Networks. IEEE Transactions on Circuits and Systems for Video Technology.. Harbin, China. arxiv e-archive.

[2] Caltech 101 dataset : http://www.vision.caltech.edu/Image_Datasets/Caltech101/
