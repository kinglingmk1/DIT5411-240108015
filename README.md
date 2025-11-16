Before read readme
This assignment only training 1002 words because the maximum of training data of my computer is 1500 words more will run out of memory
Google colab I tried maximum training is 280 words then will force you to restart because run out of memory.

Training Machine
Motherboard: H610M-B
CPU: Intel 12400f
GPU: RTX 3060 12GB GDDR6 (PCIe x16 4.0)
RAM: DDR4 16GB x 2 (3200 MHz)
OS: Windows 10 22H2

Python version 3.9.25
CUDAToolKit  11.2
CUDNN 8.1.0


Using library (may not list all python)
cv2
matplotlib
numpy (version < 2)
zipfile
shutil
tensorflow (version < 2.11) because windows gpu training only available version <= 2.10
random
json



Method of augmented image creation
For me using windows
sys.stdin.reconfigure(encoding="utf-8")
this is needed because if I direct save as chinese will make the words save as wrong encoding format.

First, I use the unzipped data by https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset

In code you can see the variable path is using the unzipped data
output_folder is for store the processed 200 image of each word in 40 pictures store after unzip

It will search the folders name and get the name then get inside 50 image
I have a word count to make it limit on 40 pictures and use
- Original Picture
- rotate 90 degree
- rotate 270 degree
- resize 100x100 (from 50x50)
- shearing rows cols [[1, 0.5, 0], [0, 0.5, 1]]

To make 5 picture out and each word get 40 picture to generate 200 words

I also use cv2.imencode not cv2.imwrite because of windows problem.

Then for me make a json output for test did it really read and write the word in chinese and save in correct format as I use the windows platform.

Then I detect the tensorflow and gpu is available and show the tensorcore version and GPU name

I using the tensorflow.kears inside models to use for training
total using four conv block then connect it
I set the learning rate to 0.001 for tunning for best rate in my experience

Finally output .kears with .json / .h5 with .pkl model

In traiing the accuracy in final is
2503/2503 [==============================] - 114s 46ms/step - loss: 0.1529 - accuracy: 0.9513 - val_loss: 0.1066 - val_accuracy: 0.9699 - lr: 2.5000e-04
Which is vaild accuracy 0.9699 / vaild loss 0.1066

Extra
AIModel Folder zipped the training AI (1002 words)
graph.py contain the 1002 words AI accuracy, loss and time grapth 
Screenshoot Folder contain some test of model and the png of the grapth
I put some sample data on output / output2 / Traditional_Chinese_Data Folder

Reference

https://blog.csdn.net/ctwy291314/article/details/80897143

https://www.youtube.com/watch?v=xJtmj6hX5Lg&t=572s (for install tensorflow < 2.11 only) (Start from 09:08)

https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset

https://www.tensorflow.org/install/pip#windows-native

https://www.anaconda.com/

AI Used for some code
Cline
Github Copilot
