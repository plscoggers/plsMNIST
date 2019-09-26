import torch
import torch.nn as nn
import torch.nn.functional as F


#Because there's no nn.Flatten module we need to build one to flatten for the FC layers
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


'''
vgg_like_structure is modeling sort of a vgg13 type model
I was trying to run the MNIST data with something a little bit deeper
however I was running into vanishing gradient issues, as well as issues
with the image size.

To avoid resizing I've made the network a bit shallower.
For something like MNIST, this is a little bit less important but if I 
were to be training something more intense I would be using ResNet or
AlexNet most likely

This obviously can be more robust by wrapping it into a class
That way I can structure the output layer and input layers better

But for now this serves it's purpose
'''
vgg_like_structure = nn.Sequential( nn.Conv2d(1,64,kernel_size=3,padding=1),
                                    nn.Conv2d(64,64,kernel_size=3,padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2),
                                    nn.Conv2d(64,128,kernel_size=3,padding=1),
                                    nn.Conv2d(128,128,kernel_size=3,padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2),
                                    nn.Conv2d(128,256,kernel_size=3,padding=1),
                                    nn.Conv2d(256,256,kernel_size=3,padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256,512,kernel_size=3,padding=1),
                                    nn.Conv2d(512,512,kernel_size=3,padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.AdaptiveAvgPool2d((7,7)),
                                    Flatten(),
                                    nn.Linear(512 * 7 * 7,4096),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(),
                                    nn.Linear(4096,4096),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(),
                                    nn.Linear(4096,10),
                                    nn.LogSoftmax(dim=1))

#init weights so we're not just randomly guessing
#historically i've used xavier, but i'm curious about pytorch's choice of kaiming normal
def init_weights(vgg_like):
    for m in vgg_like.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)