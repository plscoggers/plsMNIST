###plsMNIST
This is a short sample of working in PyTorch

Basically I designed a vgg like network in models.
train.py will download the MNIST dataset and train a model with 3 epoch's.
The model output will go in the models folder.


###Plans
I plan to eventually add a flask server to this and create a short front end.
Allows a user to draw a character on a canvas in HTML, asynch send to flask server
Calls the trained model to make a prediciton and returns the prediction to the user