## Update - 12/10/19
I have created, with a smaller style model, created a heroku app so you can play with this without having to download and do all that stuff!

https://mnist-pytorch-test.herokuapp.com/

The website has mixed results on mobile.  I could fix positions a bit better.

## plsMNIST
This is a short sample of working in PyTorch

Basically I designed a vgg like network in models.
train.py will download the MNIST dataset and train a model with 3 epoch's, it outputs the test scores, just pick which one is best and rename it to be whatever you want
The model output will go in the models folder.

main.py is a flask server.  Once running you can go to the 127.0.0.1:5000 or whatever you manually set flask to, and you should be in the main module.

You can draw a number in the box, sorry I know it's small but I was having predictions issues with larger boxes and having to resize, and it will make a prediction.

The canvas is sent to the server every 2.5s so it proactively guesses.  If it doesn't seem right, give it up to 5s.  It largely depends on the backend responding, so if you're using CPU and not GPU it may take even longer.

I was getting what appeared to be about >95% accuracy, my wife was getting about >90% accuracy when drawing on the website.  It seems to want to label 2's and 7's and 7's as 2 quite a bit.  You really need to make a pronounced underscore on the 2 to get it to recognize.  I usually draw my 7's with crosses, which definitely helps in this situation.

If I had a faster computer I could do this on and I had the time, I could probably make it better.

The model I've included had a 98.3% accuracy on the test set.  But there's obviously some differences in the way the data is handled (I'm basically sending the inverse of the image in binary form, where the MNIST data is more grayscale with values between 0 and 1 as opposed to just binary values like mine).

You can start the server by just running main.py (make sure you have the model you want to run and reference it on line 14 in main.py

My model can be downloaded here: https://www.dropbox.com/s/he5mggru2otpv89/mnist_vgg_like.pt?dl=0

Thanks!  And pls don't judge too harshly (I've never used flask before, and everyhing I've done I figured out in about 1 hour of google)


## Dependencies
This has dependencies of PyTorch, Flask, and PIL