'''
main.py
Ultimately the goal here will be to set up a flask server
create a webpage where user can input or draw a number and
the canvas is sent to the flask server where a prediction is made
and then presented in a prediction box
'''


from flask import Flask, render_template, jsonify, request
import io
import json
from torchvision import models
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from PIL import Image, ImageOps
import base64

app = Flask(__name__)
model = torch.load('models/mnist_train_epoch_1.pt')
model.eval()
model.cuda()

def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

def transform_image(image_bytes):
    trans = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    image = remove_transparency(image)
    image = image.resize((64,64), Image.ANTIALIAS)
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    return trans(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor = tensor.cuda()
    outputs = model.forward(tensor)
    print(outputs)
    _, y_hat = outputs.max(1)
    return y_hat.item(), 1

@app.route("/",methods=['GET'])
def root():
    if request.method == 'GET':
        return render_template('paint.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.form['file']   
        img_data = file.split(',')[1]
        img_data = base64.b64decode(img_data)
        pred, pred_conf = get_prediction(image_bytes=img_data)
        print(pred)
    return jsonify({'Prediction': pred, 'Confidence': pred_conf})

    

if __name__ == '__main__':
    app.run()

