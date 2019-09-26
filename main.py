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
model = torch.load('models/mnist_vgg_like.pt')
model.eval()
if torch.cuda.is_available():
    model.cuda()


#when the image returns and is converted from base64 there is a clear alpha channel instead of white
#this removes that
def remove_transparency(im, bg_colour=(255, 255, 255)):
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im

def transform_image(image_bytes):
    #most of this can be done with transforms but I'd just do it by hand for readability
    trans = transforms.Compose([transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    image = remove_transparency(image)
    image = image.resize((28,28), Image.ANTIALIAS)
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    return trans(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat.item()

@app.route("/",methods=['GET'])
def root():
    if request.method == 'GET':
        return render_template('paint.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.form['file']   
        img_data = file.split(',')[1] #The first part is an image description, this is not needed
        img_data = base64.b64decode(img_data) #The remainder is encoded in base64
        pred = get_prediction(image_bytes=img_data)
    return jsonify({'Prediction': pred})
   

if __name__ == '__main__':
    app.run()

