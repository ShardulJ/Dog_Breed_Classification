import io
import os
import json
from collections import namedtuple
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from flask import Flask, request, Response, jsonify, render_template, redirect
from model import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

dog_breeds = ['beagle', 'chihuahua', 'doberman', 'french_bulldog'
              , 'golden_retriever', 'malamute', 'pug', 'saint_bernard', 'scottish_deerhound', 'tibetan_mastiff']
              
app = Flask(__name__)
ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])
model = ResNet(resnet50_config, 10)
model.load_state_dict(torch.load('./model.pt'))
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.4758, 0.4534, 0.4032],
                                            [0.2377, 0.2342, 0.2339])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).float().unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    image = Variable(tensor, requires_grad=True)

    images = []
    labels = []
    probs = []


    y_pred, _ = model.forward(image)
    print(y_pred)
    y_prob = F.softmax(y_pred, dim = -1)
    top_pred = y_prob.argmax(1, keepdim = True)

    images.append(tensor.cpu())
    labels.append(top_pred.cpu())
    probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return images, labels, probs 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():    
    if request.method == 'POST':
        if 'photo' not in request.files:
            return redirect(request.url)
        file = request.files.get('photo')    
        img_bytes = file.read()
        images, labels, probs = get_prediction(img_bytes)
        return render_template("index2.html",s1 = dog_breeds[labels], s2 =torch.max(probs.data))


if __name__ == '__main__':
    app.run()
