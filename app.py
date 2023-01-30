# Bart simpson 0
# charles_monthomery 1
# Krusty 2
# Lisa_simpson 3
# Marge_simpson 4
# Milhouse 5
# Moe_szyslak 6
# Ned_flanders 7
# principal_skinner 8

from flask import Flask, jsonify
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import io
import torch
import torch.nn as nn
import geffnet
import json


# 이거 안쓰니까 난리남
class SimpleEFN(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b5_ns'):
        super(SimpleEFN, self).__init__()
        # Download pretrained model
        self.efn = geffnet.create_model(model_name, pretrained=True)
        self.feat = nn.Linear(self.efn.classifier.in_features, 9)
        self.efn.classifier = nn.Identity()

    def extract(self, x):
        return self.efn(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.feat(x)
        return x

app = Flask(__name__)

simpson_class_index = json.load(open('./simpson_class_index.json'))
file_name = './simpson_clf.pt' 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = torch.load(file_name, device)
model.eval()

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})

def transform_image(image_bytes):
    image_size=456
    transformation = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ToTensorV2()])
    image = np.asarray(Image.open(io.BytesIO(image_bytes)))
    
    return transformation(image=image)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor = tensor['image'].unsqueeze(0)
    
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return simpson_class_index[predicted_idx]
    
with open("./image_samples/milhouse.jpeg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes))