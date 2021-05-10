import torch
import torchvision
import numpy as np
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from efficientnet_pytorch import EfficientNet


idx_to_class_map = {0: 'class-273', 1: 'class-390', 2: 'class-4', 3: 'class-448', 4: 'class-536', 5: 'class-639', 6: 'class-653', 7: 'class-654', 8: 'class-663', 9: 'class-854'}
class_to_species_map = {'class-273': 'Crotalus Adamanteus', 'class-390': 'Nerodia Erythrogaster', 'class-4': 'Thamnophis Proximus', 'class-448': 'Lampropeltis Californiae', 'class-536': 'Thamnophis Radix', 'class-639': 'Diadophis Punctatus', 'class-653': 'Nerodia Fasciata', 'class-654': 'Storeria Occipitomaculata', 'class-663': 'Crotalus Scutulatus', 'class-854': 'Agkistrodon Piscivorus'}
model_path = "../models/efficientnet-b5-final.pt"


class EfficientNetClassifier(nn.Module):
    def __init__(self, n_classes, model_name="efficientnet-b5"):
        super(EfficientNetClassifier, self).__init__()

        self.effnet =  EfficientNet.from_pretrained(model_name)

        self.l1 = nn.Linear(1000 , 256)
        self.dropout = nn.Dropout(0.5)

        self.l2 = nn.Linear(256,n_classes) # 6 is number of classes
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        x = self.effnet(input)
        x = x.view(x.size(0),-1)

        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)

        return x


def load_efficientnet(saved_model_path):
    model_name = "efficientnet-b5"
    model = EfficientNetClassifier(n_classes=10)
    # model = EfficientNet.from_pretrained(model_name)
    image_size = EfficientNet.get_image_size(model_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(saved_model_path))
    return model, image_size, device


def load_image_transforms(image_size):
    image_transform  = torchvision.transforms.Compose([torchvision.transforms.Resize((image_size, image_size)),
                        torchvision.transforms.CenterCrop(image_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize([0.0432, 0.0554, 0.0264], [0.8338, 0.8123, 0.7803]),
                ])
    return image_transform


def predict(img):
    model, image_size, device = load_efficientnet(model_path)
    image_transform = load_image_transforms(image_size)

    img = image_transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    outputs = model(img)
    _, preds = torch.max(outputs, 1)
    class_idx = preds.item()

    print(idx_to_class_map[class_idx])
    outputs = outputs.cpu().detach().numpy()[0]
    # outputs /= outputs.max()
    outputs = outputs.tolist()
    score = max(outputs)
    label = class_to_species_map[idx_to_class_map[class_idx]]

    return label, score


st.title("Snake Species Classification with EfficientNet-B5")
st.header("Snake Speicies Classification Example")
st.text("Upload a Snake Image for species classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.')

    st.write("")
    st.write("Classifying...")

    label, score = predict(image)

    st.write("Snake Species: {}".format(label))
    st.write("Species Score: {:.4f}".format(score))
