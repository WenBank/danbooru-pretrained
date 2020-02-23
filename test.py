
# ###普通模型加载########################################################################################
# from PIL import Image
# import torch
# from torchvision import transforms
# input_image = Image.open("img/egpic2.jpg")#.convert('RGB') # load an image of your choice
# preprocess = transforms.Compose([
#     transforms.Resize(360),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
#
#
# # Load the model
# # model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')
# state_dict2 = torch.load('model/resnet50-13306192.pth',map_location='cpu')
# model
# model.load_state_dict(state_dict)
# model.eval()
#
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')
# with torch.no_grad():
#     output = model(input_batch)
# # The output has unnormalized scores. To get probabilities, you can run a sigmoid on it.
# probs = torch.sigmoid(output[0]) # Tensor of shape 6000, with confidence scores over Danbooru's top 6000 tags
####fastai#############################################################################################################
from fastai.vision import *
# if you put the resnet50 file in the current directory:
learn = load_learner(path='./model/', file='fastai_danbooru_resnet50.pkl')
# or any of these variants
model = learn.model # the pytorch model
mean_std_stats = learn.data.stats # the input means/standard deviations
class_names = learn.data.classes # the class names

# Predict on an image
input_image1 = open_image('./img/egpic2.jpg')
predicted_classes, y, probs = learn.predict(input_image1)
print(probs)
#######################################################################################################################
import matplotlib.pyplot as plt
import json
# import urllib, urllib.request
# Get class names
# with urllib.request.urlopen("https://github.com/RF5/danbooru-pretrained/raw/master/config/class_names_6000.json") as url:
#     class_names = json.loads(url.read().decode())
file = open('./config/class_names_6000.json', 'r', encoding='utf-8')
class_names = json.load(file)
# Plot image
from PIL import Image
input_image = Image.open("./img/egpic2.jpg")
plt.imshow(input_image)
plt.grid(False)
plt.axis('off')

def plot_text(thresh=0.2):
    tmp = probs[probs > thresh]
    inds = probs.argsort(descending=True)
    txt = 'Predictions with probabilities above ' + str(thresh) + ':\n'
    for i in inds[0:len(tmp)]:
        txt += class_names[i] + ': {:.4f} \n'.format(probs[i].cpu().numpy())
    plt.text(input_image.size[0]*1.05, input_image.size[1]*0.85, txt)


plot_text()
plt.tight_layout()
plt.show()
plt.imshow(input_image)