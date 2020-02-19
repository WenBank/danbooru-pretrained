import torch
# Load the model
# model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')
model = torch.load('model/resnet50-13306192.pth', map_location=torch.device('cpu'))
# model.eval()


from PIL import Image
import torch
from torchvision import transforms
input_image = Image.open("img/danbooru_resnet1.png") # load an image of your choice
preprocess = transforms.Compose([
    transforms.Resize(360),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# The output has unnormalized scores. To get probabilities, you can run a sigmoid on it.
probs = torch.sigmoid(output[0]) # Tensor of shape 6000, with confidence scores over Danbooru's top 6000 tags


import matplotlib.pyplot as plt
import json
import urllib, urllib.request
# Get class names
with urllib.request.urlopen("https://github.com/RF5/danbooru-pretrained/raw/master/config/class_names_6000.json") as url:
    class_names = json.loads(url.read().decode())
# Plot image
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