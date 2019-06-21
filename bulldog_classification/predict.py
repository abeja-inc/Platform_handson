# prediction
import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models

num_classes = 4

# load the model for CPU
device = torch.device('cpu')
model = torchvision.models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(2048, num_classes)
model.load_state_dict(torch.load(
    os.path.join(os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.'),
                 'model.pt'), map_location=device))


def image_to_tensor(img):
    img = img.convert('RGB')
    transformations = transforms.Compose([transforms.Resize(size=224),
                                          transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    image_tensor = transformations(img)[:3, :, :].unsqueeze(0)
    return image_tensor


def decode_predictions(result):
    categories = {
        0: 'BOSTON_BULL',
        1: 'BULL_MASTIFF',
        2: 'FRENCH_BULLDOG',
        3: 'STAFFORDSHIRE_BULLTERRIER'
    }
    result_with_labels = [{"label": categories[i], "probability": score} for i, score in enumerate(result)]
    return sorted(result_with_labels, key=lambda x: x['probability'], reverse=True)


def handler(_iter, ctx):
    for iter in _iter:
        img = Image.fromarray(iter)
        image_tensor = image_to_tensor(img)
        image_tensor = image_tensor.to(device)
        model.eval()
        output = model(image_tensor)
        # convert output probabilities to predicted class
        softmax = nn.Softmax(dim=1)
        preds_tensor = softmax(output)
        result = np.squeeze(preds_tensor.to(device).detach().numpy())
        sorted_result = decode_predictions(result.tolist())
        yield {"result": sorted_result}
