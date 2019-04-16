import torch
import torch.nn as nn
from torchvision import transforms
from DBCNN import DBCNN
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
])

options = {'fc': True}
scnn_root = 'your path of SCNN model'
model = nn.DataParallel(DBCNN(scnn_root, options), device_ids=[0]).cuda()
model_name = type(model).__name__
print(model)

ckpt = "your path of the checkpoint file"
image_name = "your path of test image"
checkpoint = torch.load(ckpt)
model.load_state_dict(checkpoint)

model.eval()

I = Image.open(image_name)
I = test_transform(I)
I = torch.unsqueeze(I, dim=0)
I = I.to(device)
with torch.no_grad():
    score = model(I)

format_str = 'Prediction = %.4f'
print(format_str % score)






