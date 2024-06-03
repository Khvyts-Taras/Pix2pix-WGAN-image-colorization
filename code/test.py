import os
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torchvision.utils as vutils
from models import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet = UNet().to(device)

# Загрузка контрольной точки
checkpoint_path = 'models/checkpoint.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Модель загружена с эпохи {start_epoch}")
else:
    raise FileNotFoundError(f"Контрольная точка {checkpoint_path} не найдена")

unet.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

test_dir = 'test'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = unet(img)

    vutils.save_image(output, os.path.join(output_dir, f'sample_{img_name}.png'), normalize=True)

print("Процесс завершён. Результаты сохранены в папке 'output'.")