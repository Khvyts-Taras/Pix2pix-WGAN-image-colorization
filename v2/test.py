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

# Трансформ для уменьшения изображения до 64x64 и приведения к серому формату
transform_to_64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Трансформ для приведения изображения к тензору
transform_to_tensor = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

test_dir = 'test'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    img = Image.open(img_path)
    
    # Оригинальный размер изображения
    original_size = img.size
    
    # Преобразование изображения к размеру 64x64
    img_64 = transform_to_64(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_64 = unet(img_64)
    
    # Преобразование результата к оригинальному размеру
    output_64_resized = nn.functional.interpolate(output_64, size=original_size, mode='bilinear', align_corners=False)
    
    # Преобразование оригинального изображения к серому и тензору
    img_gray = transform_to_tensor(img).unsqueeze(0).to(device)
    
    # Прибавление оригинального серого изображения
    final_output = output_64_resized + img_gray
    
    # Сохранение результата
    vutils.save_image(final_output, os.path.join(output_dir, f'sample_{img_name}.png'), normalize=True)

print("Процесс завершён. Результаты сохранены в папке 'output'.")
