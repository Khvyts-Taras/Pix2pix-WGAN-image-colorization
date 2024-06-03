import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torchvision.utils as vutils
from models import UNet, Critic

# Параметры
batch_size = 16
lr = 1e-4
num_epochs = 100
N_CRITIC = 5
GRADIENT_PENALTY = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("/data/Places365", exist_ok=True)
os.makedirs('samples', exist_ok=True)
os.makedirs('models', exist_ok=True)


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
dataset = datasets.Places365("/data/Places365", split='val', small=True, download=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Gradient Penalty
def gradient_penalty(real, fake, critic):
    m = real.shape[0]
    epsilon = torch.rand(m, 1, 1, 1, device=device)
    interpolated_img = epsilon * real + (1 - epsilon) * fake
    interpolated_img.requires_grad_(True)
    interpolated_out = critic(interpolated_img)
    grads = autograd.grad(outputs=interpolated_out, inputs=interpolated_img,
                          grad_outputs=torch.ones(interpolated_out.shape).to(device),
                          create_graph=True, retain_graph=True)[0]
    grads = grads.view(m, -1)
    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty


unet = UNet().to(device)
critic = Critic().to(device)

optimizer_g = optim.Adam(unet.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.999))

# Загрузка модели
start_epoch = 0
checkpoint_path = 'models/checkpoint.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    start_epoch = checkpoint['epoch'] + 1


total_iters = 0
for epoch in range(start_epoch, num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        gray_imgs = transforms.Grayscale()(imgs)


        optimizer_d.zero_grad()

        real_imgs = torch.cat((gray_imgs, imgs), dim=1)
        real_out = critic(real_imgs)
        fake_imgs = unet(gray_imgs) + gray_imgs
        fake_concat = torch.cat((gray_imgs, fake_imgs), dim=1)
        fake_out = critic(fake_concat.detach())

        grad_penalty = gradient_penalty(real_imgs, fake_concat, critic)
        d_loss = (fake_out.mean() - real_out.mean()) + GRADIENT_PENALTY * grad_penalty
        d_loss.backward()
        optimizer_d.step()
        total_iters += 1


        if i % N_CRITIC == 0:
            optimizer_g.zero_grad()

            fake_imgs = unet(gray_imgs) + gray_imgs
            fake_concat = torch.cat((gray_imgs, fake_imgs), dim=1)
            fake_out = critic(fake_concat)
            g_loss = -fake_out.mean()
            g_loss.backward()
            optimizer_g.step()

        if i % 100 == 0:
            log_message = (f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                           f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
            print(log_message)
            with open("training_log.txt", "a") as log_file:
                log_file.write(log_message + "\n")

        # Сохранение моделей и контрольной точки
        if i % 500 == 0:
            torch.save({
                'epoch': epoch,
                'unet_state_dict': unet.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict()
            }, checkpoint_path)
            print('model saved')

            with torch.no_grad():
                fake_imgs = unet(gray_imgs) + gray_imgs
                vutils.save_image(fake_imgs, os.path.join('samples', 'sample_' + str(epoch + 1) + f'_{i}.png'), normalize=True)
            print('image saved')
