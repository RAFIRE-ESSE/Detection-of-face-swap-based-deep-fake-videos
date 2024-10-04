import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

workers = 2
batch_size = 84
image_size = 64
nc = 3
nz = 1
ngf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root=f'./train_devil/train', transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

testset_ = torchvision.datasets.ImageFolder(root=f'./train_devil/test', transform=transform)
dataloader_ = torch.utils.data.DataLoader(testset_, batch_size=20,
                                              shuffle=True, num_workers=2)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
"""real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(6,31,1)
        self.pool =  nn.MaxPool2d(2,2)
        self.conv2 = nn.ConvTranspose2d(16,6, 5)
        self.fc1 = nn.Linear(4,36864)
        self.fc2 = nn.Linear(84,4)
        self.fc3 = nn.Linear(1,84)

    def forward(self, x):
    	x = self.fc3(x)
    	x = F.relu(self.fc2(x))
    	x = F.relu(self.fc1(x))
    	x = torch.flatten(x, 1) 
    	x = self.conv2(F.relu(self.pool(torch.reshape(x,(84,16, 48, 48)))))
    	x = self.conv1(F.relu(self.pool(x)))
    	return x


netG = Generator().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6,1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(36864,4)
        self.fc2 = nn.Linear(4, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
    	x = self.pool(F.relu(self.conv1(x)))
    	x = self.pool(F.relu(self.conv2(x)))
    	x = torch.flatten(x, 1) 
    	x = F.relu(self.fc1(x))
    	x = F.relu(self.fc2(x))
    	y=nn.Sigmoid()
    	x = y(self.fc3(x))
    	return x
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_ = nn.ConvTranspose2d(6,3,12,stride=4)
        self.pool =  nn.MaxPool2d(2,2)
        self.conv2_ = nn.ConvTranspose2d(12,6,4,stride=4)
        self.fc1 = nn.Linear(94,27648)
        self.fc2 = nn.Linear(84,94)
        self.fc3 = nn.Linear(1,84)

    def forward(self, x):
        z_x=x.shape[0]
        print(z_x)
        y=nn.Sigmoid()
        x = self.fc3(y(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc1(x))
        x = torch.flatten(x, 1) 
        x = self.conv2_(F.relu(self.pool(torch.reshape(x,(z_x,12, 48, 48)))))
        x = self.conv1_(F.relu(self.pool(x)))
        return x

netD = Discriminator().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)

criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1.
fake_label = 0.
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, nz, 1, device=device)
        fake = netG(noise)
        print(fake[0])
        print(fake[0].shape)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label) 
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                   % (epoch, num_epochs, i, len(dataloader),
           		errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

