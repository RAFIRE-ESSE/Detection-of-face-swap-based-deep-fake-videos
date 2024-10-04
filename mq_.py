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

manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

workers = 2
batch_size = 50
nz = 100
num_epochs = 5
lr = 0.1
beta1 = 0.5
ngpu=1
ngf,nc = 3,3
ndf = 64

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
def devil_pres(pri_,org_):
    b,m=0,0
    for i in zip(pri_,org_):
        if i[1]==0:
            if i[0]<=0.10:
                b+=1
        if i[1]==1:
            if i[0]>=0.90:
                m+=1

    if b>=250 and m>=250:
        return True,b,m
    else:
        return False,b,m

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 8, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 6, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 7, 3, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


netG = Generator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.devil=nn.Sequential(nn.Conv2d(3, 4,3),
			nn.MaxPool2d(2, 2),

			nn.Conv2d(4, 8, 3),
			nn.MaxPool2d(2, 2),

			nn.Conv2d(8, 16, 3),
			nn.MaxPool2d(2, 2),

			nn.Conv2d(16, 32, 3),
			nn.MaxPool2d(2, 2),

			nn.Conv2d(32, 64, 3),
			nn.MaxPool2d(2, 2),

			nn.Conv2d(64, 200, 3),
			nn.MaxPool2d(2, 2),
			nn.Sigmoid()
			)

	def forward(self,x):
		return self.devil(x)


netD = Discriminator().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)

#netG.load_state_dict(torch.load(f"./G_.pth"))
#netD.load_state_dict(torch.load(f"./D_.pth"))

criterion,img_devil = nn.BCELoss(),0
fixed_noise = torch.randn(1, nz, 1, 1, device=device)
real_label = 0.
fake_label = 1.
optimizerD = optim.Adam(netD.parameters(), betas=(beta1, 0.999))
schedulerD=torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.86)
optimizerG = optim.Adam(netG.parameters(), betas=(beta1, 0.999))
schedulerG=torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.86)

if __name__=="__main__":
	img_list = []
	G_losses = []
	D_losses = []
	iters,z_ = 0,0

	print("Starting Training Loop...")
	while(True):
	    for i, data in enumerate(dataloader, 0):
	        netD.zero_grad()
	        real_cpu = data[0].to(device)
	        b_size = real_cpu.size(0)
	        label = torch.full((b_size*200,), real_label, dtype=torch.float, device=device)
	        output = netD(real_cpu).view(-1)
	        errD_real = criterion(output, label)
	        errD_real.backward()
	        D_x = output.mean().item()

	        noise = torch.randn(b_size, nz, 1,1, device=device)
	        fake = netG(noise)
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

	        if img_devil==500:
	        	noise = torch.randn(b_size, nz, 1,1, device=device)
	        	fake = netG(noise)
	        	fig = plt.figure(figsize=(10, 10))
	        	for i in range(9):
	        		fig.add_subplot(3,3,i+1)
	        		plt.imshow(fake[i].permute(1, 2, 0).detach().numpy())
	        	plt.savefig("devil.jpg")
	        	img_devil=0
	        	torch.save(netD.state_dict(),f'./D_.pth')
	        	torch.save(netG.state_dict(),f'./G_.pth')
	        img_devil+=1
	        print(img_devil)

	    print(optimizerD.param_groups[0]["lr"],optimizerG.param_groups[0]["lr"])
	    schedulerD.step()
	    schedulerG.step()
	    print(optimizerD.param_groups[0]["lr"],optimizerG.param_groups[0]["lr"])
	    out_,lab_=[],[]
	    for i in dataloader_:
	        devil_test=netD(i[0])
	        out_+=torch.reshape(devil_test,[-1]).tolist()
	        lab_+=i[1].tolist()
	    angel_z=devil_pres(out_,lab_)
	    if angel_z[0]==True:
	        break
	    #print(out_,lab_)
	    print(angel_z[1]+angel_z[2])
	    if z_<angel_z[1]+angel_z[2]:
	        z_=angel_z[1]+angel_z[2]
	        torch.save(netD.state_dict(),f'./weight_D/D_{angel_z[1]+angel_z[2]}.pth')
	        torch.save(netG.state_dict(),f'./weight_G/G_{angel_z[1]+angel_z[2]}.pth')
	        open("weight.txt","w").write(f"{angel_z[1]+angel_z[2]}")

