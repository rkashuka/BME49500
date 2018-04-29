import torch
import torch.nn as nn
from imagefolder import TrainImageFolder
from torchvision import transforms
from torch.autograd import Variable
from colornet import ColorNet
from skimage.color import lab2rgb
from skimage.io import imsave
import numpy as np


# Hyperparameters
batch_size = 1
num_workers = 0
data_dir = "./data_grouped/data_multi/test/"
cuda = torch.cuda.is_available()
print('Initial seed:', torch.initial_seed())

# Define transformation
original_transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.Scale(256),
    # transforms.RandomCrop(224),
    #transforms.ToTensor()
])

# Read dataset
test_set = TrainImageFolder(data_dir, original_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Define model
colornet = ColorNet()
model = colornet.model
model.load_state_dict(torch.load('./revisions/multi_data/colornet_params.pkl'))
if cuda:
    model.cuda()

i = 0
for data, _ in test_loader:
    # img = data
    # imsave("./result/img_" + str(i) + ".png", data.squeeze(0).squeeze(0) / 128)
    # print(data.shape)
    if cuda:
        data = data.cuda()
    data = Variable(data)

    pred = model(data)
    pred = pred * 128
    # data = data / 128

    color_img = torch.cat((data, pred), 1)
    color_img = color_img.squeeze(0)
    color_img = color_img.data.cpu().numpy().transpose((1, 2, 0))
    print(color_img.shape)

    color_img = lab2rgb(color_img.astype(np.float64))
    imsave("./result/img_" + str(i) + ".png", color_img)
    # imsave("./result/img_" + str(i) + ".png", color_img[:, :, 1])
    i += 1




    # cur = np.zeros((256, 256, 3))
    # cur[:, :, 0] = color_me[i][:, :, 0]
    # cur[:, :, 1:] = output[i]
    # imsave("./result/img_" + str(i) + ".png", lab2rgb(cur))

