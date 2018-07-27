from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray
import torch
import numpy as np

scale_transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.RandomCrop(224),
])


class TrainImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)

            img_lab = rgb2lab(img_original)
            img_lab = img_lab.transpose(2, 0, 1)
            img_original = img_lab[0, :, :]

            img_ab = img_lab[1:, :, :] / 128
            img_ab = torch.from_numpy(img_ab).float()

            img_original = torch.from_numpy(img_original).unsqueeze(0).float()

        return img_original, img_ab
