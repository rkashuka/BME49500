import torch
import torch.nn as nn
import torch.optim as optim
from imagefolder import TrainImageFolder
from torchvision import transforms
from colornet import ColorNet
import os
from tensorboard_logger import Logger


# Hyperparameters
batch_size = 10
epochs = 3000
epochs_done = 0
kernel_size = 3
num_workers = 0
data_dir = "./data_grouped/forest/"
cuda = torch.cuda.is_available()
log_interval = 2
learning_rate = 0.01
momentum = 0.0
torch.manual_seed(1524633769)
print('Initial seed:', torch.initial_seed())


# Define transformation
original_transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.RandomCrop(224),
])

# Read dataset
train_set = TrainImageFolder(data_dir + 'train/', original_transform)
test_set = TrainImageFolder(data_dir + 'test/', val_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print("Train Set size: {0}, Test Set size: {1}\n".format(len(train_set), len(test_set)))

# Define model
colornet = ColorNet()
model = colornet.model

# Load existing weights
if os.path.exists('./colornet_params.pkl'):
    print('Loading existing weights ...\n')
    model.load_state_dict(torch.load('colornet_params.pkl'))

if cuda:
    model.cuda()

loss_fun = nn.MSELoss()
optimizer = optim.Adadelta(model.parameters())

# Define loggers
log_train = Logger("runs/run-train")
log_test = Logger("runs/run-test")

# Training
for epoch in range(epochs_done, epochs):
    avg_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        out = model(data)
        loss = loss_fun(out, target)

        loss.backward()
        optimizer.step()

        avg_loss = 0.9 * avg_loss + 0.1 * loss.item()

    log_train.log_value('loss', avg_loss, epoch)
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, avg_loss))

    torch.save(model.state_dict(), 'colornet_params.pkl')

    if epoch % 50 == 0:
        torch.save(model.state_dict(), './param_backup/colornet_params_' + str(epoch) + '.pkl')

    with torch.no_grad():
        avg_loss = 0.0
        for batch_idx, (data, target) in enumerate(test_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()

            out = model(data)

            loss = loss_fun(out, target)
            avg_loss += loss.item() * len(data)

        avg_loss = avg_loss / len(test_set)

    log_test.log_value('loss', avg_loss, epoch)
    print('Test Epoch: {} \tLoss: {:.6f}'.format(epoch, avg_loss))
