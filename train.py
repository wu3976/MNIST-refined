import torch.cuda as cuda
import torch
from torch import nn

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import numpy as np
import time

# ----------------------------Hyperparameters-----------------------#
# Size of each trainging batch.
BATCH_SIZE = 5000

# Learning rate. param -= param.grad * LEARNING_RATE
LEARNING_RATE = 0.7

# training epoch number
EPOCH = 50
# ----------------------------END Hyperparameters-------------------#


# deep learning with GPU

# get the device tp use, cuda in priority.
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# determine split indices
def split_indices(n, frac):
    validate_size = int(frac * n)
    ids = np.random.permutation(n)
    # (training_set, validation_set)
    return ids[validate_size:], ids[:validate_size]

def get_default_device():
    if cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')  # incase cuda is unavaliable.


# move data and model to device. This operation may take while
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]  # breakdown element and
        # recursively put elements into device if the data is list or tuple.

    return data.to(device, non_blocking=True)  # This may return a pointer pointing to the data in GPU


dataset = MNIST(root="data/", download=True, transform=ToTensor())


# A wrapped dataloader that put data into gpu in BATCH-WISE fashion.
# This can save GPU dram.
class DeviceDataloader():
    # Constructor
    # @param data_loader A DataLoader instance that need to be operated.
    # @param device The device going to put on.
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device

    # iterator function.
    def __iter__(self):
        for batch_tuple in self.data_loader:
            yield to_device(batch_tuple, self.device)  # do to_device. Yield is non-stop return,

    # length of self.data_loader
    def __len__(self):
        return len(self.data_loader)


# Model definition
class Mnist_NN_Model(nn.Module):

    def accuracy(self, prediction, target):
        _, max_pos = torch.max(prediction, dim=1)
        return torch.tensor(torch.sum(max_pos == target).item() / len(max_pos))

    def __init__(self, input_size, m_layer_size, output_size):
        # call parent class' initialization first
        super(Mnist_NN_Model, self).__init__()
        # hidden layer
        self.layer1 = nn.Linear(input_size, m_layer_size)
        # output layer
        self.layer2 = nn.Linear(m_layer_size, output_size)

    # @requires len(input_batch) == len(input_size)
    def forward(self, input_batch):
        # reshape batch in same memory
        input_batch = input_batch.view(input_batch.size(0), -1)  # size[0] is batch size,
        # which is fixed, so can use generic -1

        hidden_layer_result = self.layer1(input_batch)
        output_layer_input = nn.functional.relu(hidden_layer_result)
        prediction = self.layer2(output_layer_input)

        return prediction

    def training_step(self, batch):
        images = batch[0]
        labels = batch[1]
        out = self(images)  # Generate predictions
        loss = nn.functional.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, v_batch):
        images, labels = v_batch
        prediction = self(images)
        accuracy = self.accuracy(prediction, labels)
        loss = nn.functional.cross_entropy(prediction, labels)
        return {"validation loss": loss, "validation accuracy": accuracy}

# a baseline model without hidden layer.
class Mnist_Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Mnist_Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_batch):
        input_batch = input_batch.view(input_batch.size(0), -1)
        return self.linear(input_batch)

    def training_step(self, batch):
        images = batch[0]
        labels = batch[1]
        out = self(images)  # Generate predictions
        loss = nn.functional.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, v_batch):
        images, labels = v_batch
        prediction = self(images)
        accuracy = self.accuracy(prediction, labels)
        loss = nn.functional.cross_entropy(prediction, labels)
        return {"validation loss": loss, "validation accuracy": accuracy}

    def accuracy(self, prediction, target):
        _, max_pos = torch.max(prediction, dim=1)
        return torch.tensor(torch.sum(max_pos == target).item() / len(max_pos))

train_ids, validation_ids = split_indices(len(dataset), 0.1)
# Split set using split indices
training_set_sampler = SubsetRandomSampler(train_ids)
val_set_sampler = SubsetRandomSampler(validation_ids)
train_dl = DataLoader(dataset, BATCH_SIZE, sampler=training_set_sampler)  # trainging dataloader
val_dl = DataLoader(dataset, BATCH_SIZE, sampler=val_set_sampler)

# wrap dataloader
train_dl = DeviceDataloader(train_dl, get_default_device())
val_dl = DeviceDataloader(val_dl, get_default_device())


input_size = 784
output_size = 10
hidden_size = 64

model = Mnist_NN_Model(input_size, hidden_size, output_size)
# move model to GPU
to_device(model, get_default_device())
print("-----------------NN model--------------------")
print(model)

opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
for i in range(EPOCH):
    # time0 = time.time()
    for batch in train_dl:
        # print("Sample prediction: " + str(nn.functional.softmax(prediction, dim=1)[0]))
        loss = model.training_step(batch)
        loss.backward()

        opt.step()
        opt.zero_grad()
    # time1 = time.time()
    print("EPOCH: " + str(i))
    # print("Time cost in current epoch: " + str(time1 - time0))
    if i % 5 == 0 or i == EPOCH - 1:
        validation_data = [model.validation_step(validation_batch) for validation_batch in val_dl]
        losses = [data['validation loss'].data for data in validation_data]
        avg_loss = sum(losses) / len(losses)
        accs = [data['validation accuracy'].data for data in validation_data]
        avg_accuracy = sum(accs) / len(accs)
        print("Average loss: " + str(avg_loss))
        print("Average accuracy: " + str(avg_accuracy))

model_base = Mnist_Model(input_size, output_size)
to_device(model_base, get_default_device())
print("-------------------Baseline Model----------------------")
print(model_base)
opt_base = torch.optim.SGD(model_base.parameters(), lr=LEARNING_RATE)
for i in range(EPOCH):
    # time0 = time.time()
    for batch in train_dl:
        # print("Sample prediction: " + str(nn.functional.softmax(prediction, dim=1)[0]))
        loss = model_base.training_step(batch)
        loss.backward()

        opt_base.step()
        opt_base.zero_grad()
    # time1 = time.time()
    print("EPOCH: " + str(i))
    # print("Time cost in current epoch: " + str(time1 - time0))
    if i % 5 == 0 or i == EPOCH - 1:
        validation_data = [model_base.validation_step(validation_batch) for validation_batch in val_dl]
        losses = [data['validation loss'].data for data in validation_data]
        avg_loss = sum(losses) / len(losses)
        accs = [data['validation accuracy'].data for data in validation_data]
        avg_accuracy = sum(accs) / len(accs)
        print("Average loss: " + str(avg_loss))
        print("Average accuracy: " + str(avg_accuracy))