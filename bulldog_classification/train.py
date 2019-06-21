import io
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from tqdm import tqdm

from abeja.datasets import Client as DatasetClient
from abeja.train import Client as TrainClient
from abeja.train.statistics import Statistics as ABEJAStatistics

# check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# set Environment Variables
batch_size = int(os.environ.get('BATCH_SIZE', 12))
num_workers = int(os.environ.get('NUM_WORKERS', 1))
valid_size = float(os.environ.get('VALID_SIZE', 0.2))
n_epochs = int(os.environ.get('NUM_EPOCHS', 5))
learning_rate = float(os.environ.get('LEARNING_RATE', 0.001))

ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.')

save_path = os.path.join(ABEJA_TRAINING_RESULT_DIR,
                         os.environ.get('ARTIFACT_FILE_NAME', 'model.pt'))

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')

# convert data to a normalized torch.FloatTensor
train_transform = transforms.Compose([transforms.Resize(size=224),
                                      transforms.CenterCrop((224, 224)),
                                      transforms.RandomHorizontalFlip(),  # randomly flip and rotate
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([transforms.Resize(size=224),
                                      transforms.CenterCrop((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# import data from ABEJA Platform Dataset
def load_dataset_from_api(dataset):
    for item in tqdm(dataset.dataset_items.list(prefetch=True)):
        try:
            file_content = item.source_data[0].get_content()
            file_like_object = io.BytesIO(file_content)
            img = Image.open(file_like_object).convert('RGB')
            label_id = item.attributes['classification'][0]['label_id']
            yield img, label_id
        except OSError:
            print('Fail to load dataset_item_id {}.'.format(item.dataset_item_id))


# define customized dataset class
class CustomizedDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = dataset
        self.img = [item[0] for item in self.dataset]
        self.label_id = [item[1] for item in self.dataset]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        out_img = self.img[idx]
        out_label_id = self.label_id[idx]

        if self.transform:
            out_img = self.transform(out_img)

        return out_img, out_label_id


# prepare dataloader
def load_split_train_test(dataset_list):
    """ Split dataset into train and validation set """

    train_data = CustomizedDataset(dataset_list, transform=train_transform)
    valid_data = CustomizedDataset(dataset_list, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    trainloader = DataLoader(train_data, sampler=train_sampler,
                             batch_size=batch_size, num_workers=num_workers)
    validloader = DataLoader(valid_data, sampler=valid_sampler,
                             batch_size=batch_size, num_workers=num_workers)

    print('number of train datasets is {}.'.format(len(trainloader) * batch_size))
    print('number of valid datasets is {}.'.format(len(validloader) * batch_size))

    return trainloader, validloader


def train_model(trainloader, validloader, model, optimizer, criterion):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = 3.877533  # np.Inf

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    for epoch in range(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss and accuracy
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        valid_loss = 0.0
        valid_total = 0
        valid_correct = 0

        # train the model
        model.train()
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)
            # count number of correct labels
            _, preds_tensor = torch.max(output, 1)
            train_total += target.size(0)
            train_correct += (preds_tensor == target).sum().item()

        # validate the model
        model.eval()
        for data, target in validloader:
            data, target = data.to(device), target.to(device)

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item() * data.size(0)
            # count number of correct labels
            _, preds_tensor = torch.max(output, 1)
            valid_total += target.size(0)
            valid_correct += (preds_tensor == target).sum().item()

        # calculate average losses
        train_loss = train_loss / len(trainloader.dataset)
        valid_loss = valid_loss / len(validloader.dataset)
        # calculate accuracy
        train_acc = train_correct / train_total
        valid_acc = valid_correct / valid_total

        # update ABEJA statisctics
        train_client = TrainClient()
        statistics = ABEJAStatistics(num_epochs=n_epochs, epoch=epoch)
        statistics.add_stage(name=ABEJAStatistics.STAGE_TRAIN,
                             accuracy=train_acc,
                             loss=train_loss)
        statistics.add_stage(name=ABEJAStatistics.STAGE_VALIDATION,
                             accuracy=valid_acc,
                             loss=valid_loss)
        train_client.update_statistics(statistics)

        # print training/validation statistics
        print(
            'Epoch: {} \tTrain loss: {:.6f} \tTrain acc: {:.6f} \tValid loss: {:.6f} \tValid acc: {:.6f}'.format(
                epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc
            ))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model.'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
    # return trained model
    return model


def handler(context):
    # set alias specified in console
    dataset_alias = context.datasets
    dataset_id = dataset_alias['train']

    # get dataset via ABEJA Platform api
    dataset_client = DatasetClient()
    dataset = dataset_client.get_dataset(dataset_id)
    dataset_list = list(load_dataset_from_api(dataset))
    num_classes = len(dataset.props['categories'][0]['labels'])
    print('number of classes is {}.'.format(num_classes))

    # create dataloader
    trainloader, validloader = load_split_train_test(dataset_list)

    # specify model architecture (ResNet-50)
    model = models.resnet50(pretrained=True)

    # freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # replace the last fully connected layer with a Linnear layer with no. of classes out features
    model.fc = nn.Linear(2048, num_classes)
    model = model.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # train the model
    train_model(trainloader, validloader, model, optimizer, criterion)
