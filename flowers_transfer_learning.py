import numpy as np
import matplotlib.pyplot as plt
import time
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict


def validate_model(model, criterion, data_loader):
    model.eval()

    accuracy = 0
    test_loss = 0
    for inputs, labels in iter(data_loader):
        if use_gpu:
            inputs = Variable(inputs.float().cuda(), volatile=True)
            labels = Variable(labels.long().cuda(), volatile=True) 
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels, volatile=True)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).data[0]

        ps = torch.exp(output).data
 
        equality = (labels.data == ps.max(1)[1])

        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss/len(data_loader), accuracy/len(data_loader)

def train_model(model, criterion, optimizer, epochs, training_data_loader, validation_data_loader):
    model.train()

    print_every = 40
    steps = 0

    for epoch in range(epochs):
        running_loss = 0

        for inputs, labels in iter(training_data_loader):
            steps += 1

            if use_gpu:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda()) 
            else:
                inputs = Variable(inputs)
                labels = Variable(labels) 

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            if steps % print_every == 0:
                test_loss, accuracy = validate_model(model, criterion, validation_data_loader)

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                        "Training Loss: {:.3f} ".format(running_loss/print_every),
                        "Test Loss: {:.3f} ".format(test_loss),
                        "Test Accuracy: {:.3f}".format(accuracy))

                running_loss = 0
                model.train()
                
def get_class_name(label):
    c = list(model.class_to_idx.keys())[list(model.class_to_idx.values()).index(label)]

    return cat_to_name[str(c)]

def display_results(model, data_loader):
    columns = 4
    rows = 4
    max_images = columns * rows
    image_count = 0
    fig = plt.figure(figsize=(24, 24))

    for inputs, labels in iter(data_loader):

        if use_gpu:
            var_inputs = Variable(inputs.float().cuda(), volatile=True)
        else:       
            var_inputs = Variable(inputs, volatile=True)

        output = model.forward(var_inputs)
        ps = torch.exp(output).data

        for batch_index in range(0, len(labels)):    
            if image_count < max_images: 
                image = inputs[batch_index].numpy().transpose((1, 2, 0))
                
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = std * image + mean
                image = np.clip(image, 0, 1)

                fig.add_subplot(rows, columns, image_count + 1)
                prob, prob_index = torch.max(ps[batch_index], 0)
                highest_prob = prob.cpu().numpy()[0] if use_gpu else prob.numpy()[0]
                highest_prob_index = prob_index.cpu().numpy()[0] if use_gpu else prob_index.numpy()[0]

                pred_title = get_class_name(highest_prob_index)
                actual_title = get_class_name(labels[batch_index])

                plt.title("{} : {} ({:.3f})".format(actual_title, pred_title, highest_prob))
                plt.axis('off')
                plt.imshow(image)
                image_count += 1

        if image_count >= max_images:
            break

    plt.show()

def save_checkpoint(model, filepath):
    checkpoint = {'state_dict': model.state_dict()}

    torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['state_dict'])

    return model

use_gpu = torch.cuda.is_available()
print("GPU {}".format("Enabled" if use_gpu else "Disabled"))
kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {
    'training' : transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
                                                            
    'validation' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),

    'testing' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
}

image_datasets = {
    'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
    'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
    'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing'])
}

dataloaders = {
    'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True, **kwargs),
    'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True, **kwargs),
    'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=True, **kwargs)
}

enable_training = False

model = models.densenet121(pretrained=True)

model.class_to_idx = image_datasets['training'].class_to_idx

for param in model.parameters():
    param.requires_grad = False

input_size = 224 * 224 * 3
output_size = 102

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters, lr=0.001)
criterion = nn.NLLLoss()
epochs = 10

if use_gpu:
    model.cuda()
    criterion.cuda()

if enable_training:
    train_model(model, criterion, optimizer, epochs, dataloaders['training'], dataloaders['validation'])
    save_checkpoint(model, 'checkpoint.pth')

if not enable_training:
    model = load_checkpoint('checkpoint.pth')

test_loss, accuracy = validate_model(model, criterion, dataloaders['testing'])
print("Validation Accuracy: {:.3f}".format(accuracy))

display_results(model, dataloaders['testing'])
