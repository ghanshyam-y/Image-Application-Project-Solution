import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
import utility
from PIL import Image


def get_model_from_arch(arch, hidden_units):
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        classifier_input_size = model.classifier.in_features
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        classifier_input_size = model.classifier[0].in_features

    for param in model.parameters():
        param.requires_grad = False

    classifier_output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, classifier_output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


def create_model(arch, hidden_units, learning_rate, class_to_idx):
    model = get_model_from_arch(arch, hidden_units)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)
    criterion = nn.NLLLoss()

    model.class_to_idx = {class_to_idx[k]: k for k in class_to_idx}

    return model, optimizer, criterion


def save_checkpoint(file_path, model, optimizer, arch, hidden_units, epochs):
    state = {
        'arch': arch,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(state, file_path)

    print("Checkpoint Saved: '{}'".format(file_path))


def load_checkpoint(file_path):
    state = torch.load(file_path)

    model = get_model_from_arch(state['arch'], state['hidden_units'])

    model.load_state_dict(state['state_dict'])
    model.class_to_idx = state['class_to_idx']

    print("Checkpoint Loaded: '{}' (arch={}, hidden_units={}, epochs={})".format(
        file_path, state['arch'], state['hidden_units'], state['epochs']))

    return model


def validate(model, criterion, data_loader, use_gpu):
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


def train(model,
          criterion,
          optimizer,
          epochs,
          training_data_loader,
          validation_data_loader,
          use_gpu):
    model.train()

    print_every = 10
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
                test_loss, accuracy = validate(model,
                                               criterion,
                                               validation_data_loader,
                                               use_gpu)

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                      "Training Loss: {:.3f} ".format(
                          running_loss/print_every),
                      "Validation Loss: {:.3f} ".format(test_loss),
                      "Validation Accuracy: {:.3f}".format(accuracy))

                running_loss = 0
                model.train()


def predict(image_path, model, use_gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()

    image = Image.open(image_path)
    np_array = utility.process_image(image)
    tensor = torch.from_numpy(np_array)

    if use_gpu:
        var_inputs = Variable(tensor.float().cuda(), volatile=True)
    else:
        var_inputs = Variable(tensor, volatile=True)

    var_inputs = var_inputs.unsqueeze(0)

    output = model.forward(var_inputs)
    ps = torch.exp(output).data.topk(topk)
    probs = ps[0].cpu() if use_gpu else ps[0]
    classes = ps[1].cpu() if use_gpu else ps[1]

    mapped_classes = list()
    for label in classes.numpy()[0]:
        mapped_classes.append(model.class_to_idx[label])

    return probs.numpy()[0], mapped_classes
