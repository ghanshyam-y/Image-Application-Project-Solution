import argparse
from time import time
import torch
from torchvision import datasets, transforms
import utility
import model_helper
import os


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='Set directory to training images')

    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Set directory to save checkpoints')

    parser.add_argument('--arch', dest='arch', default='vgg16', action='store',
                        choices=['vgg16', 'densenet121'], help='Model architecture to use for training')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Set learning rate hyperparameter')

    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Set number of hidden units hyperparameter')

    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs used to train model')

    parser.add_argument('--gpu', dest='gpu',
                        action='store_true', help='Use GPU for training')
    parser.set_defaults(gpu=False)

    return parser.parse_args()


def main():
    start_time = time()

    in_args = get_input_args()

    use_gpu = torch.cuda.is_available() and in_args.gpu

    print("Training on {} using {}".format(
        "GPU" if use_gpu else "CPU", in_args.arch))

    print("Learning rate:{}, Hidden Units:{}, Epochs:{}".format(
        in_args.learning_rate, in_args.hidden_units, in_args.epochs))

    if not os.path.exists(in_args.save_dir):
        os.makedirs(in_args.save_dir)

    training_dir = in_args.data_dir + '/train'
    validation_dir = in_args.data_dir + '/valid'
    testing_dir = in_args.data_dir + '/test'

    data_transforms = {
        'training' : transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])]),

        'validation' : transforms.Compose([transforms.Scale(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),

        'testing' : transforms.Compose([transforms.Scale(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    }

    dirs = {'training': training_dir, 'validation': validation_dir, 'testing': testing_dir}

    image_datasets = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x])
                  for x in ['training', 'validation', 'testing']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                              shuffle=True)
               for x in ['training', 'validation', 'testing']}

    model, optimizer, criterion = model_helper.create_model(in_args.arch,
                                                            in_args.hidden_units,
                                                            in_args.learning_rate,
                                                            image_datasets['training'].class_to_idx)

    if use_gpu:
        model.cuda()
        criterion.cuda()

    model_helper.train(model,
                       criterion,
                       optimizer,
                       in_args.epochs,
                       dataloaders['training'],
                       dataloaders['validation'],
                       use_gpu)

    file_path = in_args.save_dir + '/' + in_args.arch + \
        '_epoch' + str(in_args.epochs) + '.pth'

    model_helper.save_checkpoint(file_path,
                                 model,
                                 optimizer,
                                 in_args.arch,
                                 in_args.hidden_units,
                                 in_args.epochs)

    test_loss, accuracy = model_helper.validate(
        model, criterion, dataloaders['testing'], use_gpu)
    print("Post load Validation Accuracy: {:.3f}".format(accuracy))
    image_path = 'flowers/test/28/image_05230.jpg'
    print("Predication for: {}".format(image_path))
    probs, classes = model_helper.predict(image_path, model, use_gpu)
    print(probs)
    print(classes)

    end_time = time()
    utility.print_elapsed_time(end_time - start_time)


if __name__ == "__main__":
    main()
