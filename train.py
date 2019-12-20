from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import sys
import glob
from sklearn.metrics import confusion_matrix


def accuracy_per_class(predict_label, true_label, classes):
    '''
    :param predict_label: output of model (matrix)
    :param true_label: labels from dataset (array of integers)
    :param classes: class labels list()
    :return:
    '''
    from numpy import sum, float, array
    if isinstance(classes, int):
        nclass = classes
        classes = range(nclass)
    else:
        nclass = len(classes)

    acc_per_class = []
    for i in range(nclass):
        idx = true_label == classes[i]
        if idx.sum() != 0:
            acc_per_class.append(sum(true_label[idx] == predict_label[idx]) / float(idx.sum()))
    if len(acc_per_class) == 0:
        return 0.

    print(confusion_matrix(true_label, predict_label))

    return array(acc_per_class).mean()


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def load_data(data_dir, settings):
    # Data augmentation and normalization for training
    # Just normalization for validation
    # settings have 2 keys: 'input_size' and 'batch_size'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(settings['input_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(settings['input_size']),
            transforms.CenterCrop(settings['input_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    weights = make_weights_for_balanced_classes(image_datasets['train'].imgs, len(image_datasets['train'].classes))
    weights = torch.DoubleTensor(weights)
    weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # Create training and validation dataloaders
    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=settings['batch_size'], shuffle=False, sampler=weighted_sampler,
                                               num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=settings['batch_size'], shuffle=False, num_workers=8)
    dataloaders_dict = {'train': train_loader, 'val': val_loader}
    #dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=settings['batch_size'], shuffle=True, num_workers=4) for x in ['train', 'val']}

    return dataloaders_dict, image_datasets

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def define_optim_loss(model_ft, feature_extract):
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()



    return optimizer_ft, criterion

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(model, dataloaders, image_datasets, criterion, optimizer, device, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        val_pred = np.array([], dtype=np.int_)
        val_true = np.array([], dtype=np.int_)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'val':
                    val_pred = np.concatenate((val_pred, preds.cpu().numpy()), axis=0)
                    val_true = np.concatenate((val_true, labels.data.cpu().numpy()), axis=0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'val':
                epoch_acc_cls = accuracy_per_class(val_pred, val_true, classes=len(image_datasets['train'].classes))

            if phase == 'val':
                print('{} Loss: {:.4f} Acc: {:.4f} ClsAcc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_acc_cls))
            else:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            sys.stdout.flush()

            # deep copy the model
            if phase == 'val' and epoch_acc_cls > best_acc:
                best_acc = epoch_acc_cls
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_index", type=int, default=0)
    args = parser.parse_args()


    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "../data/aml_train_val"
    #data_dir = '../data/hymenoptera_data'

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet101"

    # Number of classes in the dataset
    num_classes = len(glob.glob(data_dir + '/train/*'))

    # Batch size for training (change depending on how much memory you have)
    batch_size = 32

    # Number of epochs to train for
    num_epochs = 150

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True



    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Data loader
    dataloaders_dict, image_datasets = load_data(data_dir, settings={'input_size': input_size, 'batch_size': batch_size})
    print(image_datasets['train'].classes)


    
    # Detect if we have a GPU available
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ft = model_ft.to(device)



    # optim and loss
    optimizer_ft, criterion = define_optim_loss(model_ft, feature_extract)
    
    
    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, image_datasets, criterion, optimizer_ft, device,
                                num_epochs=num_epochs, is_inception=(model_name=="inception"))

if __name__== "__main__":
    main()
