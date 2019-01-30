import numpy as np
import torch
from torch import nn
from torch import optim
from PIL import Image
from torchvision import datasets, transforms, models

from collections import OrderedDict

### -------------------------
#   Train util functions
### -------------------------

def loading_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Defining transformations
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #Loading datasets with ImageFolder
    image_data = {
        'train': datasets.ImageFolder(train_dir, transform=train_transforms),
        'validation': datasets.ImageFolder(valid_dir, transform=test_transforms),
        'test': datasets.ImageFolder(test_dir, transform=test_transforms)
    }


    #Defining Dataloaders using the image datasets and their transforms
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_data['train'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_data['test'], batch_size=32),
        'validation': torch.utils.data.DataLoader(image_data['validation'], batch_size=32)

    }

    return dataloaders, image_data


def prepare_model(arch, hidden_units, learning_rate, output_size=102):
    if arch.lower() == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch.lower() == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch.lower() == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        raise ValueError("Unrecognized architecture", arch)

    # Obtaining the in_features from the current classifier
    input_features = model.classifier[0].in_features

    # Freezing paramenters
    for param in model.parameters():
        param.requires_grad = False

    # Defining the classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, hidden_units)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(0.5)),
        ('fc3', nn.Linear(hidden_units, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    #print(model)

    return model, classifier, criterion, optimizer


def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():

        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

    print("Testing data:")
    print("Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
          "Test Accuracy: {:.3f}".format(100 * accuracy / len(testloader)))


def saving_model(arch,model,save_dir, train_data, classifier, optimizer, epochs):

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        'arch': arch,
        'model': model,
        'epochs': epochs,
        'classifier': classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer': optimizer.state_dict()
    }

    #print(save_dir + 'checkpoint.pth')
    location = save_dir + 'checkpoint.pth'
    torch.save(checkpoint, location)
    print("Checkpoint saved to: ", location)


### -------------------------
#   Predict util functions
### -------------------------


def loading_checkpoint(filename):

    checkpoint = torch.load(filename, map_location='cpu')

    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']
    model.epochs = checkpoint['epochs']

    #print(model)
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    pixels = 256

    # Resizing
    im = im.resize((pixels, pixels))
    # Cropping
    im = im.crop(((pixels - 224) / 2, (pixels - 224) / 2, ((pixels - 224) / 2) + 224, ((pixels - 224) / 2) + 224))

    # Converting to np array and divinding by 255 color channels to get 0-1
    np_im = np.array(im) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # Normalizing image
    np_im = (np_im - mean) / std

    return np_im.transpose(2, 0, 1)


def map_category_names(filename, classes):

    class_names =  list()

    with open(filename, 'r') as f:
        cat_to_name = json.load(f)

    for k in classes:
        class_names.append(cat_to_name[k])

    return class_names

