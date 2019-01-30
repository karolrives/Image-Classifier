# Imports

from utils import loading_data, prepare_model, validation, saving_model
import torch
import argparse

def processing_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", help="Specifies the image data location", default="flower_data")
    parser.add_argument("--save_dir", help="Saves the checkpoint file to the specified location", default="./")
    parser.add_argument("--arch", help="Specifies the architecture used", default="vgg16")
    parser.add_argument("--learning_rate", type=float,  help="Specifies the learning rate", default=0.00005)
    parser.add_argument("--hidden_units", type=int, help="Specified the hidden units", default=4096)
    parser.add_argument("--epoch", type=int, help="Specifies the number of epochs", default=20)
    parser.add_argument("--gpu",help="Specifies the use of gpu", action="store_true")

    args = parser.parse_args()

    data_dir = args.data_directory
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epoch
    if args.gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    print(args)
    return data_dir, save_dir, arch, learning_rate, hidden_units, epochs, device

def do_deep_training(model, trainloader, epochs, criterion, optimizer, device):

    print("Starting training ..")
    #Training parameters
    steps = 0
    print_every = 50

    model.to(device)

    for e in range(epochs):
        running_loss = 0
        total = 0
        correct = 0

        model.train()

        for ii, (inputs,labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #Total Loss
            running_loss += loss.item()

            #Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


            if steps % print_every == 0:
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                     "Training Loss: {:.4f}.. ".format(running_loss/print_every),
                     "Accuracy: {:.4f}.. ".format(100 * correct/total))

                running_loss = 0

def main():

    #Processing command line arguments
    data_dir, save_dir, arch, learning_rate, hidden_units, epochs, device = processing_arguments()
    #Loading image data
    dataloaders, image_data= loading_data(data_dir)
    #Defining model, classifier, criterion and optimizer
    model, classifier, criterion, optimizer = prepare_model(arch, hidden_units, learning_rate)
    #Training model
    do_deep_training(model,dataloaders['train'], epochs,criterion,optimizer,device)
    #Obtaining test loss and accuracy
    validation(model,dataloaders['test'],criterion,device)
    #Saving checkpoint
    saving_model(arch,model,save_dir, image_data['train'], classifier, optimizer, epochs)


if __name__ == '__main__':
    main()