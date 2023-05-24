import torch
import torch.nn as nn
from torchvision import models, transforms
import load
import multiprocessing
import os

def main():

    model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    amount_ftrs = model.fc.in_features
    model.fc = nn.Linear(amount_ftrs, len(load.class_names))
    device = torch.device('cpu')
    #device = torch.device('cuda')
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


    iterations = 3
    for iteration in range(iterations):
        print(iteration)
        for inputs, labels in load.dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    current_dir, filename = os.getcwd(), 'model.pth'
    save_path = os.path.join(current_dir, filename)
    torch.save(model.state_dict(), save_path) #save model

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in load.dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print((int)(correct/total))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()