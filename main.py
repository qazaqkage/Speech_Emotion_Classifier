import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
from Loader import EmotionDataset
from torch.utils.data import DataLoader
from NN import model, criterion, optimizer
from dataset import*


num_epochs = 50
num_classes = 2
batch_size = 16
learning_rate = 0.000001

audio_file_dict = creating_df()

X_train, X_test = train_test_split(audio_file_dict,test_size=0.3)

train_data = EmotionDataset(audio_file_dict=X_train)
test_data = EmotionDataset(audio_file_dict=X_test)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    loss_list = []
    acc_list = []
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)
    print(f'epoch: {epoch}: acc:',np.mean(acc_list),'loss: ',np.mean(loss_list))

preds = []
outcome = []
labs = []
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        labs.append(labels)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        preds.append(predicted)
        c = (predicted == labels).squeeze()
        outcome.append(c)

outcome = torch.stack(outcome).view(-1).cpu().numpy()

print('Accuracy on test set after 50 epochs: ',100*round(outcome.sum()/len(outcome),2),'%')