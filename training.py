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