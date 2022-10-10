#!/usr/bin/env python
# coding: utf-8
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from multiprocessing import freeze_support
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np


# 하이퍼파라미터
EPOCH = 1
BATCH_SIZE = 64
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
threshold = 0.01


# MNIST 데이터셋
trainset = datasets.MNIST(
    root      = './.data',
    train     = True,
    download  = True,
    transform = transforms.ToTensor()
)
testset = datasets.MNIST(root='./.data',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 2,
    drop_last = True
)

test_loader = torch.utils.data.DataLoader(
    dataset     = testset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 2,
    drop_last = True
)

#AutoEncoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # 픽셀당 0과 1 사이로 값을 출력합니다
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = Autoencoder().to(DEVICE)
optimizer_AE = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
criterion = nn.MSELoss()

def AE_train(autoencoder, train_loader):
    autoencoder.train()
    for step, (x, label) in enumerate(train_loader):
        x = x.view(-1, 28*28).to(DEVICE)
        y = x.view(-1, 28*28).to(DEVICE)
        label = label.to(DEVICE)

        encoded, decoded = autoencoder(x)

        loss = criterion(decoded, y)
        optimizer_AE.zero_grad()
        loss.backward()
        optimizer_AE.step()

    return loss

def AE_inference(autoencoder, test_loader):
    autoencoder.eval()
    for step, (x, label) in enumerate(test_loader):
        x = x.view(-1, 28*28).to(DEVICE)
        y = x.view(-1, 28*28).to(DEVICE)
        label = label.to(DEVICE)

        encoded, decoded = autoencoder(x)
        loss = criterion(decoded, y)

    return loss

# 원본 이미지를 시각화 하기 (첫번째 열)
view_data = trainset.data[:10].view(-1, 28*28) ######################수정해야함
view_data = view_data.type(torch.FloatTensor)/255.

#CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

model     = Net().to(DEVICE)
optimizer_CNN = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def CNN_train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            # 가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy, pred


if __name__=='__main__':
    freeze_support()
    for epoch in range(1, EPOCH+1):
        Train_Loss, Train_decoded = AE_train(autoencoder, train_loader)
        Train_Loss = Train_Loss.item()

        #에폭과 Loss_value 출력
        print("[Epoch {}]".format(epoch))
        print(Train_Loss)
        print("Train_decoded:", Train_decoded)

    inference_Loss, Pred_Value = AE_inference(autoencoder, test_loader, Train_decoded)
    inference_Loss = inference_Loss.item()
    print("Pred_Value:", Pred_Value)
    print("In")

    #threshold보다 Loss_value가 높을 때, 묻기
    if inference_Loss >= threshold:
        #사람에게 물어볼 수 있도록 이미지 출력
        test_x = view_data.to(DEVICE)
        _, decoded_data = autoencoder(test_x)
        f, a = plt.subplots(2, 1, figsize=(5, 5))
        img = np.reshape(view_data.data.numpy()[0], (28, 28))
        a[0].imshow(img, cmap='gray')
        a[0].set_xticks(());a[0].set_yticks(())

        img = np.reshape(decoded_data.to("cpu").data.numpy()[0], (28, 28))
        a[1].imshow(img, cmap='gray')
        a[1].set_xticks(());a[1].set_yticks(())
        plt.show()

        input_label = input("What is it?:")

    for epoch in range(1, EPOCH + 1):
        print(trainset)
        CNN_train(model, train_loader, optimizer_CNN, epoch)
        test_loss, test_accuracy, pred = evaluate(model, test_loader)

        print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
            epoch, test_loss, test_accuracy))

