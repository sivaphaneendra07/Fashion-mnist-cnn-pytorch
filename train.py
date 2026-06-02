
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()

])

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=train_transform
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

import matplotlib.pyplot as plt

image, label = train_data[0]

plt.imshow(image.squeeze(), cmap="inferno")
plt.show()
images,labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)

import torch.nn as nn

model = nn.Sequential(nn.Conv2d(1,32,kernel_size = 3,padding = 1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
                      nn.Conv2d(32,64,kernel_size = 3,padding = 1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),

                      nn.Flatten(),#converts into single array
                      nn.Linear(7*7*64,128),nn.BatchNorm1d(128),nn.ReLU(), #7 because we did max pool twice with 2 -> 28/4 = 7
                      nn.Linear(128,10),
                     )
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

for epoch in range(7):
    total_loss = 0
    for  images,labels in train_loader:
      # MOVE DATA TO GPU
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = loss_fn(output,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(avg_loss)
model.eval()

from sklearn.metrics import accuracy_score

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        predicted = torch.argmax(outputs, dim=1)

        all_preds.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )
acc = accuracy_score(all_labels, all_preds)

preds_train = []
labels_train = []
with torch.no_grad():

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs_train = model(images)

        predicted_train = torch.argmax(outputs_train, dim=1)

        preds_train.extend(predicted_train.cpu().numpy())

        labels_train.extend(labels.cpu().numpy())

acc_train = accuracy_score(labels_train, preds_train)

print(acc)
print(acc_train)

torch.save(
    model.state_dict(),
    "fashion_mnist_cnn.pth"
)
