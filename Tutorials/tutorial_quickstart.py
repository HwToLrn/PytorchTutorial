import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.device = None
        self.loss_fn = None
        self.optimizer = None
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features=28*28, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=10),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

    def set_device(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def set_optimizer(self, lr):
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=lr)

    def set_loss(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()


def train(dataloader, model, device, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def save_model(model):
    print('Saving PyTorch Model')
    torch.save(obj=model.state_dict(), f='../SavedModel/tutorial_model.pth')
    print("Saved PyTorch Model State to tutorial_model.pth")


def load_model(filename: str):
    model = NeuralNetwork() # 모델을 불러올 때는 같은 class 모델을 가져와야 함
    model.load_state_dict(state_dict=torch.load(f=filename))
    return model


def main():
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 5
    training_data = torchvision.datasets.FashionMNIST(
        root='data', train=True, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    test_data = torchvision.datasets.FashionMNIST(
        root='data', train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )

    train_dataloader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

    for x, y in train_dataloader:
        # N:batch_size, C:channel, H:height, W:width
        print('Shape of X [N, C, H, W]: ', x.shape)
        print('Shape of y: ', y.shape)
        break

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using {} device".format(device))

    model = NeuralNetwork().to(device=device)
    model.set_device()
    model.set_loss()
    model.set_optimizer(lr=LEARNING_RATE)
    print(model)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(dataloader=train_dataloader, model=model, device=model.device, loss_fn=model.loss_fn, optimizer=model.optimizer)
        test(dataloader=train_dataloader, model=model, device=model.device, loss_fn=model.loss_fn)
    print('Done!')

    save_model(model)


if __name__ == '__main__':
    main()