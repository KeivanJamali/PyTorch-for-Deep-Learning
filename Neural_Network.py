import torch
from torch import nn
from matplotlib import pyplot as plt


class Neural_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


def plot_predictions(train=None, test=None, prediction=None):
    plt.figure(figsize=(5, 3.5))
    if train:
        plt.scatter(train[0], train[1], c="b", s=5, label="Train data")
    if test:
        plt.scatter(test[0], test[1], c="g", s=5, label="Test data")
    if prediction is not None:
        plt.scatter(test[0], prediction, c="r", s=5, label="Prediction")

    plt.legend(prop={"size": 10})


def flow(model: Neural_Net, train: list, test: list, epochs: int, learning_rate=0.01):
    epochs_count = []
    loss_values = []
    test_loss_values = []
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        y_pred = model(train[0])
        loss = loss_fn(y_pred, train[1])
        # print(f"loss in epoch {epoch} is : {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        # print(model.state_dict())
        with torch.inference_mode():
            test_pred = model(test[0])
            test_loss = loss_fn(test_pred, test[1])
            if epoch % 10 == 0 and epoch > 0:
                epochs_count.append(epoch)
                # loss_values.append(torch.tensor(loss).numpy())
                # test_loss_values.append(torch.tensor(test_loss).numpy())
                loss_values.append(loss.item())
                test_loss_values.append(test_loss.item())
                print(f"Epoch:{epoch} | Loss:{loss:.4f} | Test_loss:{test_loss:.4f}")

    return model, [epochs_count, loss_values, test_loss_values]


def test(model: Neural_Net, X_test: torch.Tensor, y_test: torch.Tensor):
    loss_fn = nn.L1Loss()
    with torch.inference_mode():
        y_pred = model(X_test)
        loss = loss_fn(y_pred, y_test)
        print(f"test loss is : {loss}")
        plot_predictions(test=[X_test, y_test], prediction=y_pred)
        print()
