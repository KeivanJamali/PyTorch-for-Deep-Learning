"""Trains PyTorch image classification."""

import os
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_epochs", help="how many epoch should we go thorough?", type=int, default=5)
parser.add_argument("-b", "--batch_size", help="what is the batch_size through each epoch?", type=int, default=32)
parser.add_argument("-u", "--hidden_units", help="how many units each layer should have?", type=int, default=10)
parser.add_argument("-l", "--learning_rate", help="what is the order of learning rate?", type=float, default=0.001)
args = parser.parse_args()
num_epochs = args.num_epochs
batch_size = args.batch_size
hidden_units = args.hidden_units
learning_rate = args.learning_rate


train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

device = "cuda" if torch.cuda.is_available() else "cpu"
data_transform = transforms.Compose([transforms.Resize(size=(64, 64)), transforms.ToTensor()])
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir, test_dir, data_transform,
                                                                               batch_size)
model = model_builder.TinyVGG(3, hidden_units, len(class_names)).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
engine.train(model, train_dataloader, test_dataloader, optimizer, loss_fn, num_epochs, device)
utils.save_model(model, "models", "05-going-modular.pth")
