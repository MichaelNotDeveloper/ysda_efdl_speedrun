import torch
from dataset import get_train_data
from torch import nn
from tqdm.auto import tqdm
from unet import Unet

# учитвыая задачу мы не хотим домножать на конастану после N успешных прогонов
# Поэтому прийдется делать это жадно


class LossScaler:
    def __init__(self):
        pass

    def scale(self, loss):
        pass

    def step(self, optimizer):
        pass

    def update(self):
        pass


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: LossScaler,
) -> None:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(
            f"Loss: {round(loss.item(), 4)} Accuracy: {round(accuracy.item() * 100, 4)}"
        )


def train(scaler: LossScaler):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(
            train_loader, model, criterion, optimizer, device=device, scaler=scaler
        )
