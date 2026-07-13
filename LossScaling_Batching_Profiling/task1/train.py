import torch
from dataset import get_train_data
from torch import nn
from tqdm.auto import tqdm
from unet import Unet


class LossScaler:
    MAX_VALUE = 65504
    def __init__(self, scaler_type : str | None = None, interval : float = 0.2, window : int = 5, scaling_mult: float = 1.1):
        self.scaler_type = scaler_type
        if scaler_type is None:
            self.scaling_factor = 1.
        elif scaler_type in ["static", "dynamic"]:
            self.scaling_factor = self.MAX_VALUE / interval
        else:
            raise ValueError("No such scaler type dumbass")
        self.window = window
        self.scaling_counter = 0
        self.scaling_mult = scaling_mult
        self.unstable_scaling = False

    def scale(self, loss : torch.Tensor) -> torch.Tensor:
        return loss.mul(self.scaling_factor)
    @torch.no_grad()
    def step(self, optimizer):
        max_stat = 0.
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad
                    if not torch.isfinite(grad).all():
                        self.unstable_scaling = True
                        print("Inf in grad!")
                        return
                    grad.mul_(self.scaling_factor ** -1)
                    if self.scaler_type == "dynamic":
                        max_stat = max(max_stat, grad.abs().max().item())
        if not self.unstable_scaling:
            optimizer.step()
                    
    def update(self):
        if self.unstable_scaling:
            self.unstable_scaling = False
            self.scaling_factor /= self.scaling_mult
            self.scaling_counter = 0
            return
        if self.scaler_type != "dynamic":
            return
        self.scaling_counter += 1
        if self.scaling_counter == self.window:
            self.scaling_factor *= self.scaling_mult
            self.scaling_counter = 0
            


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
        optimizer.zero_grad()

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
