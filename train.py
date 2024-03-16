import torch
import torch.nn as nn
from datamodule import ALKDataModule
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Optional, Any, Callable, Tuple, Dict
from pathlib import Path
from torch import Tensor
from utils import total_absolute_error
import numpy as np
from tqdm import tqdm
from net import NaiveEuclideanGNN


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable[[Tensor, Tensor], Tensor],
    pbar: Optional[Any] = None,
    optim: Optional[torch.optim.Optimizer] = None,
):
    def step(data_batch: Data) -> Tuple[float, float]:
        pred = model.forward(data_batch).float()
        target = data_batch.y.float()
        loss = criterion(pred, target)
        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        batch_mae = total_absolute_error(pred.detach(), target.detach())

        return loss.detach().item(), batch_mae

    if optim is not None:
        model.train()
        model.requires_grad_(True)
    else:
        model.eval()
        model.requires_grad_(False)

    total_loss = 0
    total_mae = 0
    device = next(model.parameters()).device
    idx = 0
    for data in loader:
        data = data[idx].to(device)
        idx += 1
        loss, batch_mae = step(data)
        total_loss += loss * data.num_graphs
        total_mae += batch_mae.sum()

        if pbar is not None:
            pbar.update(1)

    return total_loss / len(loader.dataset), total_mae / len(loader.dataset)



def train_model(
        data_module: ALKDataModule,
        model: nn.Module,
        num_epochs: int = 30,
        lr: float = 3e-4,
        batch_size: int = 32,
        weight_decay: float = 1e-8,
        best_model_path: Path = "./weight/trained_model.pth",
) -> Dict[str, Any]:
    train_loader = data_module.train_loader(batch_size=batch_size)
    val_loader = data_module.val_loader(batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-8)
    loss_fn = nn.MSELoss()

    best_val_mae = float("inf")

    result = {
        "model": model,
        "path_to_best_model": best_model_path,
        "train_loss": np.full(num_epochs, float("nan")),
        "val_loss": np.full(num_epochs, float("nan")),
        "train_mae": np.full(num_epochs, float("nan")),
        "val_mae": np.full(num_epochs, float("nan")),
    }

    def update_statistics(i_epoch: int, **kwargs: float):
        for key, value in kwargs.items():
            result[key][i_epoch] = value

    def desc(i_epoch: int) -> str:
        return " | ".join(
            [f"Epoch {i_epoch + 1:3d} / {num_epochs}"]
            + [
                f"{key}: {value[i_epoch]:8.2f}"
                for key, value in result.items()
                if isinstance(value, np.ndarray)
            ]
        )

    for i_epoch in range(0, num_epochs):
        progress_bar = tqdm(total=len(train_loader) + len(val_loader))
        try:
            progress_bar.set_description(desc(i_epoch))

            train_loss, train_mae = run_epoch(
                model, train_loader, loss_fn, progress_bar, optim
            )
            val_loss, val_mae = run_epoch(model, val_loader, loss_fn, progress_bar)

            update_statistics(
                i_epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_mae=train_mae,
                val_mae=val_mae,
            )

            progress_bar.set_description(desc(i_epoch))

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), best_model_path)
        finally:
            progress_bar.close()

    return result


def test_model(
        model: nn.Module, data_module: ALKDataModule
) -> Tuple[float, Tensor, Tensor]:
    test_mae = 0
    preds, targets = [], []
    loader = data_module.test_loader()
    device = "mps"
    for data in loader:
        data = data.to(device)
        pred = model(data)
        target = data.y
        preds.append(pred)
        targets.append(target)
        test_mae += total_absolute_error(pred, target).item()

    test_mae = test_mae / len(data_module.test_split)

    return test_mae, torch.cat(preds, dim=0), torch.cat(targets, dim=0)


if __name__ == "__main__":
    data_module = ALKDataModule()
    gcn_baseline = NaiveEuclideanGNN(64, 4, 3)
    gcn_train_result = train_model(
        data_module,
        gcn_baseline,
        num_epochs=100,
        lr=3e-4,
        batch_size=64,
    )
