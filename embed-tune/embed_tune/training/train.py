import logging

import torch
import torch.nn.functional as F

from torch import Tensor
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)


def triplet_loss(query, pos, neg, margin=0.2) -> Tensor:
    sim_qp = F.cosine_similarity(query, pos, dim=-1)
    sim_qn = F.cosine_similarity(query, neg, dim=-1)
    loss_per_sample = torch.clamp(sim_qn - sim_qp + margin, min=0.0)
    return loss_per_sample.mean()


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    device: torch.device,
    margin: float = 0.2,
    num_epochs: int = 10,
) -> None:
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs} - TRAIN", leave=False)
        for batch in train_bar:
            query, pos, neg = batch
            query, pos, neg = query.to(device), pos.to(device), neg.to(device)

            proj_query = model(query)
            # proj_pos = model(pos)
            # proj_neg = model(neg)
            # We think that learning one linear layer for the query and one for the product might be beneficial
            # As ||M1X - M2Y|| <=> ||M12X - Y|| we only apply the transformation to the query (easier engineering because we don't have to change the os embedding)

            loss = triplet_loss(proj_query, pos, neg, margin=margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_data_loader)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_data_loader, desc=f"Epoch {epoch+1}/{num_epochs} - TRAIN", leave=False)
            for batch in val_bar:
                query, pos, neg = batch
                query, pos, neg = query.to(device), pos.to(device), neg.to(device)

                proj_query = model(query)
                loss = triplet_loss(proj_query, pos, neg, margin=margin)
                val_loss_sum += loss.item()
        avg_val_loss = val_loss_sum / len(val_data_loader)

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                f"../models/models_title_margin_{margin}.pt",
            )
            logging.info(f"--> New best model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")
