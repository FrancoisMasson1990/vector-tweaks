import torch

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from embed_tune.training.loader import LazyParquetDataset
from embed_tune.training.model import EmbeddingAlign
from embed_tune.training.train import train


def main() -> None:
    emb_dim = 1536
    num_epochs = 10
    learning_rate = 1e-3
    margin = 0.2
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingAlign(emb_dim=emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset = LazyParquetDataset("../results/triplet_dataset.parquet")

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, lengths=[train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train(model, optimizer, train_loader, val_loader, device, margin=margin, num_epochs=num_epochs)


if __name__ == "__main__":
    main()
