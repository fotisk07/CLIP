import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms.v2 as v2
import tiktoken

import time


class WrappedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)

        if len(image.shape) == 3:
            image.unsqueeze(0)

        label = f"A photo of the number {label}"

        return image, label

    def __len__(self):
        return len(self.dataset)  # Preserve length


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim_text: int):
        super().__init__()
        self.embds = nn.Embedding(vocab_size, hidden_dim_text)
        self.fc1 = nn.Linear(hidden_dim_text, hidden_dim_text)

    def forward(self, input: torch.Tensor):
        input = self.embds(input)
        input = self.fc1(input)

        return input[:, -1, :]  # return last token calculations


class ImageEncoder(nn.Module):
    def __init__(self, im_size: int, hidden_dim_image: int):
        super().__init__()
        self.fcn1 = nn.Linear(im_size, hidden_dim_image)

    def forward(self, input: torch.Tensor):
        input = input.flatten(1)
        return self.fcn1(input)


class model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        im_size: int,
        hidden_dim_text: int,
        hidden_dim_im: int,
        shared_dim: int,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, hidden_dim_text)
        self.image_encoder = ImageEncoder(im_size, hidden_dim_im)

        self.connector_text = nn.Linear(hidden_dim_text, shared_dim)
        self.connector_image = nn.Linear(hidden_dim_im, shared_dim)

        self.tau = nn.Parameter(torch.tensor(0.7))

    def encode_texts(self, text_tokens: torch.Tensor) -> torch.Tensor:
        assert (
            len(text_tokens.shape) == 2
        ), f"{text_tokens.shape}"  # [B, context_lenght]

        return self.text_encoder(text_tokens)  # [B, HiddenDimText]

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4  # [B, C, H, W]

        return self.image_encoder(images)  # [B, HiddenDimImage]

    def compute_loss(self, logits):
        labels = torch.arange(logits.shape[0])
        loss_i = nn.functional.cross_entropy(logits, labels)
        loss_text = nn.functional.cross_entropy(logits.T, labels)
        loss = (loss_i + loss_text) / 2

        return loss

    def forward(self, text_tokens, images, loss=False):
        text_embds = self.encode_texts(text_tokens)
        im_embds = self.encode_images(images)

        transformed_text_emnds = self.connector_text(text_embds)  # [B, SharedDim]
        transformed_im_emnds = self.connector_image(im_embds)  # [B, SharedDim]

        # Compute cosine similarity as the dot product
        logits = (
            nn.functional.normalize(transformed_text_emnds, dim=1)
            @ nn.functional.normalize(transformed_im_emnds, dim=1).T
        ) * torch.exp(self.tau)

        if not loss:
            return logits

        loss = self.compute_loss(logits)

        return loss, logits


transforms = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
    ]
)


## train params ##
batch_size = 16
epochs = 3

## model params##
vocab_size = 100277
im_size = 28 * 28
hidden_dim_text = 20
hidden_dim_im = 30
shared_dim = 50

dataset = MNIST(
    root=".",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)
dataset = WrappedDataset(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size)

clip = model(vocab_size, im_size, hidden_dim_text, hidden_dim_im, shared_dim)
enc = tiktoken.get_encoding("cl100k_base")

# Use AdamW for decoupled weight decay
optimizer = optim.AdamW(clip.parameters(), lr=1e-3, weight_decay=1e-2)

# Cosine learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Training Loop
for epoch in range(epochs):
    start_time = time.time()
    epoch_loss = 0
    num_batches = len(dataloader)

    for i, (image, text) in enumerate(dataloader):
        text = enc.encode_batch(list(text))
        text = torch.tensor(text, dtype=torch.long)
        image = transforms(image)

        loss, logits = clip(text, image, loss=True)
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(clip.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item()

        # Print loss every 100 steps
        if i % 100 == 0 or i == num_batches - 1:
            print(
                f"[Epoch {epoch+1}/{epochs}] Step {i}/{num_batches} | Loss: {loss.item():.4f}"
            )

    # Update learning rate scheduler
    scheduler.step()

    # Print epoch summary
    elapsed_time = time.time() - start_time
    avg_loss = epoch_loss / num_batches

    print("\n------")
    print(f"Epoch {epoch+1}/{epochs} Completed")
    print(
        f"Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed_time:.2f}s"
    )
    print("------\n")
