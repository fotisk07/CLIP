import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms.v2 as v2
import time
from pathlib import Path

#############################
#### Available Models #######
#############################


class SuperSimpleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 10)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input, labels=None, loss=False):
        input = input.flatten(1)
        input = self.fc1(input)

        if not loss:
            return input

        loss = self.loss_func(input, labels)
        return loss, input

    def inference(self, images: torch.Tensor, deterministic=True):
        logits = self(images, loss=False)  # [B, 10]
        probabs = nn.functional.softmax(logits, dim=1)  # [B, 10]

        if deterministic:
            return torch.max(probabs, dim=-1)[1]
        else:
            return torch.multinomial(probabs, 1, replacement=True)


############################
### Evaluation Function#####
############################
@torch.no_grad()
def calculate_accuracy(model, dataloader, dataset_size):
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        predicted_labels = model.inference(images)
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

    model.train()
    return correct / total


@torch.no_grad()
def validate(model, dataloader, transforms):
    model.eval()
    val_loss = 0
    for i, sample in enumerate(dataloader):
        images, labels = transforms(sample)
        loss, logits = model(images, labels, loss=True)
        val_loss += loss.item()
    model.train()
    return val_loss / len(dataloader)


############################
###### Train Function ######
############################
def train_model(
    model,
    optimizer,
    train_dataloader,
    transforms,
    epochs,
    max_norm,
    val_dataloader=None,
    validate_every=False,
    scheduler=None,
    save_every=False,
):
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0
        num_batches = len(train_dataloader)

        for i, sample in enumerate(train_dataloader):
            images, labels = transforms(sample)
            loss, _ = model(images, labels, loss=True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0 or i == num_batches - 1:
                print(
                    f"Epoch {epoch+1}/{epochs} | Step {i}/{num_batches} | Loss: {loss.item():.4f}",
                )

        if scheduler:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
        else:
            lr = optimizer.param_groups[-1]["lr"]

        elapsed_time = time.time() - start_time
        avg_loss = epoch_loss / num_batches

        if save_every and epoch % save_every == 0:
            save_dir = Path(f"models/{model.__class__.__name__}")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / Path(f"{epoch}.pt")
            torch.save(model.state_dict(), save_path)

        print("\n------------------")
        print(f"Epoch {epoch+1}/{epochs} Completed")
        print(f"Avg Loss: {avg_loss:.4f} | LR: {lr:.6f} | Time: {elapsed_time:.2f}s")
        if val_dataloader and validate_every and epoch % validate_every == 0:
            val_loss = validate(model, val_dataloader, transforms)
            print(f"Validation Loss : {val_loss:.4f}")
        if save_every and epoch % save_every == 0:
            print(f"Model checkpoint saved at {save_path}")
        print("---------------------\n")


############################
#### Training Stuff ########
############################

batch_size = 64
epochs = 2
lr = 1e-3
weight_decay = 1e-2
max_norm = 1
save_every = 1
validate_every = 1
training_split = 0.8

if __name__ == "__main__":
    full_dataset = MNIST(
        root=".",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    test_dataset = MNIST(
        root=".",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    train_size = int(training_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])
    model = SuperSimpleBase()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    accuracy = calculate_accuracy(model, test_dataloader, len(test_dataset))
    print("Accuracy at init", accuracy)

    train_model(
        model,
        optimizer,
        train_dataloader,
        transforms,
        epochs,
        max_norm,
        val_dataloader=val_dataloader,
        validate_every=1,
        save_every=save_every,
    )

    accuracy = calculate_accuracy(model, test_dataloader, len(test_dataset))

    print("------------------------")
    print(f"Trained for {epochs} epochs")
    print(f"Accuracy is {accuracy:.4f}")
    print("-------------------------")
