import torch
from torch import nn, optim
from tqdm import tqdm


def train_one_epoch(
    model, train_loader, device, epoch, optimizer, criterion, scheduler=None
):
    model.train()
    num_correct = 0
    num_data = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, data in pbar:
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        num_correct += (predicted == labels).sum().item()
        num_data += labels.size(0)

        lr = optimizer.param_groups[0]["lr"]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        accumulate_accuracy = num_correct / num_data * 100
        pbar.set_description(
            f"[Training Epoch {epoch}] LR: {lr:.6f}, Loss: {loss:.4f}, Accuracy: {accumulate_accuracy:.4f}"
        )


def train_model(model, train_loader, config):
    for parameter in model.pretrained_model.parameters():
        parameter.requires_grad_(False)

    for parameter in model.pretrained_model.head.parameters():
        parameter.requires_grad_(True)

    model.to(config["device"])

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config["base_lr"]
    )

    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer=optimizer,
        base_lr=config["base_lr"],
        max_lr=config["max_lr"],
        step_size_up=4 * len(train_loader.dataset) // config["batch_size"],
        cycle_momentum=False,
    )

    for epoch in range(1, config["num_epochs"] + 1):
        train_one_epoch(
            model,
            train_loader,
            config["device"],
            epoch,
            optimizer,
            criterion,
            scheduler,
        )
