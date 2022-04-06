import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from timm.data.auto_augment import rand_augment_transform


def get_data_loader(config):
    train_transform = transforms.Compose(
        [
            transforms.Resize(size=config["image_size"]),
            rand_augment_transform(
                config_str="rand-m9-mstd0.5",
                hparams={},
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(size=config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageFolder(
        root=config["image_train_path"], transform=train_transform
    )
    test_dataset = ImageFolder(root=config["image_test_path"], transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    count_per_class = pd.Series(train_dataset.targets).value_counts()
    if len(count_per_class.unique()) > 1:
        print(
            "Found imbalanced training data. The sample size per class of the training set must be equal."
        )
        for label in train_dataset.classes:
            print(f"{label}: {count_per_class[train_dataset.class_to_idx[label]]} images")
        raise RuntimeError(
            "Training data is imbalanced. See the above message for details."
        )

    return train_loader, test_loader
