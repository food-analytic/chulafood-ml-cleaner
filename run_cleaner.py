import os
import shutil
import argparse
import warnings

import pandas as pd
import torch

from src.model import ChulaFoodNet
from src.data_loader import get_data_loader
from src.training import train_model
from src.inference import predict_proba


def main(args):
    data_path = os.path.join("data", args.run_name)
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        warnings.warn("CUDA is unavailable. Running on CPU", UserWarning)

    config = {
        "batch_size": args.batch_size,
        "image_size": (args.image_size[0], args.image_size[1]),
        "base_lr": args.base_lr,
        "max_lr": args.max_lr,
        "num_epochs": args.epochs,
        "num_workers": args.num_workers,
        "seed": 42,
        "image_train_path": train_path,
        "image_test_path": test_path,
        "device": device,
    }

    if not args.load_pred:
        train_loader, test_loader = get_data_loader(config)

        model = ChulaFoodNet(len(train_loader.dataset.classes))

        if not args.load_model:
            train_model(model, train_loader, test_loader, config)
            torch.save(
                model.state_dict(), os.path.join("models", f"{args.run_name}.pt")
            )
        else:
            model.load_state_dict(
                torch.load(os.path.join("models", f"{args.run_name}.pt"))
            )

        confidence = predict_proba(model, test_loader, config)
        filenames = [image[0].split("/")[-1] for image in test_loader.dataset.imgs]
        labels = [image[0].split("/")[-2] for image in test_loader.dataset.imgs]

        df_pred = pd.DataFrame(
            {
                "filename": filenames,
                "label": labels,
                "confidence": confidence,
            }
        )

        df_pred.to_csv(
            os.path.join("results", f"{args.run_name}_pred.csv"), index=False
        )

    else:
        df_pred = pd.read_csv(os.path.join("results", f"{args.run_name}_pred.csv"))

    clean_path = os.path.join(data_path, "clean")
    suspected_path = os.path.join(data_path, "suspected")
    trash_path = os.path.join(data_path, "trash")

    os.makedirs(clean_path, exist_ok=True)
    os.makedirs(suspected_path, exist_ok=True)
    os.makedirs(trash_path, exist_ok=True)

    for label in df_pred.label.unique():
        os.makedirs(os.path.join(clean_path, label), exist_ok=True)
        os.makedirs(os.path.join(suspected_path, label), exist_ok=True)
        os.makedirs(os.path.join(trash_path, label), exist_ok=True)

    for filename, label, confidence in zip(
        df_pred["filename"], df_pred["label"], df_pred["confidence"]
    ):
        if confidence >= args.high_threshold:
            shutil.copy(
                os.path.join(test_path, label, filename),
                os.path.join(clean_path, label, filename),
            )
        elif confidence >= args.low_threshold:
            shutil.copy(
                os.path.join(test_path, label, filename),
                os.path.join(suspected_path, label, filename),
            )
        else:
            shutil.copy(
                os.path.join(test_path, label, filename),
                os.path.join(trash_path, label, filename),
            )

    for label in df_pred.label.unique():
        shutil.copytree(
            os.path.join(train_path, label),
            os.path.join(clean_path, label),
            dirs_exist_ok=True,
        )

    df_summary = pd.DataFrame(
        [
            [
                label,
                len(os.listdir(os.path.join(clean_path, label))),
                len(os.listdir(os.path.join(suspected_path, label))),
                len(os.listdir(os.path.join(trash_path, label))),
            ]
            for label in df_pred.label.unique()
        ],
        columns=["label", "clean", "suspected", "trash"],
    )

    df_summary.to_csv(
        os.path.join("results", f"{args.run_name}_summary.csv"), index=False
    )

    print("Done!")
    print(df_summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--run_name", type=str, required=True)

    parser.add_argument("-m", "--load_model", type=bool, default=False)
    parser.add_argument("-p", "--load_pred", type=bool, default=False)

    parser.add_argument("-lt", "--low_threshold", type=float, default=0.5)
    parser.add_argument("-ht", "--high_threshold", type=float, default=0.8)

    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-is", "--image_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("-bl", "--base_lr", type=float, default=1e-4)
    parser.add_argument("-ml", "--max_lr", type=float, default=1e-3)
    parser.add_argument("-nw", "--num_workers", type=int, default=2)

    args = parser.parse_args()

    main(args)
