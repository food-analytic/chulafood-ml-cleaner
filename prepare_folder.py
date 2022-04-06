import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--run_name", type=str, required=True)
    args = parser.parse_args()

    data_path = os.path.join("data", args.run_name)
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    ignored_folders = ["train", "test", "clean", "suspected", "trash"]

    food_names = [
        path
        for path in os.listdir(data_path)
        if os.path.isdir(path) and path not in ignored_folders
    ]

    os.mkdir(train_path, exist_ok=True)
    os.mkdir(test_path, exist_ok=True)

    for food_name in food_names:
        os.rename(food_name, os.path.join(test_path, food_name))
        os.mkdir(os.path.join(train_path, food_name), exist_ok=True)
