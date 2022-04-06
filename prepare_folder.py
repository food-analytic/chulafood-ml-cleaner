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
        if os.path.isdir(os.path.join(data_path, path)) and path not in ignored_folders
    ]

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for food_name in food_names:
        os.rename(os.path.join(data_path, food_name), os.path.join(test_path, food_name))
        os.makedirs(os.path.join(train_path, food_name), exist_ok=True)
