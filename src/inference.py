import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def predict_proba(model, test_loader, config):
    model.eval()
    confidence = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for batch_idx, data in pbar:
            inputs, labels = data[0].to(config["device"]), data[1].to(config["device"])
            outputs = model(inputs)
            pred_proba = nn.functional.softmax(outputs, dim=1)

            pred_proba = pred_proba.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            pred_proba = pred_proba[range(len(pred_proba)), labels]
            confidence.append(pred_proba)

    confidence = np.concatenate(confidence)
    return confidence
