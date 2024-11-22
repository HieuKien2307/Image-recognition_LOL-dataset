import torch

def accuracy(y_pred, labels):
    with torch.no_grad():
        batch_size = labels.size(0)
        pred = torch.argmax(y_pred, dim=1)
        correct = pred.eq(labels).float().sum(0)
        acc = correct * 100 / batch_size
    return [acc]