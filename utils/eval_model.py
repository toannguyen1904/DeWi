import torch
from tqdm import tqdm


def eval(model, device, loader, metric_loss, miner, criterion, split):
    model.eval()
    print('Evaluating model on ' + split + ' data')

    ce_loss_sum = 0
    metric_loss_sum = 0
    correct = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            p, logits = model(images)
            pminer = miner(p, labels)
            p_mloss = metric_loss(p, labels, pminer)
            ce_loss = criterion(logits, labels)

            ce_loss_sum += ce_loss.item()
            metric_loss_sum += p_mloss.item()

            pred = logits.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    loss_avg = ce_loss_sum / (i+1)
    metric_loss_avg = metric_loss_sum / (i+1)

    accuracy = correct / len(loader.dataset)

    return loss_avg, metric_loss_avg, accuracy