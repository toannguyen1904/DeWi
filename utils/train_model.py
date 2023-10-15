import os
import torch
from tqdm import tqdm
from utils.eval_model import eval
from torch.autograd import Variable
from utils.mixup_utils import mixup_data, mixup_criterion


def train(model,
          device,
          trainloader,
          valloader,
          testloader,
          metric_loss,
          miner,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          best_val_acc):
    best_acc = best_val_acc
    for epoch in range(start_epoch + 1, end_epoch + 1):
        f = open(os.path.join(save_path, 'log.txt'), 'a')
        model.train()
        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']
        turn = True
        for _, data in enumerate(tqdm(trainloader)):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if turn:
                p, logits = model(images)
                pminer = miner(p, labels)
                p_mloss = metric_loss(p, labels, pminer)
                ce_loss = criterion(logits, labels)
                total_loss = ce_loss + p_mloss
            else:
                images, labels_a, labels_b, lam = mixup_data(images, labels, 1.0, True)
                images, labels_a, labels_b = map(Variable, (images, labels_a, labels_b))
                _, logits = model(images)
                ce_loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                total_loss = ce_loss
            turn = not turn

            total_loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        f.write('\nEPOCH' + str(epoch) + '\n')
        # eval valset
        val_loss_avg, val_metric_loss_avg, val_accuracy = eval(model, device, valloader, metric_loss, miner, criterion, split='val')
        print('Validation set: Avg Val CE Loss: {:.4f}; Avg Val Metric Loss: {:.4f}; Val accuracy: {:.2f}%'.format(val_loss_avg, val_metric_loss_avg, 100. * val_accuracy))
        f.write('Validation set: Avg Val CE Loss: {:.4f}; Avg Val Metric Loss: {:.4f}; Val accuracy: {:.2f}% \n'.format(val_loss_avg, val_metric_loss_avg, 100. * val_accuracy))
        # eval testset
        test_loss_avg, test_metric_loss_avg, test_accuracy = eval(model, device, testloader, metric_loss, miner, criterion, split='test')
        print('Test set: Avg Test CE Loss: {:.4f}; Avg Test Metric Loss: {:.4f}; Test accuracy: {:.2f}%'.format(test_loss_avg, test_metric_loss_avg, 100. * test_accuracy))
        f.write('Test set: Avg Test CE Loss: {:.4f}; Avg Test Metric Loss: {:.4f}; Test accuracy: {:.2f}%'.format(test_loss_avg, test_metric_loss_avg, 100. * test_accuracy))
        
        # save checkpoint
        print('Saving checkpoint')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'learning_rate': lr,
            'val_acc': val_accuracy,
            'test_acc': test_accuracy
        }, os.path.join(save_path, 'current_model' + '.pth'))

        if val_accuracy > best_acc:
            print('Saving best model')
            f.write('\nSaving best model!\n')
            best_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
                'val_acc': val_accuracy,
                'test_acc': test_accuracy
            }, os.path.join(save_path, 'best_model' + '.pth'))
        f.close()