import os
import torch
from collections import OrderedDict


def auto_load_resume(model, optimizer, scheduler, path, status, device):
    if status == 'train':
        current_model_path = os.path.join(path, 'current_model.pth')
        best_model_path = os.path.join(path, 'best_model.pth')
        if not os.path.exists(current_model_path):
            return 0, 0.0
        else:
            print('Loading pretrained model!')
            checkpoint = torch.load(current_model_path, map_location=device)
            best_model = torch.load(best_model_path, map_location=device)
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[:]        #  = k[7:] if you used DataParallel
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model = model.to(device)
            epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print('Resume from %s' % 'epoch' + str(epoch))
            return epoch, best_model['val_acc']
    elif status == 'test':
        print('Loading pretrained model!')
        checkpoint = torch.load(path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[:]        #  = k[7:] if you used DataParallel
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        epoch = checkpoint['epoch']
        print('Resume from %s' % 'epoch' + str(epoch))
        return epoch
