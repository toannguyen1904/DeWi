import torch
from dataset import pre_data


def read_dataset(input_size, batch_size, root, dataset_path):
    trainset = pre_data.Dataset(input_size=input_size, root=root, dataset_path=dataset_path, mode='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=8, drop_last=False)

    valset = pre_data.Dataset(input_size=input_size, root=root, dataset_path=dataset_path, mode='val')
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                  shuffle=False, num_workers=8, drop_last=False)

    testset = pre_data.Dataset(input_size=input_size, root=root, dataset_path=dataset_path, mode='test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)

    return trainloader, valloader, testloader
