'''
gets state dict url of models in https://github.com/chenyaofo/pytorch-cifar-models
from https://github.com/chenyaofo/pytorch-cifar-models/releases/
'''

url = {
    'cifar10_vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg11_bn-eaeebf42.pt',
    'cifar100_vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg11_bn-57d0759e.pt',
    'cifar10_vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg19_bn-57191229.pt',
    'cifar100_vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg19_bn-b98f7bd7.pt'
}

def get_state_dict_url(model_name: str, dataset_name: str):
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    return url[dataset_name + '_' + model_name]