import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from datasets import get_normalize_layer
from torch.nn.functional import interpolate
from nnverify.training.models import resnet_cifar
from nnverify.training.models import mobilenetv2

ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110"]

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif arch == "mobilenetv2":
        model = mobilenetv2().cuda()
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
