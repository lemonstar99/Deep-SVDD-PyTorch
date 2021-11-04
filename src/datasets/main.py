from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .ct import CT_Dataset
from .ep import EP_Dataset
from .rs import RS_Dataset
from .sad import SAD_Dataset
from .natops import NATOPS_Dataset

def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'ct', 'ep', 'rs', 'sad', 'natops')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'ct':
        dataset = CT_Dataset(root=data_path, normal_class=normal_class)
    
    if dataset_name == 'ep':
        dataset = EP_Dataset(root=data_path, normal_class=normal_class)
    
    if dataset_name == 'rs':
        dataset = RS_Dataset(root=data_path, normal_class=normal_class)
    
    if dataset_name == 'sad':
        dataset = SAD_Dataset(root=data_path, normal_class=normal_class)
    
    if dataset_name == 'natops':
        dataset = NATOPS_Dataset(root=data_path, normal_class=normal_class)

    return dataset
