from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .ct_LeNet import CT_LeNet, CT_LeNet_Autoencoder
from .ep_LeNet import EP_LeNet, EP_LeNet_Autoencoder
from .rs_LeNet import RS_LeNet, RS_LeNet_Autoencoder
from .sad_LeNet import SAD_LeNet, SAD_LeNet_Autoencoder
from .natops_LeNet import NATOPS_LeNet, NATOPS_LeNet_Autoencoder


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'ct_LeNet', 'ep_LeNet', 'rs_LeNet', 'sad_LeNet', 'natops_LeNet')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    if net_name == 'ct_LeNet':
        net = CT_LeNet()
    
    if net_name == 'ep_LeNet':
        net = EP_LeNet()
    
    if net_name == 'rs_LeNet':
        net = RS_LeNet()
    
    if net_name == 'sad_LeNet':
        net = SAD_LeNet()
    
    if net_name == 'natops_LeNet':
        net = NATOPS_LeNet()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'ct_LeNet', 'ep_LeNet', 'rs_LeNet', 'sad_LeNet', 'natops_LeNet')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'ct_LeNet':
        ae_net = CT_LeNet_Autoencoder()
    
    if net_name == 'ep_LeNet':
        ae_net = EP_LeNet_Autoencoder()

    if net_name == 'rs_LeNet':
        ae_net = RS_LeNet_Autoencoder()
    
    if net_name == 'sad_LeNet':
        ae_net = SAD_LeNet_Autoencoder()
    
    if net_name == 'natops_LeNet':
        ae_net = NATOPS_LeNet_Autoencoder()

    return ae_net
