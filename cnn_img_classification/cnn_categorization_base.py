from torch import nn


def cnn_categorization_base(netspec_opts):
    """
    Constructs a network for the base categorization model.

    Arguments
    --------
    netspec_opts: (dictionary), the network's architecture. It has the keys
                 'kernel_size', 'num_filters', 'stride', and 'layer_type'.
                 Each key holds a list containing the values for the
                corresponding parameter for each layer.
    Returns
    ------
     net: (nn.Sequential), the base categorization model
    """
    # instantiate an instance of nn.Sequential
    net = nn.Sequential()
    in_channels = 3
    
    # add layers as specified in netspec_opts to the network
    kernel_size = netspec_opts['kernel_size'] # list of length L specifying the kernel size
    num_filters = netspec_opts['num_filters'] # list of length L specifying the number of filters per layer
    stride = netspec_opts['stride'] # list of length L contains stride for the conv and pooling layers
    layer_type = netspec_opts['layer_type'] # list of strings 

    for idx, layer_type in enumerate(layer_type):
        name = "layer_" + str(idx + 1)

        if layer_type == 'conv':
            
            # calculate padding
            padding = ((kernel_size[idx] - 1)//2)

            # add convolutional layer to network.
            net.add_module(name, nn.Conv2d(in_channels, num_filters[idx], kernel_size[idx], stride[idx], padding))
            
            # Update in_channels to be the output of this layer for the next layer.
            in_channels = num_filters[idx]
            

        elif layer_type == 'bn':

            # add batch normalization layer to network using previous layer's output size as the input size.
            net.add_module(name, nn.BatchNorm2d(in_channels))
            
        elif layer_type == 'relu':

            # add relu later to network
            net.add_module(name, nn.ReLU())

        elif layer_type == 'pool':

            # calculate padding ( table has at 0)
            padding = 0

            # add avg pooling layer to network
            net.add_module(name, nn.AvgPool2d(kernel_size[idx], stride[idx], padding))
            

    return net


