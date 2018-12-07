import torch 
import torch.nn as nn  
import pretrainedmodels
import glob

model_configs = {'polynet':{
                   'input_size': 512,
                   'input_mean': [0.485, 0.456, 0.406, 0.406],
                   'input_std' : [0.229, 0.224, 0.225, 0.225]
                    },
                'resnet101':{
                   'input_size': 512,
                   'input_mean': [0.485, 0.456, 0.406, 0.406],
                   'input_std' : [0.229, 0.224, 0.225, 0.225]
                    },
                'resnet50':{
                   'input_size': 512,
                   'input_mean': [0.485, 0.456, 0.406, 0.406],
                   'input_std' : [0.229, 0.224, 0.225, 0.225]
                    },
                'resnext101_32x4d':{
                   'input_size': 512,
                   'input_mean': [0.485, 0.456, 0.406, 0.406],
                   'input_std' : [0.229, 0.224, 0.225, 0.225]
                    },
                'se_resnext50_32x4d':{
                   'input_size': 512,
                   'input_mean': [0.485, 0.456, 0.406, 0.406],
                   'input_std' : [0.229, 0.224, 0.225, 0.225]
                    },
                'inceptionresnetv2':{
                   'input_size': 512,
                   'input_mean': [0.5],
                   'input_std' : [0.5]
                    },
                'xception':{
                   'input_size': 512,
                   'input_mean': [0.5],
                   'input_std' : [0.5]
                    },
                'dpn68':{
                   'input_size': 512,
                   'input_mean': [0.5],
                   'input_std' : [0.5]
                    },
                'dpn98':{
                   'input_size': 512,
                   'input_mean': [0.5],
                   'input_std' : [0.5]
                    },
                'inceptionv4':{
                   'input_size': 512,
                   'input_mean': [0.5],
                   'input_std' : [0.5]
                    },
                }


def construct_rgby_model(model_name, split):

    """
    Handle 4 dimensional input
    """

    model = pretrainedmodels.__dict__[model_name](num_classes = 1000)
    modules = list(model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]

    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (4, ) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

    new_conv = nn.Conv2d(4, conv_layer.out_channels,
                         conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                         bias=True if len(params) == 2 else False)
    
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data 
    layer_name = list(container.state_dict().keys())[0][:-7] 

    setattr(container, layer_name, new_conv)

    """
    Handle 512 input size by changing the average pooling layer
    """

    if 'resnet' in model_name:
        model.avgpool = nn.AdaptiveAvgPool2d(1)
    elif 'resnext' in model_name:
        model.avg_pool = nn.AdaptiveAvgPool2d(1)

    """
    Changing the last linear layer output to 28
    """

    if 'dpn' in model_name:
        in_channels = model.last_linear.in_channels
        kernel_size = model.last_linear.kernel_size
        model.last_linear = nn.Conv2d(in_channels, 28, kernel_size)
    else:
        num_features = model.last_linear.in_features 
        model.last_linear = nn.Linear(num_features, 28)

    """
    Load existing checkpoint files
    """
    if glob.glob('{}_rgby_focal_{}*'.format(model_name, split)):
        pth_file = torch.load('{}_rgby_focal_{}.pth.tar'.format(model_name, split))
        state_dict = pth_file['state_dict']
        model.load_state_dict(state_dict)
        start_epoch = pth_file['epoch']

    return model