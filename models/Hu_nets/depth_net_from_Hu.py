import torch
import torch.nn as nn
from torch.autograd import Variable


from models.Hu_nets import resnet, densenet, senet, modules


class baseNet(nn.Module):
    def __init__(self, Encoder, num_features):
        super(baseNet, self).__init__()
        self.E = Encoder
        self.D = modules.D(num_features)
        self.num_features = num_features

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        return (x_block1, x_block2, x_block3, x_block4), x_decoder


class refineNet(nn.Module):
    def __init__(self, block_channel):
        super(refineNet, self).__init__()
        # self.up = modules._UpProjection(
        #     num_input_features=block_channel[-1], num_output_features=block_channel[-1])
        self.MFF = modules.MFF(block_channel[:-1])
        self.R = modules.R(64+block_channel[-1], out_channels=3)

    def forward(self, x_decoder, x_block1, x_block2, x_block3, x_block4):
        # x_decoder = self.up(x_decoder, [x_decoder.size(2)*2, x_decoder.size(3)*2])
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))
        return out


def build_model(is_resnet=False, is_densenet=False, is_senet=False):
    if is_resnet:
        original_model = resnet.resnet50(pretrained=True)
        Encoder = modules.E_resnet(original_model)
        num_features = 2048
        block_channel = [256, 512, 1024, 2048, num_features//32]
        # model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    elif is_densenet: # densenet 121
        # original_model = densenet.densenet161(pretrained=True)
        original_model = densenet.densenet121(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        num_features = 1024
        block_channel = [128, 256, 512, 1024, num_features//32]
        # num_features = 2208
        # block_channel = [192, 384, 1056, 2208, num_features//32]
        # model = net.model(Encoder, num_features=2208, block_channel=[192, 384, 1056, 2208])
    elif is_senet:
        # original_model = senet.senet154(pretrained='imagenet')
        original_model = senet.senet154(pretrained=None)
        Encoder = modules.E_senet(original_model)
        num_features = 2048
        block_channel = [256, 512, 1024, 2048, num_features//32]
        # model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    else:
        raise Exception('Not define encoder type for baseNet in depth_net_from_Hu.py!')
    return baseNet(Encoder, num_features), refineNet(block_channel)
