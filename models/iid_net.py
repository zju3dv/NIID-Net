# ////////////////////////////////////////////////////////////////////////////
# //  This file is part of NIID-Net. For more information
# //  see <https://github.com/zju3dv/NIID-Net>.
# //  If you use this code, please cite the corresponding publications as
# //  listed on the above website.
# //
# //  Copyright (c) ZJU-SenseTime Joint Lab of 3D Vision. All Rights Reserved.
# //
# //  Permission to use, copy, modify and distribute this software and its
# //  documentation for educational, research and non-profit purposes only.
# //
# //  The above copyright notice and this permission notice shall be included in all
# //  copies or substantial portions of the Software.
# //
# //  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# // SOFTWARE.
# ////////////////////////////////////////////////////////////////////////////

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _UpProjection(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size=None, scale_factor=2):
        if size is None:
            size = [x.size(2)*scale_factor, x.size(3)*scale_factor]
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out


class _NormalFeatureAdapter(nn.Module):
    def __init__(self, in_channel, extra_in_channel, out_channel):
        super(_NormalFeatureAdapter, self).__init__()
        self.up = _UpProjection(extra_in_channel, 64)
        self.conv = nn.Sequential(
            nn.Conv2d(64 + in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        up_y = self.up(y, scale_factor=x.shape[2]//y.shape[2])
        fts = torch.cat([x, up_y], dim=1)
        out = self.conv(fts)
        return out


class _Down_Block(nn.Module):
    def __init__(self, in_channel, extra_in_channel, out_channel):
        super(_Down_Block, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channel, affine=True),
            nn.ReLU(inplace=True)
        )
        if extra_in_channel > 0:
            self.NFA = _NormalFeatureAdapter(in_channel, extra_in_channel, out_channel)
        else:
            self.NFA = None

    def forward(self, input, extra_in=None):
        down_x = self.down(input)
        if self.NFA is not None:
            out = self.NFA(down_x, extra_in)
        else:
            out = down_x
        return out


class _Encoder(nn.Module):
    def __init__(self, in_channel, extra_channels, block_channels):
        super(_Encoder, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, block_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(block_channels[0], affine=True),
            nn.ReLU(inplace=True),
        )
        self.block0 = _Down_Block(block_channels[0], extra_channels[0], block_channels[1])
        self.block1 = _Down_Block(block_channels[1], extra_channels[1], block_channels[2])
        self.block2 = _Down_Block(block_channels[2], extra_channels[2], block_channels[3])
        self.block3 = _Down_Block(block_channels[3], 0, block_channels[3])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(block_channels[3], block_channels[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(block_channels[4], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(block_channels[4], block_channels[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(block_channels[4], affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, input, extra_fts):
        x_e0 = self.conv0(input)
        x_e1 = self.block0(x_e0, extra_fts[0])
        x_e2 = self.block1(x_e1, extra_fts[1])
        x_e3 = self.block2(x_e2, extra_fts[2])
        x_e4 = self.block3(x_e3, None)
        out = self.bottleneck(x_e4)
        return out, (x_e0, x_e1, x_e2, x_e3)


class _Up_Block(nn.Module):
    def __init__(self, in_channel, out_channel, concat_extra=False):
        super(_Up_Block, self).__init__()
        self.up = _UpProjection(in_channel, out_channel)
        in_ft = out_channel*2 if concat_extra else out_channel
        self.concat_extra = concat_extra
        self.conv = nn.Sequential(
                nn.Conv2d(in_ft, out_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channel, affine=True),
                nn.ReLU(inplace=True),)

    def forward(self, x, y=None):
        up_x = self.up(x, [x.size(2)*2, x.size(3)*2])
        if self.concat_extra:
            up_x = torch.cat([up_x, y], 1)
        out = self.conv(up_x)
        return out


class _Decoder(nn.Module):
    def __init__(self, block_channel, skip_connections):
        super(_Decoder, self).__init__()
        self.skip_connections = skip_connections
        self.up0 = _Up_Block(block_channel[0], block_channel[1], self.skip_connections[0])
        self.up1 = _Up_Block(block_channel[1], block_channel[2], self.skip_connections[1])
        self.up2 = _Up_Block(block_channel[2], block_channel[3], self.skip_connections[2])
        self.up3 = _Up_Block(block_channel[3], block_channel[4], self.skip_connections[3])
        self.conv = nn.Sequential(
                nn.Conv2d(block_channel[4], block_channel[4], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(block_channel[4], affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(block_channel[4], block_channel[5], kernel_size=1, stride=1, bias=True)
            )

    def forward(self, input, extra_inputs):
        x_d0 = self.up0(input, extra_inputs[0])
        x_d1 = self.up1(x_d0, extra_inputs[1])
        x_d2 = self.up2(x_d1, extra_inputs[2])
        x_d3 = self.up3(x_d2, extra_inputs[3])
        out = self.conv(x_d3)
        return out


class IIDNet(nn.Module):
    def __init__(self, input_nc, extra_feature_nc, out_R_nc, out_L_nc, ngf=64):
        super(IIDNet, self).__init__()
        self.E_channels = [ngf, ngf*2, ngf*4, ngf*8, ngf*16]
        self.D_R_channels = [ngf*16, ngf*8, ngf*4, ngf*2, ngf, out_R_nc]
        self.D_L_channels = [ngf*16, ngf*8, ngf*4, ngf*2, ngf, out_L_nc]

        self.encoder = _Encoder(input_nc, extra_feature_nc, self.E_channels)
        self.decoder_R = _Decoder(self.D_R_channels, skip_connections=[True, True, True, True])
        self.decoder_L = _Decoder(self.D_L_channels, skip_connections=[True, True, False, False])

    def forward(self, input_rgb, extra_features, pred_reflect=True, pred_shading=True):
        out_down, e_fts = self.encoder(input_rgb, extra_features)
        fts = [e_fts[x] for x in range(len(e_fts)-1, -1, -1)]
        out_R = self.decoder_R(out_down, fts) if pred_reflect else None
        out_L = self.decoder_L(out_down, fts) if pred_shading else None
        return out_R, out_L
