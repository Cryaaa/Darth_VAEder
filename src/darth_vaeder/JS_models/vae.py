import torch
import torch.nn.functional as F
from torch import nn


class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):
    """Basic Block Encoder

    Args:
        in_planes (int): number of input planes
        stride: (int or tuple): stride of the convolution
    """

    def __init__(self, in_planes: int, stride: int = 1):
        super().__init__()

        planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes: int, stride: int = 1):
        super().__init__()

        planes = int(in_planes / stride)

        # TODO: Why are in_planes here twice?
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        # self.bn1 could have been placed here,
        # but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()

        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv2(x)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = out + self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):
    """Resnet Encoder

    Args:
        num_blocks (list): Number of residual blocks in each of the 4 layers
        z_dim (int): Dimensionality of the latent space
        nc (int): Number of channels
    """

    def __init__(self, num_Blocks: list = [2, 2, 2, 2], z_dim: int = 10, nc: int = 3, img_size: int = 256):
        super().__init__()
        # [256]: self.img_size=256
        self.img_size = img_size
        self.in_planes = 64  # running counter of current channel depth?
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.finalConv = nn.Conv2d(512, 2 * z_dim, kernel_size=1) # changed name,not lienar one time z_dim

        self.pooling=nn.AvgPool2d(kernel_size=2)
        # [256]: self.ln1=nn.Linear(((256//(2**5))**2)*2*z_dim, 2*z_dim)  # = Linear(128*z_dim, 2*z_dim)
        # spatial dim after 4 stride-2 layers + AvgPool2d(2) = img_size // 32
        self.ln1=nn.Linear(((self.img_size//(2**5))**2)*2*z_dim, 2*z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # print(f"First layer shape: {x.shape}")
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.layer1(x)
        # print(f"second layer shape: {x.shape}")
        x = self.layer2(x)
        # print(f"Third layer shape: {x.shape}")

        x = self.layer3(x)
        #   print(f"Fourth layer shape: {x.shape}")
        x = self.layer4(x)
        # print(f"Fifth layer shape: {x.shape}")
        x = self.finalConv(x)
        x = self.pooling(x)
        # print(f"pooling shape: {x.shape}")
        x=torch.flatten(x, start_dim=1)
        # print(f"flatten layer shape: {x.shape}")

        x=torch.relu(x)
        x=self.ln1(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar
    
class ResNet18Dec(nn.Module):
    """Resnet Decoder

    Args:
        num_blocks (list): Number of residual blocks in each of the 4 layers
        z_dim (int): Dimensionality of the latent space
        nc (int): Number of channels
    """

    def __init__(self, num_Blocks: list = [2, 2, 2, 2], z_dim: int = 10, nc: int = 3, img_size: int = 256):
        super().__init__()
        self.in_planes = 512
        self.nc = nc

        # spatial start size: total decoder upsampling = ×2 (layer4) ×2 (layer3) ×2 (layer2) ×1 (layer1) ×2 (conv1) = ×16
        # [256]: self.lnout = nn.Linear(z_dim, 16*16*512)
        # [256]: self._start_size = 16
        self._start_size = img_size // 16   # 256→16, 96→6
        self.lnout = nn.Linear(z_dim, self._start_size * self._start_size * 512) # changed name, not linear one time in_planes

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)

        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

        ######

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        # TODO: why is this outside of the for-loop here, but inside the for-loop for the Encoder?
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.lnout(x)
        # [256]: x = x.view(x.size(0), 512, 16, 16)
        x = x.view(x.size(0), 512, self._start_size, self._start_size)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.conv1(x)
        # x = torch.sigmoid(x)
        return x


# defines a new class called VAEResNet18 that inherits from nn.Module,
# which is the base class for all neural networks in PyTorch
class VAEResNet18(nn.Module):
    """
    Args:
        nc (int): Number of channels
        z_dim (int): Dimensionality of the latent space
    """
    def __init__(self, nc: int, z_dim: int, img_size: int = 256) -> None:  # Constructor
        super().__init__()  # calls the parent class nn.Module constructor
        # [256]: self.encoder = ResNet18Enc(nc=nc, z_dim=z_dim)
        # [256]: self.decoder = ResNet18Dec(nc=nc, z_dim=z_dim)
        self.encoder = ResNet18Enc(nc=nc, z_dim=z_dim, img_size=img_size)

        self.decoderCell = ResNet18Dec(nc=nc//2, z_dim=z_dim, img_size=img_size)
        self.decoderNuc = ResNet18Dec(nc=nc//2, z_dim=z_dim, img_size=img_size)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        # ([128, 384, 96])
        x_cell = self.decoderCell(z)
        x_nuc = self.decoderNuc(z)
        # print( 'debugb shape:', x_cell.shape)
        # print( 'debugb shape:', x_nuc.shape)
        x=torch.cat([x_cell[:,:1,...], x_nuc[:,:1,...],x_cell[:,1:2,...], x_nuc[:,1:2,...]], dim=1)
        # print( 'debugb shape:', x.shape)
        return x, z, mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std)
        return epsilon * std + mean
