import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TorchVAE(nn.Module):
  def __init__(self, zsize=256, layer_count=5, channels=3):
    super(TorchVAE, self).__init__()

    d = 128
    self.d = d
    self.zsize = zsize

    self.layer_count = layer_count

    mul = 1
    inputs = channels
    for i in range(self.layer_count):
      setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, d * mul, 4, 2, 1))
      setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
      inputs = d * mul
      mul *= 2

    self.d_max = inputs

    self.fc1 = nn.Linear(inputs * 4 * 4, zsize)
    self.fc2 = nn.Linear(inputs * 4 * 4, zsize)

    self.d1 = nn.Linear(zsize, inputs * 4 * 4)

    mul = inputs // d // 2

    for i in range(1, self.layer_count):
      setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, d * mul, 4, 2, 1))
      setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
      inputs = d * mul
      mul //= 2

    setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, channels, 4, 2, 1))

  def encode(self, x):
    for i in range(self.layer_count):
      x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))

    x = x.view(x.shape[0], self.d_max * 4 * 4)
    h1 = self.fc1(x)
    h2 = self.fc2(x)
    return h1, h2

  def reparameterize(self, mu, logvar):
    if self.training:
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mu)
    else:
      return mu

  def decode(self, x):
    x = x.view(x.shape[0], self.zsize)
    x = self.d1(x)
    x = x.view(x.shape[0], self.d_max, 4, 4)
    # x = self.deconv1_bn(x)
    x = F.leaky_relu(x, 0.2)

    for i in range(1, self.layer_count):
      x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)

    x = torch.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
    return x

  def encode_to_z(self, x):
    mu, logvar = self.encode(x)
    mu = mu.squeeze()
    logvar = logvar.squeeze()
    z = self.reparameterize(mu, logvar)
    return z, mu, logvar

  def encode_to_z_numpy(self, imgs):
    with torch.no_grad():
      z, mu, logvar = self.encode_to_z(process_batch(imgs))
      return z.cpu().numpy()

  def forward(self, x):
    z, mu, logvar = self.encode_to_z(x)
    return self.decode(z.view(-1, self.zsize, 1, 1)), mu, logvar

  def weight_init(self, mean, std):
    for m in self._modules:
      normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()


def loss_function(recon_x, x, mu, logvar):
  BCE = torch.mean((recon_x - x) ** 2)
  KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
  return BCE, KLD * 0.1


def process_batch(batch):
  data = [x.transpose((2, 0, 1)) for x in batch]
  x = torch.from_numpy(np.asarray(data, dtype=np.float32)).cuda() / 127.5 - 1.
  x = x.view(-1, 3, 128, 128)
  return x
