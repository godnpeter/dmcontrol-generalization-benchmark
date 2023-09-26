import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial, reduce


def _get_out_shape_cuda(in_shape, layers):
	x = torch.randn(*in_shape).cuda().unsqueeze(0)
	return layers(x).squeeze(0).shape


def _get_out_shape(in_shape, layers):
	x = torch.randn(*in_shape).unsqueeze(0)
	return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
	"""Compute Gaussian log probability"""
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
	"""Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
	mu = torch.tanh(mu)
	if pi is not None:
		pi = torch.tanh(pi)
	if log_pi is not None:
		log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
	return mu, pi, log_pi


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal distribution, see https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers"""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class CenterCrop(nn.Module):
	def __init__(self, size):
		super().__init__()
		assert size in {84, 100}, f'unexpected size: {size}'
		self.size = size

	def forward(self, x):
		assert x.ndim == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		assert x.size(3) == 100, f'unexpected size: {x.size(3)}'
		if self.size == 84:
			p = 8
		return x[:, :, p:-p, p:-p]


class NormalizeImg(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x/255.


class Flatten(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


class RLProjection(nn.Module):
	def __init__(self, in_shape, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.projection = nn.Sequential(
			nn.Linear(in_shape[0], out_dim),
			nn.LayerNorm(out_dim),
			nn.Tanh()
		)
		self.apply(weight_init)
	
	def forward(self, x):
		return self.projection(x)

	def reset_parameters(self, reset_seed):
		# Save current RNG state
		original_rng_state = torch.get_rng_state()

		# Save current parameters
		current_weights = {}
		for name, param in self.named_parameters():
			current_weights[name] = param.data.clone()

		# Set the manual seed
		torch.manual_seed(reset_seed)
		# Apply the initialization
		self.apply(weight_init)

		# Restore original RNG state
		torch.set_rng_state(original_rng_state)


class SODAMLP(nn.Module):
	def __init__(self, projection_dim, hidden_dim, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.mlp = nn.Sequential(
			nn.Linear(projection_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, out_dim)
		)
		self.apply(weight_init)

	def forward(self, x):
		return self.mlp(x)


class SharedCNN(nn.Module):
	def __init__(self, obs_shape, num_layers=11, num_filters=32):
		super().__init__()
		assert len(obs_shape) == 3
		self.num_layers = num_layers
		self.num_filters = num_filters

		self.layers = [CenterCrop(size=84), NormalizeImg(), nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
		for _ in range(1, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(obs_shape, self.layers)
		self.apply(weight_init)

	def forward(self, x):
		return self.layers(x)
	
	def reset_parameters(self, reset_seed, shrink_alpha):
		# Save current RNG state
		original_rng_state = torch.get_rng_state()

		# Save current parameters
		current_weights = {}
		for name, param in self.named_parameters():
			current_weights[name] = param.data.clone()

		# Set the manual seed
		torch.manual_seed(reset_seed)
		# Apply the initialization
		self.apply(weight_init)

		# Update the weights: 0.8 * current weight + 0.2 * new weight
		for name, param in self.named_parameters():
			param.data = shrink_alpha * current_weights[name] + (1-shrink_alpha) * param.data

		# Restore original RNG state
		torch.set_rng_state(original_rng_state)



class HeadCNN(nn.Module):
	def __init__(self, in_shape, num_layers=0, num_filters=32):
		super().__init__()
		self.layers = []
		for _ in range(0, num_layers):
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
		self.layers.append(Flatten())
		self.layers = nn.Sequential(*self.layers)
		self.out_shape = _get_out_shape(in_shape, self.layers)
		self.apply(weight_init)

	def forward(self, x):
		return self.layers(x)


class Encoder(nn.Module):
	def __init__(self, shared_cnn, head_cnn, projection):
		super().__init__()
		self.shared_cnn = shared_cnn
		self.head_cnn = head_cnn
		self.projection = projection
		self.out_dim = projection.out_dim

	def forward(self, x, detach=False):
		x = self.shared_cnn(x)
		x = self.head_cnn(x)
		if detach:
			x = x.detach()
		return self.projection(x)
	
	def reset_encoder_parameters(self, reset_seed, shrink_alpha):
		self.shared_cnn.reset_parameters(reset_seed, shrink_alpha)
  
	def reset_projection_parameters(self, reset_seed):
		self.projection.reset_parameters(reset_seed)


class Actor(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim, log_std_min, log_std_max):
		super().__init__()
		self.encoder = encoder
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.mlp = nn.Sequential(
			nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, 2 * action_shape[0])
		)
		self.mlp.apply(weight_init)

	def forward(self, x, compute_pi=True, compute_log_pi=True, detach=False):
		x = self.encoder(x, detach)
		mu, log_std = self.mlp(x).chunk(2, dim=-1)
		log_std = torch.tanh(log_std)
		log_std = self.log_std_min + 0.5 * (
			self.log_std_max - self.log_std_min
		) * (log_std + 1)

		if compute_pi:
			std = log_std.exp()
			noise = torch.randn_like(mu)
			pi = mu + noise * std
		else:
			pi = None
			entropy = None

		if compute_log_pi:
			log_pi = gaussian_logprob(noise, log_std)
		else:
			log_pi = None

		mu, pi, log_pi = squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std

	def reset_policy_parameters(self, reset_seed):
		# Save current RNG state
		original_rng_state = torch.get_rng_state()

		# Set the manual seed
		torch.manual_seed(reset_seed)

		# Apply the initialization
		self.mlp.apply(weight_init)

		# Restore original RNG state
		torch.set_rng_state(original_rng_state)

class QFunction(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim):
		super().__init__()
		self.trunk = nn.Sequential(
			nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		self.apply(weight_init)

	def forward(self, obs, action):
		assert obs.size(0) == action.size(0)
		return self.trunk(torch.cat([obs, action], dim=1))


class Critic(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.Q1 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)
		self.Q2 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)

	def forward(self, x, action, detach=False):
		x = self.encoder(x, detach)
		return self.Q1(x, action), self.Q2(x, action)

	def reset_policy_parameters(self, reset_seed):
		# Save current RNG state
		original_rng_state = torch.get_rng_state()

		# Set the manual seed
		torch.manual_seed(reset_seed)

		# Apply the initialization
		self.Q1.apply(weight_init)
		self.Q2.apply(weight_init)

		# Restore original RNG state
		torch.set_rng_state(original_rng_state)

class CURLHead(nn.Module):
	def __init__(self, encoder):
		super().__init__()
		self.encoder = encoder
		self.W = nn.Parameter(torch.rand(encoder.out_dim, encoder.out_dim))

	def compute_logits(self, z_a, z_pos):
		"""
		Uses logits trick for CURL:
		- compute (B,B) matrix z_a (W z_pos.T)
		- positives are all diagonal elements
		- negatives are all other elements
		- to compute loss use multiclass cross entropy with identity matrix for labels
		"""
		Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
		logits = torch.matmul(z_a, Wz)  # (B,B)
		logits = logits - torch.max(logits, 1)[0][:, None]
		return logits



class BehaviorCloning(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.mlp = nn.Sequential(
			nn.Linear(encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, action_shape[0])
		)
		self.apply(weight_init)

	def forward(self, x):
		h = self.encoder(x)
		return self.mlp(h)

	def reset_policy_parameters(self, reset_seed):
		# Save current RNG state
		original_rng_state = torch.get_rng_state()

		# Set the manual seed
		torch.manual_seed(reset_seed)

		# Apply the initialization
		self.mlp.apply(weight_init)

		# Restore original RNG state
		torch.set_rng_state(original_rng_state)



class InverseDynamics(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.mlp = nn.Sequential(
			nn.Linear(2*encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, action_shape[0])
		)
		self.apply(weight_init)

	def forward(self, x, x_next):
		h = self.encoder(x)
		h_next = self.encoder(x_next)
		joint_h = torch.cat([h, h_next], dim=1)
		return self.mlp(joint_h)

	def reset_policy_parameters(self, reset_seed):
		# Save current RNG state
		original_rng_state = torch.get_rng_state()

		# Set the manual seed
		torch.manual_seed(reset_seed)

		# Apply the initialization
		self.mlp.apply(weight_init)

		# Restore original RNG state
		torch.set_rng_state(original_rng_state)

class SODAPredictor(nn.Module):
	def __init__(self, encoder, hidden_dim):
		super().__init__()
		self.encoder = encoder
		self.mlp = SODAMLP(
			encoder.out_dim, hidden_dim, encoder.out_dim
		)
		self.apply(weight_init)

	def forward(self, x):
		return self.mlp(self.encoder(x))


#################################################
###############IMPALA RELATED CODE###############
def _get_out_shape(in_shape, layers):
	x = torch.randn(*in_shape).unsqueeze(0)
	for l in layers:
		x = l(x)

	return x.squeeze(0).shape

class ResidualBlock(nn.Module):
	def __init__(self,
					in_channels):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

		self.layer_norm1 = nn.GroupNorm(1, in_channels, affine=True)
		self.layer_norm2 = nn.GroupNorm(1, in_channels, affine=True)

		self.residual_layers = nn.Sequential(
			nn.ReLU(),
			self.conv1,
			self.layer_norm1,
			nn.ReLU(),
			self.conv2,
			self.layer_norm2
		)


	def forward(self, x):
		out = self.residual_layers(x)
		return out + x

class ImpalaBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ImpalaBlock, self).__init__()
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
		self.layer_norm = nn.GroupNorm(1, out_channels, affine=True)
		self.res1 = ResidualBlock(out_channels)
		self.res2 = ResidualBlock(out_channels)

		self.block_layers = nn.Sequential(
			self.conv,
			self.layer_norm,
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			self.res1,
			self.res2
		)

	def forward(self, x):
		x = self.block_layers(x)
		return x

class Impala(nn.Module):
	def __init__(self,
					obs_shape,
					width_expansion):
		super(Impala, self).__init__()
		self.in_channels = obs_shape[0]
		self.backbone = list()
		self.backbone.append(ImpalaBlock(in_channels=self.in_channels, out_channels=16 * width_expansion))
		self.backbone.append(ImpalaBlock(in_channels=16 * width_expansion, out_channels=32 * width_expansion))
		self.backbone.append(ImpalaBlock(in_channels=32 * width_expansion, out_channels=32 * width_expansion))
		self.backbone = nn.Sequential(*self.backbone)

		self.out_shape = _get_out_shape(obs_shape, self.backbone)
		self.apply(weight_init)

		#FIXME need to 
        # if weight_init_type == 'xavier':
        #     self.apply(xavier_uniform_init)
        # elif weight_init_type == 'dmc':
        #     self.apply(utils.weight_init)

	def forward(self, x):
		x = x / 255.0 - 0.5
		# Extract observation tensor from dictionary x
		x = self.backbone(x)
		x = nn.ReLU()(x)
		x = x.view(x.shape[0], -1)
		return x

	def reset_parameters(self, reset_seed, shrink_alpha=0.5):
		# Save current RNG state
		original_rng_state = torch.get_rng_state()

		# Save current parameters
		current_weights = {}
		for name, param in self.named_parameters():
			current_weights[name] = param.data.clone()

		# Set the manual seed
		torch.manual_seed(reset_seed)
		# Apply the initialization
		self.apply(weight_init)

		# Update the weights(e.g., 0.5 * current weight + 0.5 * new weight)
		for name, param in self.named_parameters():
			param.data = shrink_alpha * current_weights[name] + (1-shrink_alpha) * param.data

		# Restore original RNG state
		torch.set_rng_state(original_rng_state)

#################################################
#################################################
    
