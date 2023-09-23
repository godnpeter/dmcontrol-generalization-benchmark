import os
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.datasets as datasets
import kornia.augmentation as aug
import utils
import random
from einops import rearrange
import cv2
from PIL import Image
import numpy as np

places_dataloader = None
places_iter = None

def _load_places(batch_size=256, image_size=84, num_workers=4, use_val=False):
	global places_dataloader, places_iter 
	partition = 'val' if use_val else 'train'
	print(f'Loading {partition} partition of places365_standard...')
	for data_dir in utils.load_config('datasets'):
		if os.path.exists(data_dir):
			fp = os.path.join(data_dir, 'places365_standard', partition)
			if not os.path.exists(fp):
				print(f'Warning: path {fp} does not exist, falling back to {data_dir}')
				fp = data_dir
			places_dataloader = torch.utils.data.DataLoader(
				datasets.ImageFolder(fp, TF.Compose([
					TF.RandomResizedCrop(image_size),
					TF.RandomHorizontalFlip(),
					TF.ToTensor()
				])),
				batch_size=batch_size, shuffle=True,
				num_workers=num_workers, pin_memory=True)
			places_iter = iter(places_dataloader)
			break
	if places_iter is None:
		raise FileNotFoundError('failed to find places365 data at any of the specified paths')
	print('Loaded dataset from', data_dir)


def _get_places_batch(batch_size):
    global places_iter
    try:
        imgs, _ = next(places_iter)
        if imgs.size(0) < batch_size:
            places_iter = iter(places_dataloader)
            imgs, _ = next(places_iter)
        elif imgs.size(0) > batch_size:
            imgs_idx = torch.randint(imgs.size(0), [batch_size])
            imgs = imgs[imgs_idx]
    except StopIteration:
        places_iter = iter(places_dataloader)
        imgs, _ = next(places_iter)
    return imgs.cuda()


class RandomOverlay(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, dataset='places365_standard'):
        """Randomly overlay an image from Places"""
        global places_iter

        if dataset == 'places365_standard':
            if places_dataloader is None:
                _load_places(batch_size=x.size(0), image_size=x.size(-1))
            imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1)//3, 1, 1)
        else:
            raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')
        
        return ((1-self.alpha)*(x/255.) + (self.alpha)*imgs)*255.

def save_obs_jpg(obs: torch.Tensor, batch_idx: int, file_name: str ='test'):
    Image.fromarray(np.uint8(obs.cpu().numpy().transpose(0,2,3,1))[batch_idx, :, :, :3]).save(file_name+'1.jpg')
    Image.fromarray(np.uint8(obs.cpu().numpy().transpose(0,2,3,1))[batch_idx, :, :, 3:6]).save(file_name+'2.jpg')
    Image.fromarray(np.uint8(obs.cpu().numpy().transpose(0,2,3,1))[batch_idx, :, :, 6:]).save(file_name+'3.jpg')


class CutoutColor(nn.Module):
    def __init__(self, patch_size=28):
        super(CutoutColor, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        # Choose random location for the patch
        
        n, c, h, w = x.shape
        for i in range(n):
            temp_x = x[i:i+1]

            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            bottom = min(top + self.patch_size, h)
            right = min(left + self.patch_size, w)

            # Create the random color
            random_color = (torch.rand(3)*255).to(torch.int).float().repeat(3) # For RGB images

            # Apply patch
            temp_x[:, :, top:bottom, left:right] = random_color[None, :, None, None]
            out = temp_x
            total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
        return total_out.reshape(n, c, h, w)
        

# This augmentation is 4x faster than previous one
class RandomConv(nn.Module):
    def __init__(self, num_stacked_frames=3, image_channel=3, kernel_size=3):
        super().__init__()
        self.num_stacked_frames = num_stacked_frames
        self.image_channel = image_channel
        self.kernel_size = kernel_size
        self.batchconv2d = BatchConv2DLayer()

    def forward(self, x):
        n, c, h, w = x.size()
        
        x = F.pad(x, pad=[1]*4, mode='replicate')
        _, _, padded_h, padded_w = x.shape
        x = x.view(n, self.num_stacked_frames, self.image_channel, padded_h, padded_w)
        n, t, img_c, padded_h, padded_w = x.size()
        assert h == w
        x = x / 255.
        
        weights = torch.randn(n, t, img_c, self.kernel_size, self.kernel_size).to(x.device)
        out = torch.sigmoid(self.batchconv2d(x, weights)) * 255.
        
        out = rearrange(out, 'b t img_c h w -> b (t img_c) h w')
        return out

class BatchConv2DLayer(nn.Module):
    """
    Applies a consistent augmentation along the time axis, while applying a different augmentation for each instance within the batch
    Perform this operation in a batch-wise manner (without any for loop)
    reference : https://github.com/pytorch/pytorch/issues/17983
    """
    def __init__(self, stride=1, padding=0, dilation=1):
        super(BatchConv2DLayer, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x, weight):
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"

        b_i, b_j, c, h, w = x.shape # [128, 3, 3, 84, 84]

        b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape # [128, 3, 3, 3, 3]

        out = x.permute([1, 0, 2, 3, 4]).reshape(b_j, b_i * c, h, w)
        weight = weight.reshape(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)

        out = F.conv2d(out, weight=weight, stride=self.stride, dilation=self.dilation, groups=b_i, padding=self.padding)

        out = out.reshape(b_j, b_i, out_channels, out.shape[-2], out.shape[-1])

        out = out.permute([1, 0, 2, 3, 4])

        return out

class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise
    
class RandomFlip(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_flip = aug.RandomHorizontalFlip(p=1.0, same_on_batch=True)
        self.v_flip = aug.RandomVerticalFlip(p=1.0, same_on_batch=True)

    def forward(self, x):
        n, c, h, w = x.shape 
        if random.random() < 0.5:
            for i in range(n):
                temp_x = x[i:i+1].reshape(-1, 3, h, w) / 255.0
                out = self.h_flip(temp_x) * 255.
                total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
            return total_out.reshape(n, c, h, w)
        else:
            for i in range(n):
                temp_x = x[i:i+1].reshape(-1, 3, h, w) / 255.0
                out = self.v_flip(temp_x) * 255.
                total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
            return total_out.reshape(n, c, h, w)

class RandomRotate(nn.Module):
    def __init__(self, degrees):
        super().__init__()
        self.random_rotate = aug.RandomRotation(degrees=degrees, same_on_batch=True, p=1.0)

    def forward(self, x):
        n, c, h, w = x.shape 
        for i in range(n):
            temp_x = x[i:i+1].reshape(-1, 3, h, w) / 255.0
            out = self.random_rotate(temp_x) * 255.
            total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
        return total_out.reshape(n, c, h, w)
        
class Projection_Transformation(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_pts = np.float32([[0, 0], [84, 0],
                           [0, 84], [84, 84]])
        
        self.left_pts = np.float32([[0, 0], [68, 16],
                            [0, 84], [68, 68]])
        
        self.right_pts = np.float32([[16, 16], [84, 0],
                    [16, 68], [84, 84]])
        
        self.top_pts = np.float32([[0, 0], [84, 0],
                    [16, 68], [68, 68]])
        
        self.bottom_pts = np.float32([[16, 16], [68, 16],
                    [0, 84], [84, 84]])
        
    def forward(self, x):
        n, c, h, w = x.shape
        total_out = None

        for i in range(n):
            random_value = random.random()
            temp_x = x[i].permute(1, 2, 0).cpu().numpy()  # Change the order of dimensions and convert to numpy

            if random_value < 0.25:
                matrix = cv2.getPerspectiveTransform(self.base_pts, self.left_pts)
            elif 0.25 <= random_value < 0.5:
                matrix = cv2.getPerspectiveTransform(self.base_pts, self.right_pts)
            elif 0.5 <= random_value < 0.75:
                matrix = cv2.getPerspectiveTransform(self.base_pts, self.top_pts)
            else:
                matrix = cv2.getPerspectiveTransform(self.base_pts, self.bottom_pts)
            
            out = cv2.warpPerspective(temp_x, matrix, (w, h))
            out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)  # Convert back to torch tensor and add batch dimension

            if total_out is None:
                total_out = out
            else:
                total_out = torch.cat([total_out, out], axis=0)

        return total_out


class RandomShiftsAug(nn.Module):
    def __init__(self, pad, random_shift_mode='bilinear'):
        super().__init__()
        self.pad = pad
        self.random_shift_mode = random_shift_mode

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')

        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,  
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift

        # default random_shift_mode: bilinear (in pytorch)
        result = F.grid_sample(x,
                             grid,
                             mode=self.random_shift_mode,
                             padding_mode='zeros',
                             align_corners=False)
        
        return result

class RandomSpatialMaskAug(nn.Module):
    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w

        mask_ratio = self.mask_ratio
        s = int(h * w)
        len_keep = round(s * (1 - mask_ratio))

        # sample random noise
        noise = torch.cuda.FloatTensor(n, s).normal_()
        # noise = torch.rand(n, s)  # noise in cpu
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([n, s], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # repeat channel_wise
        mask = mask.repeat(1, c)
        mask = mask.reshape(n, c, h, w)

        # mask-out input
        x = x * mask

        return x
    

class Augmentation(nn.Module):
    def __init__(self, obs_shape, aug_types=[]):
        super().__init__()
        random_shift_mode = 'bilinear'

        mask_ratio = None

        self.layers = []

        if aug_types == 'None' or aug_types == 'none':
             aug_types = ['None']

        for aug_type in aug_types:
            # asymmetry (no aug)
            if aug_type in ['None','none']:
                print('No augmentation in trg_Q')
                print(self.type)
                pass

            elif aug_type == 'random_shift':
                _, W, H = obs_shape
                self.layers.append(
                    RandomShiftsAug(pad=4,
                                    random_shift_mode=random_shift_mode))
                
            elif aug_type == 'random_overlay':
                 self.layers.append(RandomOverlay(alpha = 0.5))

            elif aug_type == 'random_conv':
                 self.layers.append(RandomConv())

            elif aug_type == 'cutout_color':
                self.layers.append(CutoutColor())
            
            elif aug_type == 'cutout':
                self.layers.append(aug.RandomErasing(p=0.5))
            
            elif aug_type == 'h_flip':
                self.layers.append(aug.RandomHorizontalFlip(p=0.1))

            elif aug_type == 'v_flip':
                self.layers.append(aug.RandomVerticalFlip(p=0.1))

            elif aug_type == 'rotate':
                self.layers.append(aug.RandomRotation(degrees=5.0))

            elif aug_type == 'intensity':
                self.layers.append(Intensity(scale=0.05))

            elif aug_type == 'spatial_mask':
                if mask_ratio is None:
                    mask_ratio = 0.25
                self.layers.append(RandomSpatialMaskAug(mask_ratio))

            else:
                raise ValueError

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__=='__main__':
    import time
    device = torch.device('cuda')

    obs_shape = (32, 4, 84, 84)
    input_tensor = torch.randn(obs_shape).to(device)
    aug_types = ['random_shift']
    aug_func = Augmentation(obs_shape, aug_types).to(device)
    
    ipdb.set_trace()
    # kornia augmentation speed
    t1 = time.time()
    for _ in range(100):
        aug_func(input_tensor)
    t2 = time.time()
    print(t2 - t1)

    # PyTorch augmentation speed
    aug_types = ['random_shift']
    aug_func = Augmentation(obs_shape, aug_types).to(device)
    t1 = time.time()
    for _ in range(100):
        aug_func(input_tensor)
    t2 = time.time()
    print(t2 - t1)

    # Random Spaital Augmentation
    aug_types = ['spatial_mask']
    aug_func = Augmentation(obs_shape, aug_types).to(device)
    t1 = time.time()
    for _ in range(100):
        aug_func(input_tensor)
    t2 = time.time()
    print(t2 - t1)