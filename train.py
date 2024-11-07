import os
import re
import datetime
import json

import torch
import torchmetrics
import torcheval

from training.trainable_module import TrainableModule
from training.dataset import ImageFolderDataset, PairwiseImageDataset
from training.triplane import TriPlaneGenerator

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class UNet(torch.nn.Module):
    def __init__(self, num_narrowings=3):
        super().__init__()
        
        # Initial convolution to get to 96 channels
        self.init_conv = torch.nn.Conv2d(1, 96, kernel_size=3, padding=1)
        
        # Encoder blocks
        self.encoder_blocks = torch.nn.ModuleList()
        current_channels = 96
        for i in range(num_narrowings):
            out_channels = current_channels * 2
            self.encoder_blocks.append(torch.nn.Sequential(
                torch.nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True)
            ))
            current_channels = out_channels
            
        # Decoder blocks
        self.decoder_blocks = torch.nn.ModuleList()
        for i in range(num_narrowings):
            in_channels = current_channels
            out_channels = current_channels // 2
            self.decoder_blocks.append(torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True)
            ))
            current_channels = out_channels
            
        # Final convolution to get back to 96 channels
        self.final_conv = torch.nn.Conv2d(96, 96, kernel_size=1)

    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)
        
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder path
        for block in self.encoder_blocks:
            encoder_outputs.append(x)
            x = block(x)
            
        # Decoder path with skip connections
        for block, skip in zip(self.decoder_blocks, reversed(encoder_outputs)):
            x = block(x)
            # Add skip connection
            x = x + skip
            
        # Final 1x1 convolution
        x = self.final_conv(x)
        
        return x


class Model(TrainableModule):
    def __init__(self, c_dim: int = 25, img_resolution: int = 256, img_channels: int = 1, rendering_kwargs: dict = None, use_unet: bool = False, num_narrowings: int = 3):
        super().__init__()
        # Custom
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.generator = TriPlaneGenerator(c_dim=c_dim, img_resolution=img_resolution, img_channels=img_channels, rendering_kwargs=rendering_kwargs)
        self.use_unet = use_unet
        if use_unet:
            self.unet = UNet(num_narrowings=num_narrowings)

    def forward(self, images, c):
        if self.use_unet:
            planes = self.unet(images)
        else:
            planes = self.conv(images)
        planes = planes.squeeze(1)
        generated_images = self.generator.synthesis(planes, c)
        #Normalize 0-255 to 0-1
        return generated_images / 255.0

def main(args):
    model = Model(rendering_kwargs=args.rendering_kwargs, use_unet=args.use_unet, num_narrowings=args.num_narrowings)

    # Create logdir with timestamp
    timestamp = datetime.datetime.now()
    args.logdir = f"logdir_{timestamp.strftime('%Y%m%d')}_{timestamp.strftime('%H%M%S')}"
    # Save args to args.json in logdir
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    dataset_images = ImageFolderDataset(path=args.dataset, use_labels=True, max_size=None, xflip=False)
    dataset_pairs = PairwiseImageDataset(dataset_images, size=args.pairwise_dataset_size)

    generator = torch.Generator().manual_seed(42)
    train, dev = torch.utils.data.random_split(dataset_pairs, [int(0.9*len(dataset_pairs)), len(dataset_pairs) - int(0.9*len(dataset_pairs))], generator=generator)

    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dev, batch_size=args.batch_size)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        loss=torch.nn.MSELoss(),
        metrics=[torchmetrics.image.PeakSignalNoiseRatio()],
        logdir=args.logdir,
    )

    model.fit(dataloader=train, epochs=args.epochs, dev=dev)

if __name__ == '__main__':
    args = Args(batch_size=8,
                epochs=10,
                lr=1e-3,
                dataset='datasets/chest_128.zip',
                pairwise_dataset_size=10000,
                use_unet=False,
                num_narrowings=3)
    args.rendering_kwargs = rendering_options = {
        'image_resolution': 128,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'c_gen_conditioning_zero': True, # if true, fill generator pose conditioning label with dummy zero vector
        'gpc_reg_prob': 0.5,
        'c_scale': 1, # mutliplier for generator pose conditioning label
        'superresolution_noise_mode': 'none', # [random or none], whether to inject pixel noise into super-resolution layers
        'density_reg': 0.25, # strength of density regularization
        'density_reg_p_dist': 0.004, # distance at which to sample perturbed points for density regularization
        'reg_type': 'l1', # for experimenting with variations on density regularization
        'decoder_lr_mul': 1, # learning rate multiplier for decoder
        'sr_antialias': True,
    }
    rendering_options.update({
            'depth_resolution': 64,
            'depth_resolution_importance': 64,
            'ray_start': 'auto',
            'ray_end': 'auto',
            'box_warp': 9.5,
            'white_back': False,
            'avg_camera_radius': 10.5,
            'avg_camera_pivot': [0, 0, 0],
        })
    main(args)
