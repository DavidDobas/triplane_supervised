import os
import re
import datetime
import json

import torch
import torchmetrics
import torcheval
import torchvision

from training.trainable_module import TrainableModule
from training.dataset import ImageFolderDataset, PairwiseImageDataset
from training.triplane import TriPlaneGenerator
from training.loss import SSIMLoss, MSESSIMLoss, PerceptualLoss
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
    def __init__(self, c_dim: int = 25, img_resolution: int = 256, img_channels: int = 1, rendering_kwargs: dict = None, use_unet: bool = False, num_narrowings: int = 3, args: Args = None):
        super().__init__()
        # Custom
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.generator = TriPlaneGenerator(c_dim=c_dim, img_resolution=img_resolution, img_channels=img_channels, rendering_kwargs=rendering_kwargs)
        self.use_unet = use_unet
        if use_unet:
            self.unet = UNet(num_narrowings=num_narrowings)
        self.args = args.copy()
        if args.use_final_cnn:
            # Upscaling convolution from [N,32,64,64] to [N,16,128,128]
            self.upscale_conv = torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
            # Final convolution to single channel [N,1,128,128]
            self.final_conv = torch.nn.Conv2d(16, 1, kernel_size=3, padding=1)
    def forward(self, images, c):
        if self.use_unet:
            planes = self.unet(images)
        else:
            planes = self.conv(images)
        planes = planes.squeeze(1)
        feature_image = self.generator.synthesis(planes, c)

        if args.use_final_cnn:
            
            
            # Apply the convolutions
            feature_image = self.upscale_conv(feature_image)
            feature_image = self.final_conv(feature_image)
            return feature_image

        rgb_image = feature_image[:, :3].contiguous()
        # Upscale from 64x64 to 128x128 using bilinear interpolation
        rgb_image = torch.nn.functional.interpolate(rgb_image, size=(128, 128), mode='bilinear', align_corners=False)

        # Convert RGB image to greyscale
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=rgb_image.device, dtype=rgb_image.dtype)
        grayscale_image = torch.sum(rgb_image * rgb_weights[None, :, None, None], dim=1, keepdim=True)

        # sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        # return {'image_raw': rgb_image, 'image_depth': depth_image}
        return grayscale_image

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

    if args.loss == 'mse_ssim':
        loss = MSESSIMLoss()
    elif args.loss == 'perceptual':
        loss = PerceptualLoss()
    elif args.loss == 'mse':
        loss = torch.nn.MSELoss()
    else:
        raise ValueError(f"Invalid loss function: {args.loss}")

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        loss=loss,
        metrics=[torchmetrics.image.PeakSignalNoiseRatio()],
        logdir=args.logdir,
    )

    model.fit(dataloader=train, epochs=args.epochs, dev=dev)
    
    # Save the first image from the dataset to logdir
    sample = next(iter(dev))[0]

    first_image = sample[0][1,:,:,:]  # Get the first image in the batch
    first_image_path = os.path.join(args.logdir, 'first_image.png')
    torchvision.utils.save_image(first_image, first_image_path)

    # Make a prediction on the first image and its camera
    model.eval()
    with torch.no_grad():
        predictions = model.predict(dev, as_numpy=False)
    prediction_path = os.path.join(args.logdir, 'prediction.png')
    torchvision.utils.save_image(predictions[0][0], prediction_path)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--use_unet', type=bool, default=True, help='Whether to use U-Net in the model.')
    parser.add_argument('--num_narrowings', type=int, default=3, help='Number of narrowings in the U-Net.')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function to use.')
    return parser.parse_args()

args_user = parse_args()

if __name__ == '__main__':
    args = Args(batch_size=8,
                epochs=10,
                lr=1e-4,
                dataset='datasets/chest_128.zip',
                pairwise_dataset_size=10000,
                use_unet=True,
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
    args.epochs = args_user.epochs
    args.lr = args_user.lr
    args.batch_size = args_user.batch_size
    args.use_unet = args_user.use_unet
    args.loss = args_user.loss
    print("Training for {} epochs with batch size {}".format(args.epochs, args.batch_size))
    main(args)
