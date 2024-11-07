import torch
import torchmetrics

from training.trainable_module import TrainableModule
from training.dataset import ImageFolderDataset, PairwiseImageDataset
from training.triplane import TriPlaneGenerator

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Model(TrainableModule):
    def __init__(self, c_dim: int = 25, img_resolution: int = 256, img_channels: int = 1, rendering_kwargs: dict = None):
        super().__init__()
        self.generator = TriPlaneGenerator(c_dim=c_dim, img_resolution=img_resolution, img_channels=img_channels, rendering_kwargs=rendering_kwargs)

    def forward(self, images, c):
        generated_images = self.generator.synthesis(images, c)
        #Normalize 0-255 to 0-1
        return generated_images / 255.0

def main(args):
    model = Model(rendering_kwargs=args.rendering_kwargs)

    dataset_images = ImageFolderDataset(path=args.dataset, use_labels=True, max_size=None, xflip=False)
    dataset_pairs = PairwiseImageDataset(dataset_images)

    generator = torch.Generator().manual_seed(42)
    train, dev = torch.utils.data.random_split(dataset_pairs, [int(0.9*len(dataset_pairs)), len(dataset_pairs) - int(0.9*len(dataset_pairs))], generator=generator)

    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dev, batch_size=args.batch_size)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        loss=torch.nn.MSELoss(),
        metrics=[torchmetrics.MeanSquaredError()],
    )

    model.fit(dataloader=train, epochs=args.epochs, dev=dev)

if __name__ == '__main__':
    args = Args(batch_size=8,
                epochs=10,
                lr=1e-3,
                dataset='datasets/chest_128.zip',)
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
