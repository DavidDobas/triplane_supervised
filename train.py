import torch
import torchmetrics

from training.trainable_module import TrainableModule
from training.dataset import ImageFolderDataset
from training.triplane import TriPlaneGenerator

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Model(TrainableModule):
    def __init__(self, c_dim: int = 25, img_resolution: int = 256, img_channels: int = 1):
        super().__init__()
        self.generator = TriPlaneGenerator(c_dim=c_dim, img_resolution=img_resolution, img_channels=img_channels)

    def forward(self, images, c):
        return self.generator(images, c)

def main(args):
    model = Model()

    dataset = ImageFolderDataset(path=args.dataset, use_labels=True, max_size=None, xflip=False)
    generator = torch.Generator().manual_seed(42)
    train, dev = torch.utils.data.random_split(dataset, [int(0.9*len(dataset)), len(dataset) - int(0.9*len(dataset))], generator=generator)

    print(dataset[0][1])
    print(dataset[1][1])
    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dev, batch_size=args.batch_size)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        loss=torch.nn.MSELoss(),
        metrics=[torchmetrics.MeanSquaredError()],
    )
    model.fit(dataloader=train, epochs=args.epochs, dev=dev)

if __name__ == '__main__':
    args = Args(batch_size=16,
                epochs=10,
                lr=1e-3,
                dataset='datasets/chest_128.zip')
    main(args)
