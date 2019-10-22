import argparse
import logging
import os

import torch
import torchvision.datasets as datasets
from torch.optim import Adam
from torchvision import transforms
from dcgan import DCGenerator, DCDiscriminator, DCGANTrainer


def get_config():
    parser = argparse.ArgumentParser(description='Training DCGAN on CIFAR10')

    parser.add_argument('--log-root', type=str, default='../logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_dcgan.log')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--latent-size', type=int, default=100,
                        help='latent vector size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train ')
    parser.add_argument('--image-size', type=int, default=32,
                        help='size of images to generate')
    parser.add_argument('--n_show_samples', type=int, default=8)
    parser.add_argument('--show_img_every', type=int, default=10)
    parser.add_argument('--log_metrics_every', type=int, default=100)
    config = parser.parse_args()
    config.cuda = not config.no_cuda and torch.cuda.is_available()

    return config


def main():
    config = get_config()
    os.makedirs(config.log_root, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_root,
                                             config.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    transform = transforms.Compose([transforms.Scale(32), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root=config.data_root, download=True,
                               transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                             num_workers=0, pin_memory=True)

    discriminator = DCDiscriminator(config.image_size, 3)
    generator = DCGenerator(config.latent_size, config.image_size, 3)

    trainer = DCGANTrainer(generator=generator, discriminator=discriminator,
                           optimizer_d=Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
                           optimizer_g=Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
                           metrics_dir='metrics', device='cpu')

    trainer.train(dataloader, config.epochs, config.n_show_samples, config.show_img_every, config.log_metrics_every)


if __name__ == '__main__':
    main()
