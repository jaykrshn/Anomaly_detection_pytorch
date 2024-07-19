import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from pytorch_ssim import pytorch_ssim

import optuna
from optuna.trial import TrialState

import os
import shutil
from tqdm import trange, tqdm
from collections import defaultdict
import argparse

import Helpers as hf
from vgg19 import VGG19
from RES_VAE_Dynamic import VAE

parser = argparse.ArgumentParser(description="Training Params")
# string args
parser.add_argument("--dataset_root", "-dr", help="Dataset root dir", type=str, default="/home/jayakrishnan/git/CNN-VAE/data")
# parser.add_argument("--dataset_root", "-dr", help="Dataset root dir", type=str, required=True)

parser.add_argument("--save_dir", "-sd", help="Root dir for saving model and data", type=str, default=".")
parser.add_argument("--norm_type", "-nt",
                    help="Type of normalisation layer used, BatchNorm (bn) or GroupNorm (gn)", type=str, default="bn")

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=10)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=128)
parser.add_argument("--ch_multi", '-w', help="Channel width multiplier", type=int, default=64)

parser.add_argument("--num_res_blocks", '-nrb',
                    help="Number of simple res blocks at the bottle-neck of the model", type=int, default=1)

parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--latent_channels", "-lc", help="Number of channels of the latent space", type=int, default=256)
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)
parser.add_argument("--block_widths", '-bw', help="Channel multiplier for the input of each block",
                    type=int, nargs='+', default=(1, 2, 4, 8))
parser.add_argument('--validation-epochs', default=1, type=int, help='the number epochs between running validation')
# float args

parser.add_argument("--kl_scale", "-ks", help="KL penalty scale", type=float, default=1)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")
parser.add_argument("--deep_model", '-dm', action='store_true',
                    help="Deep Model adds an additional res-identity block to each down/up sampling stage")
parser.add_argument('--use-cuda', default=True, action='store_true', help='Use CUDA to train model')

args = parser.parse_args()

def train(train_loader,vae_net,optimizer, feature_extractor, feature_scale, scaler, device, epoch):
    vae_net.train()
    for i, (images, _) in enumerate(tqdm(train_loader, leave=False)):
        current_iter = i + epoch * len(train_loader)
        images = images.to(device)
        bs, c, h, w = images.shape

        # We will train with mixed precision!
        with torch.cuda.amp.autocast():
            recon_img, mu, log_var = vae_net(images)

            kl_loss = hf.kl_loss(mu, log_var)
            mse_loss = F.mse_loss(recon_img, images)
            loss = args.kl_scale * kl_loss + mse_loss

            # Perception loss
            if feature_extractor is not None:
                feat_in = torch.cat((recon_img, images), 0)
                feature_loss = feature_extractor(feat_in)
                loss += feature_scale * feature_loss
                # data_logger["feature_loss"].append(feature_loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(vae_net.parameters(), 10)
        scaler.step(optimizer)
        scaler.update()


def test(test_images, vae_net, optimizer, device):
    # In eval mode the model will use mu as the encoding instead of sampling from the distribution
    vae_net.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # Save an example from testing and log a test loss
            recon_img, mu, log_var = vae_net(test_images.to(device))
            test_loss = F.mse_loss(recon_img, test_images.to(device)).item()

        # Set the model back into training mode!!
        vae_net.train()

        return test_loss

def objective(trial):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    # if args.use_cuda and torch.cuda.is_available():
    #     torch.backends.cudnn.benchmark = True
    #     logging.info("Using CUDA...")
    #     print('Using Cuda')


    # define parameters
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.001, 0.999, log=True)
    feature_scale = trial.suggest_float("feature_scale", 0.0001, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1,2])


    # Create dataloaders
    # This code assumes there is no pre-defined test/train split and will create one for you
    print("-Target Image Size %d" % args.image_size)
    transform = transforms.Compose([transforms.Resize(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])

    data_set = Datasets.ImageFolder(root=args.dataset_root, transform=transform)

    # Randomly split the dataset with a fixed random seed for reproducibility
    test_split = 0.9
    n_train_examples = int(len(data_set) * test_split)
    n_test_examples = len(data_set) - n_train_examples
    train_set, test_set = torch.utils.data.random_split(data_set, [n_train_examples, n_test_examples],
                                                        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Get a test image batch from the test_loader to visualise the reconstruction quality etc
    dataiter = iter(test_loader)
    test_images, _ = next(dataiter)

    # Create AE network.
    vae_net = VAE(channel_in=test_images.shape[1],
                ch=args.ch_multi,
                blocks=args.block_widths,
                latent_channels=args.latent_channels,
                num_res_blocks=args.num_res_blocks,
                norm_type=args.norm_type,
                deep_model=args.deep_model).to(device)

    # Setup optimizer
    # optimizer = optim.Adam(vae_net.parameters(), lr=args.lr, weight_decay=weight_decay)
    optimizer = getattr(optim, optimizer_name)(vae_net.parameters(), lr=lr, weight_decay=weight_decay)

    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler()

    if args.norm_type == "bn":
        print("-Using BatchNorm")
    elif args.norm_type == "gn":
        print("-Using GroupNorm")
    else:
        ValueError("norm_type must be bn or gn")

    # Create the feature loss module if required
    if feature_scale > 0:
        feature_extractor = VGG19().to(device)
        print("-VGG19 Feature Loss ON")
    else:
        feature_extractor = None
        print("-VGG19 Feature Loss OFF")

    # Let's see how many Parameters our Model has!
    num_model_params = 0
    for param in vae_net.parameters():
        num_model_params += param.flatten().shape[0]

    print("-This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))
    fm_size = args.image_size//(2 ** len(args.block_widths))
    print("-The Latent Space Size Is %dx%dx%d!" % (args.latent_channels, fm_size, fm_size))


    ssim_loss_fuction = pytorch_ssim.SSIM(window_size=11, size_average=True).to(device)
    # Start training loop
    for epoch in trange(0, args.nepoch, leave=False):
        train(train_loader, vae_net, optimizer, feature_extractor, feature_scale, scaler, device, epoch)

        # Save results and a checkpoint at regular intervals
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            # In eval mode the model will use mu as the encoding instead of sampling from the distribution
            test_loss = test(test_images, vae_net, optimizer, device)

            trial.report(test_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return test_loss


if __name__ == '__main__':
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("End of Tuning")