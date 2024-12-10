import torch
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader
import torchvision.transforms as transforms # type: ignore
import torch.nn.functional as F
import torchvision.utils as vutils # type: ignore
import logging
from pytorch_ssim import pytorch_ssim
import mlflow

import os
import shutil
from tqdm import trange, tqdm # type: ignore
from collections import defaultdict
import argparse

import Helpers as hf
from vgg19 import VGG19
from RES_VAE_Dynamic import VAE

parser = argparse.ArgumentParser(description="Training Params")
# string args
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)
parser.add_argument("--dataset_root", "-dr", help="Dataset root dir", type=str, required=False)

parser.add_argument("--save_dir", "-sd", help="Root dir for saving model and data", type=str, default=".")
parser.add_argument("--norm_type", "-nt",
                    help="Type of normalisation layer used, BatchNorm (bn) or GroupNorm (gn)", type=str, default="bn")

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=100)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=32)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=128)
parser.add_argument("--ch_multi", '-w', help="Channel width multiplier", type=int, default=16)

parser.add_argument("--num_res_blocks", '-nrb',
                    help="Number of simple res blocks at the bottle-neck of the model", type=int, default=1)

parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--latent_channels", "-lc", help="Number of channels of the latent space", type=int, default=512)
parser.add_argument('--validation_epochs', default=1, type=int, help='the number epochs between running validation')
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)
parser.add_argument("--block_widths", '-bw', help="Channel multiplier for the input of each block",
                    type=int, nargs='+', default=(1, 2, 4, 8))
# float args
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--feature_scale", "-fs", help="Feature loss scale", type=float, default=1)
parser.add_argument("--kl_scale", "-ks", help="KL penalty scale", type=float, default=1)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")
parser.add_argument("--deep_model", '-dm', action='store_true',
                    help="Deep Model adds an additional res-identity block to each down/up sampling stage")
parser.add_argument('--use_cuda', default=True, action='store_true', help='Use CUDA to train model')

args = parser.parse_args()


def train(train_loader,vae_net, data_logger,optimizer, ssim_loss_func, device, epoch):
    vae_net.train()
    train_loss = 0
    train_mse_loss = 0
    train_ssim_loss = 0
    train_kl_loss = 0
    train_feature_loss = 0
    for i, (images, label_images) in enumerate(tqdm(train_loader, leave=False)):
        #current_iter = i + epoch * len(train_loader)
        images = images.to(device)
        label_images = label_images.to(device) 
        bs, c, h, w = images.shape

        # We will train with mixed precision!
        with torch.cuda.amp.autocast():
            recon_img, mu, log_var = vae_net(images)

            kl_loss = hf.kl_loss(mu, log_var)
            mse_loss = F.mse_loss(recon_img, label_images)
            ssim_loss = 1- ssim_loss_func(recon_img, label_images)
            loss = args.kl_scale * kl_loss + ssim_loss
            # loss = args.kl_scale * kl_loss + mse_loss

            # Perception loss
            if feature_extractor is not None:
                feat_in = torch.cat((recon_img, images), 0)
                feature_loss = feature_extractor(feat_in)
                loss += args.feature_scale * feature_loss
                data_logger["feature_loss"].append(feature_loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(vae_net.parameters(), 10)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_mse_loss += loss.item()
        train_ssim_loss += loss.item()
        train_kl_loss += loss.item()
        if feature_extractor is not None:
            train_feature_loss+=feature_loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_mse_loss = train_mse_loss / len(train_loader)
    avg_train_ssim_loss = train_ssim_loss / len(train_loader)
    avg_train_kl_loss = train_kl_loss / len(train_loader)
    if feature_extractor is not None:
            avg_train_feature_loss = train_feature_loss / len(train_loader)

    data_logger["avg_train_loss"].append(avg_train_loss)
    mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)
    mlflow.log_metric("avg_train_mse_loss", avg_train_mse_loss, step=epoch)
    mlflow.log_metric("avg_train_ssim_loss", avg_train_ssim_loss, step=epoch)
    mlflow.log_metric("avg_train_kl_loss", avg_train_kl_loss, step=epoch)
    if feature_extractor is not None:
            mlflow.log_metric("avg_train_feature_loss", avg_train_feature_loss, step=epoch)


def test(test_images, test_labels, vae_net, data_logger, optimizer, ssim_loss_func, device):
    vae_net.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # Save an example from testing and log a test loss
            recon_img, mu, log_var = vae_net(test_images.to(device))
            # data_logger['test_mse_loss'].append(F.mse_loss(recon_img,test_labels.to(device) ).item())
            avg_test_ssim_loss = 1 - ssim_loss_func(recon_img,test_labels.to(device) ).item()
            data_logger['test_ssim_loss'].append(avg_test_ssim_loss)
            mlflow.log_metric("avg_test_ssim_loss", avg_test_ssim_loss, step=epoch)

            img_cat = torch.cat((recon_img.cpu(), test_images), 2).float()
            vutils.save_image(img_cat,
                            "%s/%s/%s_%d_test.png" % (args.save_dir,
                                                        "Results",
                                                        args.model_name,
                                                        args.image_size),
                            normalize=True)

        # Keep a copy of the previous save in case we accidentally save a model that has exploded...
        if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
            shutil.copyfile(src=args.save_dir + "/Models/" + save_file_name + ".pt",
                            dst=args.save_dir + "/Models/" + save_file_name + "_copy.pt")

        # Save a checkpoint
        torch.save({
                    'epoch': epoch + 1,
                    'data_logger': dict(data_logger),
                    'model_state_dict': vae_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, args.save_dir + "/Models/" + save_file_name + ".pt")
        

        # Set the model back into training mode!!
        vae_net.train()

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("Using CUDA...")
        print('Using Cuda')

    # Create dataloaders
    # This code assumes there is no pre-defined test/train split and will create one for you
    print("-Target Image Size %d" % args.image_size)
    transform = transforms.Compose([transforms.Resize(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    #transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])

    train_input_dir = "./dataset/data/train/img"
    train_label_dir = "./dataset/data/train/ann"
    valid_input_dir = "./dataset/data/valid/img"
    valid_label_dir = "./dataset/data/valid/ann"

    train_dataset = hf.CustomDataset(input_dir=train_input_dir, label_dir=train_label_dir, transform=transform)
    valid_dataset = hf.CustomDataset(input_dir=valid_input_dir, label_dir=valid_label_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Get a test image batch from the test_loader to visualise the reconstruction quality etc
    dataiter = iter(test_loader)
    test_images, test_labels = next(dataiter)

    # Create AE network.
    vae_net = VAE(channel_in=test_images.shape[1],
                ch=args.ch_multi,
                #   blocks=args.block_widths,
                #   num_res_blocks=args.num_res_blocks,
                #   norm_type=args.norm_type,
                #   deep_model=args.deep_model,
                latent_channels=args.latent_channels
                ).to(device)

    # Setup optimizer
    optimizer = optim.Adam(vae_net.parameters(), lr=args.lr)

    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler()

    if args.norm_type == "bn":
        print("-Using BatchNorm")
    elif args.norm_type == "gn":
        print("-Using GroupNorm")
    else:
        ValueError("norm_type must be bn or gn")

    # Create the feature loss module if required
    if args.feature_scale > 0:
        feature_extractor = VGG19().to(device)
        print("-VGG19 Feature Loss ON")
    else:
        feature_extractor = None # type: ignore
        print("-VGG19 Feature Loss OFF")

    # Let's see how many Parameters our Model has!
    num_model_params = 0
    for param in vae_net.parameters():
        num_model_params += param.flatten().shape[0]

    print("-This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))
    fm_size = args.image_size//(2 ** len(args.block_widths))
    print("-The Latent Space Size Is %dx%dx%d!" % (args.latent_channels, fm_size, fm_size))

    # Create the save directory if it does not exist
    if not os.path.isdir(args.save_dir + "/Models"):
        os.makedirs(args.save_dir + "/Models")
    if not os.path.isdir(args.save_dir + "/Results"):
        os.makedirs(args.save_dir + "/Results")

    # Checks if a checkpoint has been specified to load, if it has, it loads the checkpoint
    # If no checkpoint is specified, it checks if a checkpoint already exists and raises an error if
    # it does to prevent accidental overwriting. If no checkpoint exists, it starts from scratch.
    save_file_name = args.model_name + "_" + str(args.image_size)
    if args.load_checkpoint:
        if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
            checkpoint = torch.load(args.save_dir + "/Models/" + save_file_name + ".pt",
                                    map_location="cpu")
            print("-Checkpoint loaded!")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            vae_net.load_state_dict(checkpoint['model_state_dict'])

            if not optimizer.param_groups[0]["lr"] == args.lr:
                print("Updating lr!")
                optimizer.param_groups[0]["lr"] = args.lr

            start_epoch = checkpoint["epoch"]
            data_logger = defaultdict(lambda: [], checkpoint["data_logger"])
        else:
            raise ValueError("Warning Checkpoint does NOT exist -> check model name or save directory")
    else:
        # If checkpoint does exist raise an error to prevent accidental overwriting
        if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
            raise ValueError("Warning Checkpoint exists -> add -cp flag to use this checkpoint")
        else:
            print("Starting from scratch")
            start_epoch = 0
            # Loss and metrics logger
            data_logger = defaultdict(lambda: [])
    print("")

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")
    # mlflow.set_tracking_uri(uri="sqlite:///mlflow.db")
    # Create a new MLflow Experiment
    mlflow.set_experiment("Anomaly Detection")

    params = {
    "epochs": "args.nepoch",
    "batch_size": args.batch_size,
    "learning_rate": args.lr,
    "feature_scale": args.feature_scale,
    }

    ssim_loss_func = pytorch_ssim.SSIM(window_size=11, size_average=True).to(device)

    # Start an MLflow run
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.set_tag("Test run", "Test run for mlflow and custom loss")
        # Start training loop
        for epoch in trange(start_epoch, args.nepoch, leave=False):
            train(train_loader,vae_net, data_logger,optimizer, ssim_loss_func, device, epoch)
            vae_net.train()

            # Save results and a checkpoint at regular intervals
            if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
                # In eval mode the model will use mu as the encoding instead of sampling from the distribution

                test(test_images, test_labels, vae_net, data_logger, optimizer, ssim_loss_func, device)

    
        mlflow.pytorch.log_model(vae_net, "final_model")
    
    print("End of Training!!")