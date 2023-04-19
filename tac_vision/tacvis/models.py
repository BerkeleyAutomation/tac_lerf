from turtle import forward
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18,resnet34
from torchvision.models.regnet import regnet_y_800mf,regnet_y_400mf,regnet_y_1_6gf
import numpy as np
from tacvis.unet_parts import DoubleConv, Down, UPNoRes, Up, OutConv
import matplotlib.pyplot as plt
from tacvis.dataset import PREPROC_IMG
import cv2
from typing import List
from skimage.util import view_as_windows
import torchvision.transforms.functional as TF


class Encoder(nn.Module):
    '''
    an image encoder which uses a resnet18 backbone
    '''

    def __init__(self, feature_dim=32,
                 model_type='resnet18',
                 **kwargs):
        super().__init__()
        if model_type == 'resnet18':
            self.resnet = resnet18(pretrained=True)
            self.resnet.fc = nn.Linear(512, feature_dim)
        elif model_type == 'regnet_400':
            self.resnet = regnet_y_400mf(pretrained=True)
            self.resnet.fc = nn.Linear(440, feature_dim)
        elif model_type == 'regnet_800':
            self.resnet = regnet_y_800mf(pretrained=True)
            self.resnet.fc = nn.Linear(784,feature_dim)
        elif model_type == 'regnet_1600':
            self.resnet = regnet_y_1_6gf(pretrained=True)
            self.resnet.fc = nn.Linear(888,feature_dim)

    def forward(self, batch: torch.tensor):
        '''
        takes in an image and returns the resnet18 features
        '''
        features = self.resnet(batch)
        feat_norm = torch.norm(features,dim=1)
        return features/feat_norm.view(features.shape[0],1)

    def encode(self, im: np.ndarray):
        '''
        takes in an image and returns the resnet18 features
        '''
        im = im.unsqueeze(0)
        with torch.no_grad():
            features = self(im)
        return features

    def save(self, save_name):
        torch.save(self.state_dict(), save_name)

    def load(self, save_name):
        self.load_state_dict(torch.load(save_name))


class RotationEncoder(nn.Module):
    '''
    an image encoder which uses a resnet18 backbone
    '''

    def __init__(self, feature_dim=32,
                 model_type='resnet18',
                 **kwargs):
        super().__init__()
        if model_type == 'resnet18':
            self.resnet = resnet18(pretrained=True)
            self.resnet.fc = nn.Linear(512, feature_dim)
        elif model_type == 'regnet_400':
            self.resnet = regnet_y_400mf(pretrained=True)
            self.resnet.fc = nn.Linear(440, feature_dim)
        elif model_type == 'regnet_800':
            self.resnet = regnet_y_800mf(pretrained=True)
            self.resnet.fc = nn.Linear(784,feature_dim)
        elif model_type == 'regnet_1600':
            self.resnet = regnet_y_1_6gf(pretrained=True)
            self.resnet.fc = nn.Linear(888,feature_dim)

    def forward(self, batch: torch.tensor):
        '''
        takes in an image and returns the resnet18 features
        '''
        features = self.resnet(batch)
        return features

    def encode(self, im: np.ndarray):
        '''
        takes in an image and returns the resnet18 features
        '''
        im = im.unsqueeze(0)
        with torch.no_grad():
            features = self(im)
        return features

    def save(self, save_name):
        torch.save(self.state_dict(), save_name)

    def load(self, save_name):
        self.load_state_dict(torch.load(save_name))


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        if 'rgb' in params["type"]:
            self.in_h = params['rgb_size'][0] // 4
            self.in_w = params['rgb_size'][1] // 4
            self.pad = [params['rgb_size'][0] % 4, params['rgb_size'][1] % 4]

        else:
            self.in_h = params['tac_size'][0] // 4
            self.in_w = params['tac_size'][1] // 4
            self.pad = [params['tac_size'][0] % 4, params['tac_size'][1] % 4]

        self.in_c = 32

        self.decoder_lin = nn.Sequential(
            nn.Linear(params['feature_dim'], 512),
            nn.ReLU(),
            nn.Linear(512, self.in_c * self.in_h * self.in_w),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(self.in_c, self.in_h, self.in_w))

        self.decoder_conv = nn.Sequential(
            # nn.ConvTranspose2d(32, 32, 2, stride=2, output_padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            nn.ConvTranspose2d(self.in_c, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 2, stride=2),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            # nn.ConvTranspose2d(8, 6, 2, stride=2),
            # nn.BatchNorm2d(6),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(6, 3, 2, stride=2, output_padding=self.pad),
            # nn.BatchNorm2d(3),
            # nn.ReLU(True),
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(3, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, out_channels=3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        if x.device != DEVICE:
            x = x.to(DEVICE)

        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = self.final_layer(x)
        return x

    def save(self, save_name):
        torch.save(self.state_dict(), save_name)

    def load(self, save_name):
        self.load_state_dict(torch.load(save_name))


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img: torch.tensor):
        x = self.encoder(img)
        return self.decoder(x), x

    def loss(self, data):
        recon, feature = self(data)
        return self.loss_fn(recon, data), recon, feature

    def save(self, save_name):
        torch.save(self.state_dict(), save_name)

    def load(self, save_name):
        self.load_state_dict(torch.load(save_name))

    def all_parameters(self):
        parameters = [{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}]
        return parameters


class AutoEncoder(BaseNet):
    def __init__(self, params, device, encoder=None):
        super().__init__()
        if encoder is None:
            self.encoder = Encoder(params)
        else:
            self.encoder = encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.decoder = Decoder(params)

        self.loss_fn = nn.MSELoss(reduction='mean')
        self.device = device


class UNetEncoder(nn.Module):
    """docstring for UNetEncoder."""

    def __init__(self, n_channels=3, bilinear=True):
        super(UNetEncoder, self).__init__()
        factor = 2 if bilinear else 1
        self.model = nn.Sequential(DoubleConv(n_channels, 16),
                                   Down(16, 32),
                                   Down(32, 64),
                                   Down(64, 128),
                                   Down(128, 256),
                                   Down(256, 512))

    def forward(self, batch: torch.tensor):
        return self.model(batch)

    def save(self, save_name):
        torch.save(self.state_dict(), save_name)

    def load(self, save_name):
        self.load_state_dict(torch.load(save_name))


class UNetDecoder(nn.Module):
    """docstring for UNetEncoder."""

    def __init__(self, n_classes=3, bilinear=True):
        super(UNetDecoder, self).__init__()
        factor = 2 if bilinear else 1
        self.model = nn.Sequential(
            UPNoRes(512, 256, bilinear),
            UPNoRes(256, 128, bilinear),
            UPNoRes(128, 64, bilinear),
            UPNoRes(64, 32, bilinear),
            UPNoRes(32, 16, bilinear),
            OutConv(16, n_classes), nn.Sigmoid())

    def forward(self, batch: torch.tensor):
        return self.model(batch)

    def save(self, save_name):
        torch.save(self.state_dict(), save_name)

    def load(self, save_name):
        self.load_state_dict(torch.load(save_name))


class UNetAE(BaseNet):
    def __init__(self, params, device, n_channels=3, n_classes=3, bilinear=True, encoder=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        if encoder is None:
            self.encoder = UNetEncoder(n_channels, bilinear)
        else:
            self.encoder = encoder
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.decoder = UNetDecoder(n_classes, bilinear)

        self.loss_fn = nn.MSELoss(reduction='mean')
        self.device = device


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()

        layers = []
        last_size = input_size
        for size in hidden_layers + [output_size]:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU(True))
            # layers.append(nn.Dropout(0.1))
            last_size = size

        layers.pop()
        layers.pop()
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Projector(BaseNet):
    def __init__(self, encoder, decoder, params):
        super(Projector, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.projector = MLP(params["feature_dim"], params["feature_dim"],
                             params["hidden_size"]).to(DEVICE)
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, data):
        latent = self.encoder(data)
        dec_latent = self.projector(latent)
        rec = self.decoder(dec_latent)
        return rec, dec_latent

    def loss(self, enc_input, dec_output):
        recon, feature = self(enc_input)
        return self.loss_fn(recon, dec_output), recon, feature


def rank_distance(img_feats, tac_feats):
    '''
    returns for each image in img_batch, the indices of the closest tac_batch images in feature space
    '''
    # dists is a BxB matrix representing pairwise distance in feature space
    dists = img_feats.mm(tac_feats.t())
    _, indices = torch.sort(dists, dim=-1, descending=True)
    return indices


def get_crop_embeddings(in_img, img_enc, patch_size, stride, crop_ratio, grayscale=True):
    assert in_img.shape[2] == 3, 'in image must have h,w,c'
    img_enc.eval()
    rgb_w, rgb_h = in_img.shape[1], in_img.shape[0]
    scale = patch_size[0] / (crop_ratio * in_img.shape[0])
    width = int(rgb_w * scale)
    height = int(rgb_h * scale)
    dim = (width, height)
    # resize image
    in_img = cv2.resize(in_img, dim, interpolation=cv2.INTER_AREA)
    rgb_crops = []
    for i in range(3):  # this is for 3 channels
        rgb_crop = view_as_windows(in_img[..., i], patch_size, stride)
        rgb_crops.append(np.concatenate(rgb_crop, axis=0))
    rgb_crops_color = np.stack(rgb_crops, axis=1)  # M C W H
    if grayscale:
        rgb_crops = np.mean(rgb_crops_color, axis=1)[:, None, ...].astype(np.uint8)  # M C W H
        rgb_crops = np.repeat(rgb_crops, 3, axis=1)
    eval_batch = 512
    with torch.no_grad():
        rgb_feats = []
        for batch_idx in range(0, rgb_crops.shape[0], eval_batch):
            print("batch", batch_idx)
            endid = min(batch_idx + eval_batch, rgb_crops.shape[0])
            rgb_batches = torch.as_tensor(rgb_crops[batch_idx:endid, ...]).to(
                dtype=torch.get_default_dtype()).cuda().div(255)
            rgb_feats.append(img_enc(rgb_batches).cpu())
        rgb_feats = torch.cat(rgb_feats, dim=0)
    return rgb_feats, rgb_crops_color


def compute_similarity_heatmap(in_img: np.ndarray, in_tac: List[np.ndarray], crop_ratios: list,
                               rotations: list, img_enc: Encoder, tac_enc: Encoder, im_size,
                               stride=10):
    '''
    in_img: np.ndarray of shape (H,W,C)
    in_tac: np.ndarray of shape (N,H,W,C)
    scale: describes what percent of the original image the the tactile reading is
    img_enc: an image encoder
    tac_enc: a tactile encoder
    '''

    device = next(img_enc.parameters()).device
    img_enc.eval()
    tac_enc.eval()
    tac_batch = [PREPROC_IMG(tac) for tac in in_tac]  # cwh
    tac_batch = torch.stack(tac_batch).to(device)  # N*c*w*h
    hpad = int(
        np.clip(max(tac_batch.shape[2], tac_batch.shape[3]) - tac_batch.shape[2], 0, np.inf) / 2)
    wpad = int(
        np.clip(max(tac_batch.shape[2], tac_batch.shape[3]) - tac_batch.shape[3], 0, np.inf) / 2)
    tac_batch = TF.pad(tac_batch, [wpad, hpad])
    tac_batch = TF.rotate(tac_batch, 90)
    tac_batch = TF.resize(tac_batch, im_size)
    tac_batches = [TF.rotate(tac_batch, rot) for rot in rotations]
    tac_batches = torch.cat(tac_batches)
    eval_batch = 512

    def calc_heatmap(rgb_global: np.ndarray, stride, tac_feat_local):
        patch_size = (tac_batches.shape[2], tac_batches.shape[3])
        rgb_crops = []
        for i in range(3):  # this is for 3 channels
            rgb_crop = view_as_windows(rgb_global[..., i], patch_size, stride)
            rgb_crops.append(np.concatenate(rgb_crop, axis=0))
        rgb_crops = np.stack(rgb_crops, axis=1)  # M C W H
        rgb_crops = np.mean(rgb_crops, axis=1)[:, None, ...].astype(np.uint8)  # M C W H
        rgb_crops = np.repeat(rgb_crops, 3, axis=1)
        with torch.no_grad():
            # rgb_batches = PREPROC_IMG(rgb_crops) #M C W H
            rgb_feats = []
            for batch_idx in range(0, rgb_crops.shape[0], eval_batch):
                endid = min(batch_idx + eval_batch, rgb_crops.shape[0])
                rgb_batches = torch.as_tensor(rgb_crops[batch_idx:endid, ...]).to(
                    dtype=torch.get_default_dtype()).div(255).cuda()
                rgb_feats.append(img_enc(rgb_batches).cpu())
            rgb_feats = torch.cat(rgb_feats, dim=0)
            dists = rgb_feats.mm(tac_feat_local.t())  # M rot*N

            dists_with_max_rot = \
            torch.max(dists.reshape(dists.shape[0], len(rotations), -1), axis=1)[0].cpu().reshape(
                rgb_crop.shape[0], rgb_crop.shape[1], -1)  # (rgb patch numbers) N
            pad_size = int(patch_size[0] // (2 * stride))
            dists_with_max_rot = torch.nn.functional.pad(dists_with_max_rot.permute(2, 0, 1),
                                                         [pad_size, pad_size, pad_size, pad_size])

            dists_max = torch.max(dists, axis=0)[0].cpu().reshape(len(rotations), -1).t()  # N rot
            ind_max_rot = torch.argmax(dists, axis=0).cpu().reshape(len(rotations),
                                                                    -1).t()  # rgb indices for N rot tactile images

            rot_ind = torch.argmax(dists_max, axis=1)  # rotation for each tactile, N
            rgb_ind = [ind_max_rot[i, rot_ind[i]] for i in range(ind_max_rot.shape[0])]

        rgb_matches = [rgb_crops[i] for i in rgb_ind]
        rot_matches = [rotations[i] for i in rot_ind]
        return dists_with_max_rot.cpu().numpy(), rgb_matches, rot_matches

    step = stride
    rgb_h, rgb_w = in_img.shape[0], in_img.shape[1]
    with torch.no_grad():
        tac_feat = tac_enc(tac_batches).cpu()  # rot*N dim

    heatmap_list = []
    rgb_crop_list = []
    rotation_list = []
    for crop_ratio in crop_ratios:
        scale = tac_batch.shape[2] / (crop_ratio * rgb_w)
        width = int(rgb_w * scale)
        height = int(rgb_h * scale)
        dim = (width, height)

        # resize image
        resized = cv2.resize(in_img, dim, interpolation=cv2.INTER_AREA)
        heatmaps, best_rgb_crop, best_rotation = calc_heatmap(resized, int(step * scale), tac_feat)
        heatmaps = [
            cv2.resize(heatmap, (in_img.shape[1], in_img.shape[0]), interpolation=cv2.INTER_AREA)
            for heatmap in heatmaps]
        # heatmaps N (patch numbers), rgb_crop: N C W H, rotation: N

        heatmap_list.append(heatmaps)
        rgb_crop_list.append(best_rgb_crop)
        rotation_list.append(best_rotation)
    heatmap_list = np.array(heatmap_list)
    heatmap_list = heatmap_list.transpose(1, 0, 2, 3)
    # heatmap output is len(scales) x len(tac_ims) x height x width
    crops = np.array(rgb_crop_list).transpose(1, 0, 2, 3, 4)
    rotations = np.array(rotation_list).transpose(1, 0)
    return heatmap_list, crops, rotations
