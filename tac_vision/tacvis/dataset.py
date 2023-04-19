import os
import os.path as osp
from torch.utils.data import Dataset
import numpy as np
import PIL
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import torch
import random
import matplotlib.pyplot as plt
from glob import glob
import cv2

# this one goes from raw uint8 image to tensor
PREPROC_IMG = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# this describes augments for training which jitter color
RGB_AUGMENTS = torchvision.transforms.Compose([
    torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=(0.8, 1.1),
                                                                           contrast=(.7, 1.3),
                                                                           saturation=0.2,
                                                                           hue=0.0)], p=.8),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(5, sigma=(.5, 1))],
                                       p=.5),
])
TAC_AUGMENTS = torchvision.transforms.Compose([
    torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=(0.9, 1.1),
                                                                           contrast=(.9, 1.1),
                                                                           saturation=0.2,
                                                                           hue=0.05)], p=.8),
    # torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(5,sigma=(.5,1))],p=.5)
])


def collate_paired(batch):
    # this function should be passed to the DataLoader collate_fn argument
    output = {'rgb': [], 'tac': []}
    for b in batch:
        output['rgb'].append(b['rgb'])
        output['tac'].append(b['tac'])
    output['rgb'] = torch.concat(output['rgb'])
    output['tac'] = torch.concat(output['tac'])
    return output


def collate_paired_rotation(batch):
    # this function should be passed to the DataLoader collate_fn argument
    output = {'rgb': [], 'tac': [], 'label': []}
    for b in batch:
        output['rgb'].append(b['rgb'])
        output['tac'].append(b['tac'])
        output['label'].append(b['label'])
    output['rgb'] = torch.concat(output['rgb'])
    output['tac'] = torch.concat(output['tac'])
    output['label'] = torch.concat(output['label'])

    return output


class PairedDatasetZoomedOut(Dataset):
    def __init__(self, params):
        super().__init__()
        print("\nUsing zoomed dataset, will perform cropping in the center\n")
        assert 'im_scale_range' in params
        assert 'augment' in params
        assert 'dataset_dir' in params
        assert 'spatial_aug' in params
        self.params = params
        self.augment = params['augment']
        self.repeat_rotations = params['repeat_rotations']
        self.data_dir = params['dataset_dir']
        if not isinstance(self.data_dir, list):
            self.data_dir = [self.data_dir]

        self.rgb_fnames = []
        self.tac_fnames = []
        for data_dir in self.data_dir:
            rgb_fnames = os.listdir(osp.join(data_dir, "images_rgb"))
            rgb_fnames = [f'{data_dir}/images_rgb/{e}' for e in rgb_fnames]
            self.rgb_fnames.extend(rgb_fnames)
            tac_fnames = [r.replace('rgb', 'tac') for r in rgb_fnames]
            self.tac_fnames.extend(tac_fnames)
        if self.params['use_background']:
            self.tac_background = Image.open(f'{self.data_dir}/tac_background.jpg')
            self.tac_background = PREPROC_IMG(self.tac_background)
        # self.data_cache={}#for now, no data cache bc the images are huge (4k imgs)

    def __getitem__(self, index):
        rgb_fname = self.rgb_fnames[index]
        tac_fname = self.tac_fnames[index]
        rgb = Image.open(rgb_fname)
        tac = Image.open(tac_fname)
        rgb = PREPROC_IMG(rgb)

        max_scale = self.params['rgb_size'][0] / (
                self.params['im_scale_range'][0] * min(rgb.shape[1], rgb.shape[2]))
        scaled_size = (int(max_scale * rgb.shape[1]), int(max_scale * rgb.shape[2]))
        rgb = TF.resize(rgb, scaled_size)
        im_size = rgb.shape
        rgb = TF.center_crop(rgb, np.ceil(
            np.sqrt(2) * self.params['im_scale_range'][1] * max(rgb.shape[1], rgb.shape[2])))
        tac = PREPROC_IMG(tac)
        if self.augment:
            # apply color jitters before spatial aug, we want to keep the same colors for each pair
            rgb = RGB_AUGMENTS(rgb)
            tac = TAC_AUGMENTS(tac)
        if self.params['use_background']:
            tac = tac - self.tac_background
        hpad = int(np.clip(max(tac.shape[1], tac.shape[2]) - tac.shape[1], 0, np.inf) / 2)
        wpad = int(np.clip(max(tac.shape[1], tac.shape[2]) - tac.shape[2], 0, np.inf) / 2)
        tac = TF.pad(tac, [wpad, hpad])
        tac = TF.rotate(tac, 90)
        rgbs, tacs = [], []
        for _ in range(self.params['repeat_rotations']):
            rgb2, tac2 = rgb.clone(), tac.clone()
            if self.augment:
                if 'spatial_aug' not in self.params or self.params['spatial_aug'] == 'none':
                    pass
                elif self.params['spatial_aug'] == 'independent':
                    r_amnt1 = random.uniform(-180, 180)
                    r_amnt2 = random.uniform(-180, 180)
                    rgb2 = TF.rotate(rgb2, r_amnt1)
                    tac2 = TF.rotate(tac2, r_amnt2)
                elif self.params['spatial_aug'] == 'paired':
                    # augmentations which preserve the relative orientations of rgb and tac images
                    r_amnt = random.uniform(-180, 180)
                    rgb2 = TF.rotate(rgb2, r_amnt)
                    tac2 = TF.rotate(tac2, r_amnt)
            # after rotating/flipping the images, we need to center-crop the rgb image
            im_scale = random.uniform(self.params['im_scale_range'][0],
                                      self.params['im_scale_range'][1])
            crop_size = im_scale * max(im_size[1], im_size[2])
            rgb2 = TF.center_crop(rgb2, crop_size)
            # finally, resize them to their specified dimensions
            rgb2 = TF.resize(rgb2, self.params['rgb_size'])
            tac2 = TF.resize(tac2, self.params['tac_size'])
            rgbs.append(rgb2)
            tacs.append(tac2)
        data = {"rgb": torch.stack(rgbs), "tac": torch.stack(tacs)}
        return data

    def __len__(self):
        return len(self.rgb_fnames)


class PairedDatasetRotation(Dataset):
    def __init__(self, params):
        super().__init__()
        print("\nUsing zoomed dataset, will perform cropping in the center\n")
        assert 'im_scale_range' in params
        assert 'augment' in params
        assert 'dataset_dir' in params
        assert 'spatial_aug' in params
        assert 'continuous' in params
        self.params = params
        self.augment = params['augment']
        self.data_dir = params['dataset_dir']
        self.continuous = params['continuous']
        self.do_buckets = params['do_buckets']
        if not isinstance(self.data_dir, list):
            self.data_dir = [self.data_dir]

        self.rgb_fnames = []
        self.tac_fnames = []
        for data_dir in self.data_dir:
            rgb_fnames = os.listdir(osp.join(data_dir, "images_rgb"))
            rgb_fnames = [f'{data_dir}/images_rgb/{e}' for e in rgb_fnames]
            self.rgb_fnames.extend(rgb_fnames)
            tac_fnames = [r.replace('rgb', 'tac') for r in rgb_fnames]
            self.tac_fnames.extend(tac_fnames)
        if self.params['use_background']:
            self.tac_background = Image.open(f'{self.data_dir}/tac_background.jpg')
            self.tac_background = PREPROC_IMG(self.tac_background)
        # self.data_cache={}#for now, no data cache bc the images are huge (4k imgs)
        self.rotation_list = params["rotation_list"]
        if self.do_buckets:
            self.bucket_interval = (self.rotation_list[1] - self.rotation_list[0]) / 2
            for i in range(len(self.rotation_list) - 1):
                assert self.rotation_list[i + 1] - self.rotation_list[
                    i] == 2 * self.bucket_interval, \
                    'all intervals must be identical for bucket formulation'

    def __getitem__(self, index):
        rgb_fname = self.rgb_fnames[index]
        tac_fname = self.tac_fnames[index]
        rgb = Image.open(rgb_fname)
        tac = Image.open(tac_fname)

        # index_paird = random.choice(np.arange(len(self.rotation_list)))
        # r_paird = self.rotation_list[index_paird]
        r_paird = random.uniform(0, 360)
        rgb = PREPROC_IMG(rgb)
        tac = PREPROC_IMG(tac)

        hpad = int(np.clip(max(tac.shape[1], tac.shape[2]) - tac.shape[1], 0, np.inf) / 2)
        wpad = int(np.clip(max(tac.shape[1], tac.shape[2]) - tac.shape[2], 0, np.inf) / 2)
        tac = TF.pad(tac, [wpad, hpad])  # TODO is this order swapped
        tac = TF.rotate(tac, 90)

        # for eval only
        # max_scale = self.params['rgb_size'][0] / (
        #             self.params['im_scale_range'][0] * min(rgb.shape[1], rgb.shape[2]))
        # scaled_size = (int(max_scale * rgb.shape[1]), int(max_scale * rgb.shape[2]))
        # rgb = TF.resize(rgb, scaled_size)
        # im_size = rgb.shape
        # rgb = TF.center_crop(rgb, np.ceil(
        #     np.sqrt(2) * self.params['im_scale_range'][1] * max(rgb.shape[1], rgb.shape[2])))
        #
        # rgb2, tac2 = rgb.clone(), tac.clone()
        # # only rotate the tac and predict what's the rotation angle
        # im_scale = random.uniform(self.params['im_scale_range'][0],
        #                           self.params['im_scale_range'][1])
        # crop_size = im_scale * max(im_size[1], im_size[2])
        # rgb2 = TF.center_crop(rgb2, crop_size)
        # # finally, resize them to their specified dimensions
        # rgb2 = TF.resize(rgb2, self.params['rgb_size'])
        # tac2 = TF.resize(tac2, self.params['tac_size'])
        #
        # data = {"rgb": rgb2, "tac": tac2}
        # return data

        # start edits
        # tac_np = (tac.numpy().transpose(1,2,0)*255).astype(np.uint8).copy()
        # cv2.putText(tac_np,f"__________",(10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        # tac = torch.tensor(tac_np.transpose(2,0,1).astype(np.float32)/255)

        # rgb_np = (rgb.numpy().transpose(1,2,0)*255).astype(np.uint8).copy()
        # cv2.putText(rgb_np,f"__________",(rgb_np.shape[0]//2,rgb_np.shape[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        # rgb = torch.tensor(rgb_np.transpose(2,0,1).astype(np.float32)/255)
        # end edits

        # if np.random.rand()>0.5:
        #     rgb=TF.hflip(rgb)
        #     tac=TF.hflip(tac)
        # #also vertically flip
        # if np.random.rand()>0.5:
        #     rgb=TF.vflip(rgb)
        #     tac=TF.vflip(tac)
        # then apply random rotation to both
        rgb = TF.rotate(rgb, r_paird)

        max_scale = self.params['rgb_size'][0] / (
                self.params['im_scale_range'][0] * min(rgb.shape[1], rgb.shape[2]))
        scaled_size = (int(max_scale * rgb.shape[1]), int(max_scale * rgb.shape[2]))
        rgb = TF.resize(rgb, scaled_size)
        im_size = rgb.shape
        rgb = TF.center_crop(rgb, np.ceil(
            np.sqrt(2) * self.params['im_scale_range'][1] * max(rgb.shape[1], rgb.shape[2])))

        # tac = torch.ones_like(tac)#use this line to replace the tactile image with a white image for sanity checking
        # first flip the images

        if self.augment:
            # apply color jitters before spatial aug, we want to keep the same colors for each pair
            rgb = RGB_AUGMENTS(rgb)
            tac = TAC_AUGMENTS(tac)

        if self.params['use_background']:
            tac = tac - self.tac_background

        rgbs, tacs, labels = [], [], []
        for r_index, r_amnt in enumerate(self.params['rotation_list']):
            rgb2, tac2 = rgb.clone(), tac.clone()

            if self.continuous:
                rot = random.uniform(0, 180)
            else:
                rot = r_amnt
                if self.do_buckets:
                    rot = rot + random.uniform(-self.bucket_interval, self.bucket_interval)

            tac2 = TF.rotate(tac2, r_paird + rot)

            # only rotate the tac and predict what's the rotation angle
            im_scale = random.uniform(self.params['im_scale_range'][0],
                                      self.params['im_scale_range'][1])
            crop_size = im_scale * max(im_size[1], im_size[2])
            rgb2 = TF.center_crop(rgb2, crop_size)
            # finally, resize them to their specified dimensions
            rgb2 = TF.resize(rgb2, self.params['rgb_size'])
            tac2 = TF.resize(tac2, self.params['tac_size'])

            rgbs.append(rgb2)
            tacs.append(tac2)

            if self.continuous:
                labels.append(torch.Tensor((rot / 180,)))
            else:
                label = torch.zeros(len(self.rotation_list))
                label[r_index] = 1.0
                labels.append(label)

        data = {"rgb": torch.stack(rgbs), "tac": torch.stack(tacs), "label": torch.stack(labels)}
        return data

    def __len__(self):
        return len(self.rgb_fnames)


class PairedDataset(Dataset):
    def __init__(self, params):
        super().__init__()
        self.augment = params['augment']
        self.params = params
        self.data_dir = params['dataset_dir']

        self.rgb_fnames = os.listdir(osp.join(self.data_dir, "images_rgb"))
        self.tac_fnames = [t.replace('rgb', 'tac') for t in self.rgb_fnames]
        self.data_cache = {}

    def __getitem__(self, index):
        if not self.params['cache_data'] or index not in self.data_cache:
            rgb_fname = f'{self.data_dir}/images_rgb/{self.rgb_fnames[index]}'
            tac_fname = f'{self.data_dir}/images_tac/{self.tac_fnames[index]}'

            rgb = Image.open(rgb_fname)
            tac = Image.open(tac_fname)
            # rotate the tac image by 90 deg
            tac = tac.rotate(90, expand=True)
            # resize the rgb image to match the tac image
            rgb_size = tuple((np.array(rgb.size) * (tac.size[0] / rgb.size[0])).astype(int))
            rgb = rgb.resize(rgb_size, PIL.Image.Resampling.BILINEAR)
            rgb = PREPROC_IMG(rgb)
            tac = PREPROC_IMG(tac)
            if self.params['cache_data']:
                self.data_cache[index] = (rgb, tac)
        else:
            rgb, tac = self.data_cache[index]

        if self.augment:
            rgb = RGB_AUGMENTS(rgb)
            tac = TAC_AUGMENTS(tac)
            if 'spatial_aug' not in self.params or self.params['spatial_aug'] == 'none':
                pass
            elif self.params['spatial_aug'] == 'independent':
                rgb = SPATIAL_AUGMENTS(rgb)
                tac = SPATIAL_AUGMENTS(tac)
            elif self.params['spatial_aug'] == 'paired':
                # augmentations which preserve the relative orientations of rgb and tac images
                if np.random.rand() > 0.5:
                    rgb = TF.hflip(rgb)
                    tac = TF.hflip(tac)
                # also vertically flip
                if np.random.rand() > 0.5:
                    rgb = TF.vflip(rgb)
                    tac = TF.vflip(tac)
        data = {"rgb": rgb, "tac": tac}
        return data

    def __len__(self):
        return len(self.rgb_fnames)


class PairedDatasetAE(PairedDataset):
    def __init__(self, data_dir):
        super(PairedDatasetAE, self).__init__(data_dir)

    def __getitem__(self, index):
        rgb_fname = f'{self.data_dir}/images_rgb/{self.rgb_fnames[index]}'
        tac_fname = f'{self.data_dir}/images_tac/{self.tac_fnames[index]}'

        rgb = Image.open(rgb_fname)
        tac = Image.open(tac_fname)
        # rotate the tac image by 90 deg
        tac = tac.rotate(90, expand=True)
        # resize the rgb image to match the tac image
        rgb_size = tuple((np.array(rgb.size) * (tac.size[1] / rgb.size[1])).astype(int))
        rgb = rgb.resize(rgb_size, PIL.Image.Resampling.BILINEAR)
        rgb = PREPROC_IMG(rgb)
        tac = PREPROC_IMG(tac)
        w_mid = rgb.shape[2] // 2
        rgb = rgb[:, :, w_mid - tac.shape[2] // 2:w_mid + tac.shape[2] // 2]
        if self.augment:
            rgb = RGB_AUGMENTS(rgb)
            tac = TAC_AUGMENTS(tac)
        data = {"rgb": rgb, "tac": tac}
        return data


if __name__ == '__main__':
    import yaml

    with open('config/train_contrastive.yaml', 'r') as stream:
        params = yaml.safe_load(stream)
    dataset = PairedDatasetZoomedOut(params)
    for d in dataset:
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(d['rgb'][0, ...].permute(1, 2, 0))
        tacim = d['tac'][0, ...]
        axs[1].imshow(tacim.permute(1, 2, 0))
        original = tacim + dataset.tac_background
        axs[1].imshow(original.permute(1, 2, 0))
        plt.show()
