from dataclasses import dataclass, field
from typing import Tuple, Type
from PIL import Image

import torch
import torchvision
import torchvision.transforms.functional as TF

import numpy as np

from lerf.encoders.image_encoder import BaseImageEncoder, BaseImageEncoderConfig
from tacvis.lightning_modules import ContrastiveModule, RotationModule
from tacvis.dataset import PREPROC_IMG, TAC_AUGMENTS


@dataclass
class TacNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: TacNetwork)
    tac_dim: Tuple[int] = (128, 128)
    rgb_dim: Tuple[int] = (128, 128)
    encode_dim: int = 8
    rotations: Tuple[int] = (0,)
    model_dir: str = "/home/abrashid/lerf/models/contrastive564715/models/epoch=459-step=2300.ckpt"

class TacNetwork(BaseImageEncoder):
    def __init__(self, config: TacNetworkConfig):
        super().__init__()
        self.config = config
        self.PREPROC_IMG = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
            ])
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.model = ContrastiveModule.load_from_checkpoint(self.config.model_dir).eval().cuda()
        self.model.eval_mode = True
        self.device = next(self.model.parameters()).device
        self.tac_enc = self.model.tac_enc
        self.img_enc = self.model.rgb_enc
        self.tac_size = self.config.tac_dim
        self.rgb_size = self.config.rgb_dim
        self.rotations = self.config.rotations

        """
        Incorporate recieving these tac batches from the viewer later if the results are good
            For now we can hard code them in
        """
        #Preprocess the tac_images
        # tac_batches = "Path/to/tac_images"
        self.positives = ["/home/abrashid/lerf/data/tac_data/test_data/images_tac/image_0_tac.jpg",
                          "/home/abrashid/lerf/data/tac_data/heatmap/images_set_0/image_tac_0.jpg",
                          "/home/abrashid/lerf/data/tac_data/test_data/images_tac/image_49_tac.jpg",
                          "/home/abrashid/lerf/data/tac_data/heatmap/images_set_7/image_tac_0.jpg",
                          "/home/abrashid/lerf/data/tac_data/heatmap/images_set_6/image_tac_0.jpg", 
                          "/home/abrashid/lerf/data/tac_data/towel_rotated_yellow/images_set_2/image_tac_1.jpg",
                          "/home/abrashid/lerf/data/tac_data/towel_rotated_yellow/images_set_2/image_tac_2.jpg",
                          "/home/abrashid/lerf/data/tac_data/towel_rotated_yellow/images_set_2/image_tac_3.jpg"]

        self.tac_list = []
        # path = "/home/abrashid/lerf/data/tac_data/test_data/images_tac/image_0_tac.jpg"
        for path in self.positives:
            self.tac_list.append(np.asarray(Image.open(path)))
        self.tac_batches = self.preprocess_tac(self.tac_list)
        
        with torch.no_grad():
            self.tac_embeds = self.tac_enc(self.tac_batches)  # rot*N x dim
        
    def preprocess_tac(self, tac_list, do_aug=False, do_preproc=True):
        '''
        input: list of HxWxC np arrays
        output: Nx3xWxW torch tensor, where N == len(self.rotations)*len(tac_list)
        '''
        tac_batch = tac_list
        if do_preproc:
            tac_batch = [PREPROC_IMG(tac) for tac in tac_list]  # cwh
        if do_aug:
            tac_batch = [TAC_AUGMENTS(tac) for tac in tac_list]
        tac_batch = torch.stack(tac_batch).to(self.device)  # N*c*w*h
        hpad = int(np.clip(max(tac_batch.shape[2], tac_batch.shape[3]) - tac_batch.shape[2], 0,
                        np.inf) / 2)
        wpad = int(np.clip(max(tac_batch.shape[2], tac_batch.shape[3]) - tac_batch.shape[3], 0,
                        np.inf) / 2)
        tac_batch = TF.pad(tac_batch, [wpad, hpad])
        tac_batch = TF.rotate(tac_batch, 90)
        tac_batch = TF.resize(tac_batch, self.tac_size)
        # tac_batches = [TF.rotate(tac_batch,rot) for rot in self.rotations]
        # tac_batches = torch.cat(tac_batches) #num_rot*N dim
        return tac_batch

    def preprocess_rgb(self, rgb_list, im_scale, rgb_size, grayscale=True):
        '''
        input: list of HxWxC np arrays
        output: Nx3xWxW torch tensor, where N == len(self.rotations)*len(rgb_list)
        '''
        rgb_batch = [PREPROC_IMG(rgb) for rgb in rgb_list]  # cwh
        rgb_batch = torch.stack(rgb_batch).to(self.device)  # N*c*w*h
        if grayscale:
            rgb_batch = TF.rgb_to_grayscale(rgb_batch, num_output_channels=3)
        max_scale = rgb_size[0] / (im_scale * min(rgb_batch.shape[2], rgb_batch.shape[3]))
        scaled_size = (int(max_scale * rgb_batch.shape[2]), int(max_scale * rgb_batch.shape[3]))
        rgb = TF.resize(rgb_batch, scaled_size)
        im_size = rgb_batch.shape[1:]
        rgb = TF.center_crop(rgb, np.ceil(np.sqrt(2) * im_scale * max(rgb.shape[2], rgb.shape[3])))
        crop_size = im_scale * max(im_size[1], im_size[2])
        rgb = TF.center_crop(rgb, crop_size)
        # finally, resize them to their specified dimensions
        rgb = TF.resize(rgb, rgb_size)
        return rgb

    @property
    def name(self) -> str:
        return "tac encoder"

    @property
    def embedding_dim(self) -> int:
        return self.config.encode_dim

    def encode_image(self, input):
        processed_input = self.process(input)
        # f = open("/home/abrashid/lerf/tac_encoder_log.txt", "a")
        # f.write(str(type(input)) + "   " + str(input.shape) + "\n")
        # f.close()
        return self.img_enc(processed_input)


    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        # embed dim - (# of rays in batch) x embed_dim

        # f = open("/home/abrashid/lerf/tac_encoder_log.txt", "a")
        # f.write(str(embed.get_device()) + "   " + str(embed.shape) + "\n")
        # f.write(str(self.tac_embeds.get_device()) + "   " + str(self.tac_embeds.shape) + "\n")
        # f.write("\n")
        # f.close()

        tac = self.tac_embeds.to(embed.dtype) # (#Tac_img*rot) x dim
        sim = embed.mm(tac.T)  # (# of rays in batch) x (#Tac_img*rot)
        sim = sim[..., positive_id : positive_id + 1] # (# of rays in batch) x 1
        return sim

if __name__ == '__main__':
    x = TacNetwork(TacNetworkConfig())