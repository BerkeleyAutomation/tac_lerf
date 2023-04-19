from dataclasses import dataclass, field
from typing import Tuple, Type

import torch
import torchvision
import torchvision.transforms.functional as TF

from lerf.encoders.image_encoder import BaseImageEncoder, BaseImageEncoderConfig
from tacvis.lightning_modules import ContrastiveModule, RotationModule

@dataclass
class TacNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: TacNetwork)
    tac_dim: int = 128
    rgb_dim: int = 128
    rotations: Tuple[int] = (0)
    model_dir: str = "/home/abrashid/lerf/tac-lerf/models/contrastive564715/models/epoch=459-step=2300.ckpt"

class TacNetwork(BaseImageEncoder):
    def __init__(self, config: TacNetworkConfig):
        self.config = config
        self.PREPROC_IMG = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
            ])
        self.process = torchvision.transforms.Compose(
            [
                # torchvision.transforms.functional.rgb_to_grayscale(num_output_channels=3),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.model = ContrastiveModule.load_from_checkpoint(self.config.model_dir).eval().cuda()
        self.tac_enc = self.model.tac_enc
        self.img_enc = self.model.rgb_enc
        self.rotations = self.config.rotations

        """
        Incorporate recieving these tac batches from the viewer later if the results are good
            For now we can hard code them in
        """
        #Preprocess the tac_images
        tac_batches = "Path/to/tac_images"
        tac_batches = [TF.rotate(tac_batches, rot) for rot in self.rotations]
        tac_batches = torch.cat(tac_batches)  # num_rot*N dim
        with torch.no_grad():
            self.tac_embeds = self.tac_enc(tac_batches).cpu()  # rot*N dim

    @property
    def name(self) -> str:
        return "tac encoder"

    @property
    def embedding_dim(self) -> int:
        return self.config.tac_dim

    def encode_image(self, input):
        processed_input = self.process(input)
        return self.img_enc(processed_input).cpu().numpy()


    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        tac = self.tac_embeds.to(embed.dtype)
        dists = embed.mm(tac.T)  # M rot*N
        dists = dists[..., positive_id : positive_id + 1]
        # tac_batch = [self.PREPROC_IMG(tac) for tac in in_tac]  # cwh
        # return tf.tensordot(embed, )
