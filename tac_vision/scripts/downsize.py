from PIL import Image
import os
import os.path as osp
data_dir='data/zoomed_out'

rgb_fnames = os.listdir(osp.join(data_dir, "images_rgb"))
for name in rgb_fnames:
    rgb = Image.open(f'{data_dir}/images_rgb/{name}')
    new_rgb=rgb.resize((2048,1080))
    newname=f'{data_dir}/images_rgb/{name}'
    new_rgb.save(newname)

# tac_fnames = os.listdir(osp.join(data_dir, "images_tac"))
# for name in tac_fnames:
#     tac = Image.open(f'{data_dir}/images_tac/{name}')
#     newname=f'{data_dir}/images_tac/{name}'
#     newname = newname.replace('png','jpg')
#     tac.save(newname)