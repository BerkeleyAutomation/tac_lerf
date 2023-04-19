from json import load
from PIL import Image
from tacvis.dataset import PREPROC_IMG, TAC_AUGMENTS
from tacvis.capture import DataCaptureUR5, l_p
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import os.path as osp
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from scipy import ndimage
import torch
from skimage.util import view_as_windows
from autolab_core import RigidTransform
import cv2
from typing import List
from tacvis.lightning_modules import ContrastiveModule, RotationModule
from sklearn.manifold import TSNE
from matplotlib.backend_bases import MouseButton
from scipy.spatial import geometric_slerp
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tacvis.dataset import PairedDatasetZoomedOut, collate_paired
from scipy import ndimage
import glob


class Visualizer():
    # TODO move tsne here
    pass


class Tester():
    def __init__(self, params) -> None:
        self.params = params
        self.model = ContrastiveModule.load_from_checkpoint(params["encoder"]["model_dir"],
                                                            strict=True).eval().cuda()
        if self.params["encoder"]["rot_model_dir"] is not None:
            rot_dir = glob.glob(params["encoder"]["rot_model_dir"] + '*')
            print(rot_dir[0])
            self.rotation_net = RotationModule.load_from_checkpoint(rot_dir[0],
                                                                    strict=True).eval().cuda()
        else:
            self.rotation_net = None
        self.tac_enc = self.model.tac_enc
        self.img_enc = self.model.rgb_enc
        self.device = next(self.model.parameters()).device
        self.save_dir = self.params["out_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/images_rgb", exist_ok=True)
        self.load_dirs = self.params["load_dirs"]
        self.rgb_size = self.params["encoder"]["rgb_size"]
        self.tac_size = self.params["encoder"]["tac_size"]
        self.rotations = self.params["rotation"]
        self.crop_ratios = self.params["crop_ratios"]
        self.grayscale = self.params["grayscale"]
        self.eval_batch = self.params["eval_batch"]
        self.stride = DataCaptureUR5.RES[0] // self.params["stride"]

    def load_data(self, load_dir):
        in_img = np.asarray(Image.open(f'{load_dir}/image_global.jpg'))
        rgb_list = []
        tac_list = []
        for filename in os.listdir(load_dir):
            if 'rgb' in filename:
                rgb = np.asarray(Image.open(f'{load_dir}/{filename}'))
                tac = np.asarray(Image.open(f'{load_dir}/{filename.replace("rgb", "tac")}'))
                rgb_list.append(rgb)
                tac_list.append(tac)
        return in_img, rgb_list, tac_list

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

    def get_crop_embeddings(self, in_img, crop_ratio, stride, rgb_enc=None):
        assert in_img.shape[2] == 3, 'in image must have h,w,c'
        patch_size = self.tac_size
        if rgb_enc is not None:
            rgb_enc.eval()

        self.img_enc.eval()

        rgb_h, rgb_w = in_img.shape[0], in_img.shape[1]
        scale = patch_size[0] / (crop_ratio * rgb_w)
        width = int(rgb_w * scale)
        height = int(rgb_h * scale)
        dim = (width, height)
        # resize image
        in_img = cv2.resize(in_img, dim, interpolation=cv2.INTER_AREA)

        # max_scale = in_img.shape[0]/(crop_ratio*min(in_img.shape[0],in_img.shape[1]))
        # scaled_size = (int(max_scale*in_img.shape[0]),int(max_scale*in_img.shape[1]))
        # in_img = cv2.resize(in_img, scaled_size, interpolation = cv2.INTER_AREA)

        rgb_crops = []
        for i in range(3):  # this is for 3 channels
            rgb_crop = view_as_windows(in_img[..., i], patch_size, stride)
            rgb_crops.append(np.concatenate(rgb_crop, axis=0))
        rgb_crops_color = np.stack(rgb_crops, axis=1)  # M C W H
        if self.grayscale:
            rgb_crops = np.mean(rgb_crops_color, axis=1)[:, None, ...].astype(np.uint8)  # M C W H
            rgb_crops = np.repeat(rgb_crops, 3, axis=1)

        with torch.no_grad():
            rgb_feats = []
            rgb_feats_rot = []
            for batch_idx in range(0, rgb_crops.shape[0], self.eval_batch):
                endid = min(batch_idx + self.eval_batch, rgb_crops.shape[0])
                rgb_batches = torch.as_tensor(rgb_crops[batch_idx:endid, ...]).to(
                    dtype=torch.get_default_dtype()).cuda().div(255)
                rgb_feats.append(self.img_enc(rgb_batches).cpu())
                if rgb_enc is not None:
                    rgb_feats_rot.append(rgb_enc(rgb_batches).cpu())
            rgb_feats = torch.cat(rgb_feats, dim=0)
        if rgb_enc is not None:
            return rgb_feats, rgb_crops_color, rgb_crop.shape, torch.cat(rgb_feats_rot, dim=0)
        else:
            return rgb_feats, rgb_crops_color, rgb_crop.shape

    def eig_decomp(self, in_imgs, in_embeddings, n=3):
        # visualize the eigen decomp of the in embeddings with the first 3 eigen vectors shown
        _, sm, vh = np.linalg.svd(in_embeddings)
        resized = []
        for i in range(in_imgs.shape[0]):
            resized.append(cv2.resize(in_imgs[i, ...].transpose(1, 2, 0), (32, 32)))
        in_imgs = np.stack(resized, axis=0)
        print("Singular values are", sm)
        for i in range(n):
            fig, ax = plt.subplots(1)
            vec = vh[i, :]
            dotprods = in_embeddings.dot(vec)
            for im_id in range(in_imgs.shape[0]):
                im = in_imgs[im_id, ...]
                ranking = dotprods[im_id]
                im = OffsetImage(im, zoom=.9)
                ab = AnnotationBbox(im, (ranking, np.random.uniform(0, 1)), xycoords='data',
                                    frameon=False)
                ab.index = i
                ax.add_artist(ab)
            ax.update_datalim([(-1, 0), (1, 1)])
            ax.autoscale()
            plt.show()

    def eig_from_dirs(self, load_dirs):
        for cp in self.crop_ratios:
            img_crops, in_embeddings = self.embeddings_from_dirs(load_dirs, cp)
            self.eig_decomp(img_crops, in_embeddings)

    def eig_from_batch(self, batch):
        in_imgs, in_embeddings = self.embeddings_from_batch(batch, 'rgb')
        self.eig_decomp(in_imgs, in_embeddings)

    def scatter_images(self, tsne_features, images, ax, fig, original_embeddings):
        # tsne is Nx2
        # images is Nx3xhxw
        # first resize the images to be easier to draw
        resized = []
        for i in range(images.shape[0]):
            # resized.append(cv2.resize(images[i, ...].transpose(1, 2, 0), (32, 32)))
            resized.append(cv2.resize(images[i, ...].transpose(1, 2, 0), (64, 64)))

        images = np.stack(resized, axis=0)
        artists = []
        select1, select2 = None, None
        default_zoom = .9

        def artist_cb(art, event):
            nonlocal select1, select2, artists
            if art.contains(event)[0]:
                if event.button == MouseButton.LEFT and select1 is None:
                    art.get_children()[0].set_zoom(default_zoom * 2)
                    select1 = art.index
                elif event.button == MouseButton.RIGHT and select2 is None:
                    art.get_children()[0].set_zoom(default_zoom * 2)
                    select2 = art.index
                fig.canvas.draw()
                return True, {}
            else:
                return False, {}

        def on_press(event):
            nonlocal select1, select2, artists, artist_cb
            if event.key == 'i':
                artists = []
                ax.clear()
                vec1 = original_embeddings[select1, :].astype(np.float64)
                vec2 = original_embeddings[select2, :].astype(np.float64)
                vec1 /= np.linalg.norm(vec1)
                vec2 /= np.linalg.norm(vec2)
                interps = geometric_slerp(vec1, vec2, np.linspace(0, 1, 40)).astype(np.float32)
                new_embeddings = np.concatenate((original_embeddings, interps), axis=0)
                tsne = TSNE(metric='cosine', init='pca')
                vis_vectors = tsne.fit(new_embeddings).embedding_
                # plot the originals with the axes objects
                for i in range(original_embeddings.shape[0]):
                    im = OffsetImage(images[i, ...], zoom=default_zoom)
                    ab = AnnotationBbox(im, (vis_vectors[i, 0], vis_vectors[i, 1]), xycoords='data',
                                        frameon=False, picker=artist_cb)
                    ab.index = i
                    ax.add_artist(ab)
                    artists.append(ab)
                # plot the interpolation data
                ax.scatter(vis_vectors[original_embeddings.shape[0]:, 0],
                           vis_vectors[original_embeddings.shape[0]:, 1], marker='.')
                ax.update_datalim(vis_vectors)
                ax.autoscale()
                fig.canvas.draw()
                select1, select2 = None, None
            elif event.key == 'r':
                for a in artists:
                    a.get_children()[0].set_zoom(default_zoom)
                select1, select2 = None, None
                fig.canvas.draw()

        fig.canvas.mpl_connect('key_press_event', on_press)
        for i in range(tsne_features.shape[0]):
            im = OffsetImage(images[i, ...], zoom=default_zoom)
            ab = AnnotationBbox(im, (tsne_features[i, 0], tsne_features[i, 1]), xycoords='data',
                                frameon=False, picker=artist_cb)
            ab.index = i
            ax.add_artist(ab)
            artists.append(ab)
        ax.update_datalim(tsne_features)
        ax.autoscale()

    def get_adaptive_thresh(self, query_embedding, cached_tac_file='/raid/jkerr/tac_vision/data/ur_data/tac_embedding_564715.npy'):
        """
        as an upper bound on vision-guided search, calculate the max distance between query embedding and the 
        tactile train set embeddings (cached as np array)
        """
        tac_embeddings = np.load(cached_tac_file)
        dists = query_embedding.dot(tac_embedding.T)[0]
        return max(dists)
    
    def embeddings_from_batch(self, batch, vis):
        # returns images and embeddings (tactile and images) from the batch
        # vis can be 'tac', 'rgb', or 'both'
        tac_batches = batch['tac'].cuda()
        img_crops = batch['rgb'].cuda()
        with torch.no_grad():
            tac_embedding = self.tac_enc(tac_batches).cpu().numpy()
            img_embedding = self.img_enc(img_crops).cpu().numpy()
        if vis == 'both':
            in_embeddings = np.concatenate((img_embedding, tac_embedding), axis=0)
            in_imgs = torch.concat((img_crops, tac_batches), dim=0).cpu().numpy()
        elif vis == 'rgb':
            in_embeddings = img_embedding
            in_imgs = img_crops.cpu().numpy()
        elif vis == 'tac':
            in_embeddings = tac_embedding
            in_imgs = tac_batches.cpu().numpy()
        return in_imgs, in_embeddings

    def embeddings_from_dirs(self, load_dirs, cp):
        # returns images and embeddings from a set of load dirs at the given crop ratio (cp)
        imgs, tacs = [], []
        for load_dir in load_dirs:
            in_img, _, tac_list = self.load_data(load_dir)
            imgs.append(in_img)
            tacs.append(tac_list)
        tacs = np.concatenate(tacs, axis=0)
        tac_batches = self.preprocess_tac(tacs)
        with torch.no_grad():
            tac_embedding = self.model.tac_enc(tac_batches).cpu().numpy()
            img_crops = []
            in_embeddings = []
            for i in imgs:
                in_embedding, img_crop, _ = self.get_crop_embeddings(i, cp, self.stride)
                img_crops.append(img_crop)
                in_embeddings.append(in_embedding)
            img_crops = np.concatenate(img_crops, axis=0)
            in_embeddings = np.concatenate(in_embeddings, axis=0)
            img_crops = img_crops.astype(np.float32) / 255
            in_embeddings = np.concatenate((in_embeddings, tac_embedding), axis=0)
            img_crops = np.concatenate((img_crops, tac_batches.cpu().numpy()), axis=0)
        return img_crops, in_embeddings

    def tsne_from_dir(self, load_dirs):
        # TODO: PCA and isomap .
        fig, axs = plt.subplots(len(self.crop_ratios))
        fig.set_size_inches(16, 4)
        for idx, cp in enumerate(self.crop_ratios):
            if len(self.crop_ratios) == 1:
                ax = axs
            else:
                ax = axs[idx]
            tsne = TSNE(metric='cosine', init='pca')
            img_crops, in_embeddings = self.embeddings_from_dirs(load_dirs, cp)
            vis_vectors = tsne.fit(in_embeddings).embedding_
            self.scatter_images(vis_vectors, img_crops, ax, fig, in_embeddings)
            # dimension of this will be Nxcomponents
        plt.show()

    def tsne_from_batch(self, batch):
        in_imgs, in_embeddings = self.embeddings_from_batch(batch)
        tsne = TSNE(metric='cosine', init='pca')
        vis_vectors = tsne.fit(in_embeddings).embedding_
        fig, ax = plt.subplots(1)
        self.scatter_images(vis_vectors, in_imgs, ax, fig, in_embeddings)
        plt.show()

    def compute_similarity_heatmap(self, in_img: np.ndarray, tac_list: List[np.ndarray]):
        '''
        in_img: np.ndarray of shape (H,W,C)
        in_tac: np.ndarray of shape (N,H,W,C)
        scale: describes what percent of the original image the the tactile reading is
        img_enc: an image encoder
        tac_enc: a tactile encoder
        '''

        def heatmap_cal(dists):
            dists_with_max_rot = \
            torch.max(dists.reshape(dists.shape[0], len(self.rotations), -1), axis=1)[
                0].cpu().reshape(rgb_crop_shape[0], rgb_crop_shape[1], -1)  # (rgb patch numbers) N
            pad_size = int(self.tac_size[0] // (2 * stride))
            heatmaps = torch.nn.functional.pad(dists_with_max_rot.permute(2, 0, 1),
                                               [pad_size, pad_size, pad_size,
                                                pad_size]).cpu().numpy()
            heatmaps = [cv2.resize(heatmap, (in_img.shape[1], in_img.shape[0]),
                                   interpolation=cv2.INTER_AREA) for heatmap in heatmaps]

            dists_max = torch.max(dists, axis=0)[0].cpu().reshape(len(self.rotations),
                                                                  -1).t()  # N rot
            ind_max_rot = torch.argmax(dists, axis=0).cpu().reshape(len(self.rotations),
                                                                    -1).t()  # rgb indices for N rot tactile images

            rot_ind = torch.argmax(dists_max, axis=1)  # rotation for each tactile, N
            rgb_ind = [ind_max_rot[i, rot_ind[i]] for i in range(ind_max_rot.shape[0])]

            best_rgb_crop = [rgb_crops_color[i] for i in rgb_ind]
            best_rotation = [self.rotations[i] for i in rot_ind]

            return heatmaps, best_rgb_crop, best_rotation

        tac_batches = self.preprocess_tac(tac_list)
        tac_batches = [TF.rotate(tac_batches, rot) for rot in self.rotations]
        tac_batches = torch.cat(tac_batches)  # num_rot*N dim
        with torch.no_grad():
            tac_feat = self.tac_enc(tac_batches).cpu()  # rot*N dim
        heatmap_list = []
        rgb_crop_list = []
        rotation_list = []

        for cp in self.crop_ratios:
            scale = self.tac_size[0] / (cp * in_img.shape[1])
            stride = int(scale * self.stride)
            rgb_feats, rgb_crops_color, rgb_crop_shape = self.get_crop_embeddings(in_img, cp,
                                                                                  stride)

            dists = rgb_feats.mm(tac_feat.t())  # M rot*N

            # dists_with_max_rot = torch.max(dists.reshape(dists.shape[0],len(self.rotations),-1),axis=1)[0].cpu().reshape(rgb_crop_shape[0],rgb_crop_shape[1],-1) #(rgb patch numbers) N
            # pad_size = int(self.tac_size[0]//(2*stride))
            # heatmaps = torch.nn.functional.pad(dists_with_max_rot.permute(2,0,1),[pad_size,pad_size,pad_size,pad_size]).cpu().numpy()
            # heatmaps = [cv2.resize(heatmap,(in_img.shape[1],in_img.shape[0]), interpolation = cv2.INTER_AREA) for heatmap in heatmaps]

            # dists_max = torch.max(dists,axis=0)[0].cpu().reshape(len(self.rotations),-1).t() # N rot
            # ind_max_rot = torch.argmax(dists,axis=0).cpu().reshape(len(self.rotations),-1).t() #rgb indices for N rot tactile images

            # rot_ind = torch.argmax(dists_max,axis=1) #rotation for each tactile, N
            # rgb_ind = [ind_max_rot[i,rot_ind[i]] for i in range(ind_max_rot.shape[0])]

            # best_rgb_crop = [rgb_crops_color[i] for i in rgb_ind]
            # best_rotation = [self.rotations[i] for i in rot_ind]  

            # heatmaps N (patch numbers), rgb_crop: N C W H, rotation: N
            heatmaps, best_rgb_crop, best_rotation = heatmap_cal(dists)
            heatmap_list.append(heatmaps)
            rgb_crop_list.append(best_rgb_crop)
            rotation_list.append(best_rotation)

        heatmap_list = np.array(heatmap_list)
        heatmap_list = heatmap_list.transpose(1, 0, 2, 3)
        # heatmap output is len(tac_ims) x len(scales) x height x width
        crops = np.array(rgb_crop_list).transpose(1, 0, 2, 3, 4)
        rotations = np.array(rotation_list).transpose(1, 0)

        return heatmap_list, crops, rotations

    def visualize_heatmaps(self, load_dir):
        in_img, rgb_list, tac_list = self.load_data(load_dir)

        def vis(heatmaps, best_crops, rotations, name):
            for i in range(len(tac_list)):  # for each tac
                fig, axs = plt.subplots(len(self.crop_ratios) + 1, 3)
                fig.set_size_inches(16, 4)
                axs[0, 0].imshow(rgb_list[i])
                axs[0, 1].imshow(np.rot90(tac_list[i], k=1))
                axs[0, 2].imshow(in_img)

                for j in range(len(self.crop_ratios)):
                    axs[j + 1, 0].imshow(best_crops[i, j].transpose(1, 2, 0))
                    # we put the +90 to rotations since the "zero rotation" image is actually rotated once already
                    axs[j + 1, 1].imshow(
                        ndimage.rotate(tac_list[i], rotations[i, j] + 90, reshape=False))
                    axs[j + 1, 2].imshow(heatmaps[i, j], cmap='inferno')

                plt.savefig(name + f'_{i}.jpg', dpi=300)
                plt.show()

                img = rgb_list[i]
                hmap = (heatmaps[i, j] * 256).astype(np.uint8)
                # breakpoint()
                hmap_img = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
                # breakpoint()
                hmap_img = cv2.resize(hmap_img, img.shape[:2])
                breakpoint()
                superimposed = cv2.addWeighted(img, 0.5, hmap_img, 0.5, 0)
                plt.imshow(superimposed)
                plt.show()
                # breakpoint()

                plt.figure()
                plt.imshow(heatmaps[i, j], cmap='inferno')
                plt.savefig(name+f'_hmap.jpg', dpi=300)
                plt.show()


        heatmap_list_all, crops_all, rotations_all = self.compute_similarity_heatmap(in_img,
                                                                                     tac_list)
        model_results_dir = f"{load_dir}/{self.params['encoder']['model_yaml'][-19:]}"
        os.makedirs(model_results_dir, exist_ok=True)
        vis(heatmap_list_all, crops_all, rotations_all,
            f"{model_results_dir}/results_all_{load_dir[-3:]}")

    def query_network_tac(self, tac, tac_enc=None, do_aug=False, do_preproc=True):
        if not isinstance(tac, list):
            tac = [tac]
        tac_batches = self.preprocess_tac(tac, do_aug=do_aug, do_preproc=do_preproc)
        with torch.no_grad():
            if tac_enc is None:
                tac_embedding = self.model.tac_enc(tac_batches).cpu().numpy()
            else:
                tac_embedding = tac_enc(tac_batches).cpu().numpy()
        return tac_embedding

    def query_network_rgb(self, rgb, im_scale, rgb_size, grayscale=True, rgb_enc=None, preprocess=True):
        if preprocess:
            if not isinstance(rgb, list):
                rgb = [rgb]
            rgb_batches = self.preprocess_rgb(rgb, im_scale, rgb_size , grayscale)
        else:
            rgb_batches = rgb
        with torch.no_grad():
            if rgb_enc is None:
                rgb_embedding = self.model.rgb_enc(rgb_batches).cpu().numpy()
            else:
                rgb_embedding = rgb_enc(rgb_batches).cpu().numpy()
        return rgb_embedding

    def query_rotation_network(self, rgb_im, im_scale, rgb_size, tac_im, grayscale=True):
        # given hxwxc input numpy arrays, output the angle from rgb image to tac image
        # TODO @raven is this the correct direction lol
        if not isinstance(rgb_im,list):
            rgb_im = [rgb_im]
        if not isinstance(tac_im,list):
            tac_im = [tac_im]
        rgb_batch = self.preprocess_rgb(rgb_im, im_scale, rgb_size, grayscale)
        tac_batch = self.preprocess_tac(tac_im)

        # plt.imshow(tac_batch[0].cpu().numpy().transpose(1,2,0))
        # plt.show()

        if self.params['continuous']:
            pred = 180 * self.rotation_net.cont_rotation(rgb_batch, tac_batch)[0, :]
            return pred.item()

        with torch.no_grad():
            rot_probs = self.rotation_net.rotation_probs(rgb_batch, tac_batch)[0, :]
        rotations = self.rotation_net.rotation_list
        best_ind = torch.argmax(rot_probs, 0).item()
        ind1 = (best_ind + 1) % len(rotations)
        ind2 = (best_ind - 1) % len(rotations)
        ang1, ang2, ang3 = rotations[ind1].item(), rotations[best_ind].item(), rotations[
            ind2].item()
        # print("rotation",rot_probs)

        # return ang2,rot_probs.cpu().numpy()
        conf1, conf2, conf3 = rot_probs[ind1], rot_probs[best_ind], rot_probs[ind2]
        if best_ind == 0:
            ang2 -= 360
        if best_ind == len(rotations) - 1:
            ang1 += 360
        # print(ang1, ang2, conf1, conf2, conf3)
        return (ang1 * conf1 + ang2 * conf2 + ang3 * conf3 / (conf1 + conf2 + conf3)).item(),rot_probs.cpu().numpy()

    def query_rotation_network_batch(self, rgb_im, im_scale, rgb_size, tac_im, grayscale=True):
        # given hxwxc input numpy arrays, output the angle from rgb image to tac image
        # TODO @raven is this the correct direction lol
        if not isinstance(rgb_im,list):
            rgb_im = [rgb_im]
        if not isinstance(tac_im,list):
            tac_im = [tac_im]
        rgb_batch = self.preprocess_rgb(rgb_im, im_scale, rgb_size, grayscale)
        tac_batch = self.preprocess_tac(tac_im)

        # plt.imshow(tac_batch[0].cpu().numpy().transpose(1,2,0))
        # plt.show()

        if self.params['continuous']:
            pred = 180 * self.rotation_net.cont_rotation(rgb_batch, tac_batch)
            return pred.item()

        with torch.no_grad():
            rot_probs = self.rotation_net.rotation_probs(rgb_batch, tac_batch)
        rotations = self.rotation_net.rotation_list
        best_ind = torch.argmax(rot_probs,-1).cpu().numpy()
        ind1 = (best_ind + 1) % len(rotations)
        ind2 = (best_ind - 1) % len(rotations)
        ang1, ang2, ang3 = rotations[ind1].cpu().numpy(), rotations[best_ind].cpu().numpy(), rotations[ind2].cpu().numpy()
        # print("rotation",rot_probs)

        # return ang2,rot_probs.cpu().numpy()
        conf1, conf2, conf3 = rot_probs[range(len(ind1)),ind1].cpu().numpy(), rot_probs[range(len(ind1)),best_ind].cpu().numpy(), rot_probs[range(len(ind1)),ind2].cpu().numpy()
        index = np.where(best_ind==0)[0]
        ang2[index] -= 360
        index = np.where(best_ind==len(rotations) - 1)[0]
        ang1[index] += 360
        # print(ang1, ang2, conf1, conf2, conf3)
        return (ang1 * conf1 + ang2 * conf2 + ang3 * conf3 / (conf1 + conf2 + conf3)),rot_probs.cpu().numpy()

    def visualize_live(self, OUTPUT_DIR, rgb_flag, save_flag, im_scale, rgb_size):
        self.capture.home()
        new_dir = osp.join(OUTPUT_DIR, f"images_set_{len(os.listdir(OUTPUT_DIR)) + 1}")
        os.makedirs(new_dir)
        self.capture.robot.start_freedrive()
        input("Press enter to take a picture")
        trans = self.capture.robot.get_pose().translation
        self.capture.robot.stop_freedrive()

        for _ in range(10):
            self.capture.d.get_frame(True)
        # use the x,y from this commanded pose
        if rgb_flag:
            camera_pose = l_p([trans[0], trans[1], self.capture.HOME_Z])
            camera_pose = camera_pose * RigidTransform(np.eye(3), -self.capture.CAM_OFFSET,
                                                       camera_pose.from_frame,
                                                       camera_pose.from_frame)
            self.capture.robot.move_pose(camera_pose, interp='tcp', vel=0.2, acc=.2)
            time.sleep(2)
            # capture the global image here
            global_rgb = self.capture.web.frames(True)
            Image.fromarray(global_rgb).save(f"{new_dir}/image_reference_rgb.jpg")
            global_rgb = np.asarray(Image.open(f"{new_dir}/image_reference_rgb.jpg"))
            reference_embedding = self.query_network_rgb([global_rgb], im_scale, rgb_size)
            im = self.preprocess_rgb([global_rgb], im_scale, rgb_size)
            cv2.imshow("Reference image", im.cpu().numpy()[0, ...].transpose(1, 2, 0))
            cv2.waitKey(10)
        else:
            digit_pose = l_p([trans[0], trans[1], self.capture.HOME_Z])
            self.capture.robot.move_pose(digit_pose, interp='tcp', vel=0.5, acc=2)
            self.capture.robot.move_until_contact([0, 0, -0.11, 0, 0, 0], 20, acc=.4)  # was 18
            global_tac = self.capture.d.frames(True)
            Image.fromarray(global_tac).save(f"{new_dir}/image_reference_tac.jpg")
            reference_embedding = self.query_network_tac([global_tac])
            cv2.imshow("Reference image", global_tac)
            cv2.waitKey(10)

        idx = 0
        dist_buffer = []
        tac_readings = []
        # self.capture.robot.start_freedrive()

        while True:
            self.capture.robot.start_freedrive()
            cmd = input("take the tac")
            if cmd != "":
                break
            self.capture.robot.stop_freedrive()
            cur_pos = self.capture.robot.get_pose().translation
            tac_pos = l_p([cur_pos[0], cur_pos[1], self.capture.HOME_Z])
            web_pos = tac_pos * RigidTransform(np.eye(3), -self.capture.CAM_OFFSET,
                                               tac_pos.from_frame, tac_pos.from_frame)
            self.capture.robot.move_pose(web_pos, interp='tcp', vel=0.1, acc=1)
            time.sleep(1.5)
            web_rgb = self.capture.web.frames(True)
            Image.fromarray(web_rgb).save(f"{new_dir}/image_rgb_{idx}.jpg")
            self.capture.robot.move_pose(tac_pos, interp='tcp', vel=0.5, acc=2)
            self.capture.robot.move_until_contact([0, 0, -0.11, 0, 0, 0], 18, acc=.4)  # was 18
            time.sleep(.5)
            tac = self.capture.d.get_frame(True)

            cur_pos = self.capture.robot.get_pose().translation
            up_pos = l_p([cur_pos[0], cur_pos[1], self.capture.HOME_Z])
            self.capture.robot.move_pose(up_pos, interp='tcp', vel=0.1, acc=1)

            tac_embedding = self.query_network_tac([tac])
            # rot_tac_embedding = self.query_network_tac(tac,self.rotation_net.tac_enc)
            dists = reference_embedding.dot(tac_embedding.T)
            # dists_rot = rot_reference_embedding.dot(rot_tac_embedding.T)
            rotation, probs = self.query_rotation_network(global_rgb, im_scale, rgb_size, tac, grayscale=True)
            plt.figure()
            plt.bar(self.rotation_net.rotation_list,probs, width=10)
            plt.savefig(f"{self.save_dir}/rotation_dist_{idx}.jpg")
            # if len(dist_buffer) > 10:
            #     dist_buffer.pop(0)
            # dist_buffer.extend([dists.item()])  
            # dist_mean = np.mean(dist_buffer)
            tac = ndimage.rotate(tac, 90)
            if save_flag:
                Image.fromarray(tac).save(f"{new_dir}/image_tac_{idx}.jpg")
            tac = ndimage.rotate(tac,-rotation)
            cv2.putText(tac, f'{dists.item():.2f},rot: {rotation:.2f}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)

            new_tac = Image.fromarray(tac)
            tac_readings.append(new_tac)
            if save_flag:
                new_tac.save(f"{new_dir}/image_newtac_{idx}.jpg")

            tac = cv2.resize(tac, (5 * tac.shape[1], 5 * tac.shape[0]))
            # cv2.imshow('Tac image',tac)
            plt.imshow(tac)
            plt.show()
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                print("breaking")
                cv2.destroyAllWindows()
                break
            idx += 1
        self.capture.robot.stop_freedrive()
        if input("save y/n") == 'y':
            tac_readings[0].save(f"{new_dir}/out.gif", save_all=True,
                                 append_images=tac_readings[1:], duration=100, loop=0)
        return new_dir

    def test_live(self, load_dir, rgb_flag):
        if load_dir is None:
            self.capture = DataCaptureUR5(self.save_dir)
            self.capture.home()
            inp = input("Continue? Y/n")
            while 'y' in inp.lower() or inp == '':
                dir = self.visualize_live(self.save_dir, rgb_flag, save_flag=True, im_scale=0.15,
                                          rgb_size=self.rgb_size)
                inp = input("Continue? Y/n")
        else:
            if rgb_flag:
                in_img = np.asarray(Image.open(f'{load_dir}/image_reference_rgb.jpg'))
                reference_embedding = self.query_network_rgb([in_img], im_scale=0.1,
                                                             rgb_size=self.rgb_size)
            else:
                in_img = np.asarray(Image.open(f'{load_dir}/image_reference_tac.jpg'))
                reference_embedding = self.query_network_tac([in_img])

            tac_list = []
            rgb_list = []
            for filename in os.listdir(load_dir):
                if 'tac' in filename and 'reference' not in filename and 'new' not in filename:
                    tac = np.asarray(Image.open(f'{load_dir}/{filename}'))
                    tac = ndimage.rotate(tac, -90)
                    tac_list.append(tac)
                    rgb_file = filename.replace('tac', 'rgb')
                    rgb = np.asarray(Image.open(f'{load_dir}/{rgb_file}'))
                    rgb_list.append(rgb)

            tac_embedding = self.query_network_tac(tac_list)
            dists = reference_embedding.dot(tac_embedding.T)[0]
            new_dir = f"{load_dir}/{self.params['encoder']['rot_model_dir'][43:-8]}"
            os.makedirs(new_dir)
            Image.fromarray(in_img).save(f"{new_dir}/image_reference_rgb.jpg")
            rotations = []
            for i, tac in enumerate(tac_list):
                rotation,probs = self.query_rotation_network(in_img, 0.15, self.rgb_size, tac)
                rotations.append(rotation)
                plt.figure()
                plt.bar(self.rotation_net.rotation_list,probs, width=10)
                plt.savefig(f"{new_dir}/rotation_dist_{i}.jpg")
                tac = ndimage.rotate(tac, 90)
                tac = ndimage.rotate(tac,-rotation)
                cv2.putText(tac, f'{dists[i]:.2f},rot: {rotation:.2f}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                Image.fromarray(tac).save(f"{new_dir}/image_newtac_{i}.jpg")
                Image.fromarray(rgb_list[i]).save(f"{new_dir}/image_rgb_{i}.jpg")

                # tac = cv2.resize(tac,(5*tac.shape[1],5*tac.shape[0]))
                # cv2.imshow('Tac image',tac)
                # k = cv2.waitKey(1) & 0xFF
                # if k == 27:
                #     print("breaking")
                #     cv2.destroyAllWindows()
                #     break

            # idxes = np.argsort(dists)
            # fig,axs=plt.subplots(1,len(tac_list)+1)
            # axs[0,0].imshow(in_img)
            # for i,idx in enumerate(idxes):
            #     axs[0,i+1].imshow(tac_list[idx])
            #     axs[0,i+1].set_title(f"score:{dists[idx]}")
            # plt.savefig(f"{load_dir}/results.png")
            # plt.imshow()

    def run_tests(self, load_dir):
        if self.params['test_type'] == 'heatmap':
            self.visualize_heatmaps(load_dir)
        elif self.params['test_type'] == 'tsne_single':
            self.tsne_from_dir([load_dir])
        elif self.params['test_type'] == 'tsne_batch':
            ds = PairedDatasetZoomedOut(self.params['dataset'])
            dl = DataLoader(ds, batch_size=self.params['dataset']['batch_size'] //
                                           self.params['dataset']['repeat_rotations'],
                            num_workers=self.params['dataset']["num_cores"], shuffle=True,
                            drop_last=False, collate_fn=collate_paired)
            for b in dl:
                self.tsne_from_batch(b)
        elif self.params['test_type'] == 'pca_single':
            self.eig_from_dirs([load_dir])
        elif self.params['test_type'] == 'pca_all':
            self.eig_from_dirs(self.load_dirs)
        elif self.params['test_type'] == 'pca_batch':
            ds = PairedDatasetZoomedOut(self.params['dataset'])
            dl = DataLoader(ds, batch_size=self.params['dataset']['batch_size'] //
                                           self.params['dataset']['repeat_rotations'],
                            num_workers=self.params['dataset']["num_cores"], shuffle=True,
                            drop_last=False, collate_fn=collate_paired)
            for b in dl:
                self.eig_from_batch(b)
        elif self.params['test_type'] == 'transfer':
            self.transfer(load_dir)
        else:
            print("Unexpected input test_type")

    def test(self):
        if self.params['test_type'] == 'tsne_all':
            self.tsne_from_dir(self.load_dirs)
            return
        if self.params["load"]:
            for dir in self.load_dirs:
                self.run_tests(dir)
        else:
            self.capture = DataCaptureUR5(self.save_dir)
            self.capture.home()
            inp = input("Continue? Y/n")
            while 'y' in inp.lower() or inp == '':
                dir = self.capture.manual_collect()
                # dir = self.capture.test_collect(num = 2,x_size=.3,y_size=.15,fixed=True,rotate=False,large_capture=False)
                self.run_tests(dir)
                inp = input("Continue? Y/n")
