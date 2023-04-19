from tacvis.tester import Tester
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from tacvis.dataset import TAC_AUGMENTS, PREPROC_IMG
import torchvision.transforms.functional as TF
import random
import torch
from pytouch.tasks import ContactArea

import cProfile, pstats, io
from pstats import SortKey

def prof_dec(f):
    def new_f(*args,**kwargs):
        pr = cProfile.Profile()
        pr.enable()
        x = f(*args,**kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return x
    return new_f


'''
defines a stop condition to be used in sliding
'''
def rot_aug(tac_img, n, rot_angles=None, do_aug=False):
    tac = PREPROC_IMG(tac_img)
    if do_aug:
        # apply color jitters before spatial aug, we want to keep the same colors for each pair
        tac = TAC_AUGMENTS(tac)
    hpad = int(np.clip(max(tac.shape[1], tac.shape[2]) - tac.shape[1], 0, np.inf) / 2)
    wpad = int(np.clip(max(tac.shape[1], tac.shape[2]) - tac.shape[2], 0, np.inf) / 2)
    tac = TF.pad(tac, [wpad, hpad])
    tac = TF.rotate(tac, 90)
    tacs = []
    for i in range(n):
        tac2 = tac.clone()
        if rot_angles is None:
            r_amnt = random.uniform(-180, 180)
        else:
            r_amnt = rot_angles[i]
        tac2 = TF.rotate(tac2, r_amnt)
        tacs.append(tac2)
        # after rotating/flipping the images, we need to center-crop the rgb image
    return tacs


class StopCond:
    def __init__(self, tester: Tester, visualize=True):
        self.tester = tester
        self.visualize = visualize
        self.update = True
        self.sim_diff_rotation = None


    def should_stop(self):
        return True
    
    def rot_aug(self, tac_img, n, rot_angles=None, do_aug=False):
        tac = PREPROC_IMG(tac_img)
        if do_aug:
            # apply color jitters before spatial aug, we want to keep the same colors for each pair
            tac = TAC_AUGMENTS(tac)
        hpad = int(np.clip(max(tac.shape[1], tac.shape[2]) - tac.shape[1], 0, np.inf) / 2)
        wpad = int(np.clip(max(tac.shape[1], tac.shape[2]) - tac.shape[2], 0, np.inf) / 2)
        tac = TF.pad(tac, [wpad, hpad])
        tac = TF.rotate(tac, 90)
        tacs = []
        for i in range(n):
            tac2 = tac.clone()
            if rot_angles is None:
                r_amnt = random.uniform(-180, 180)
            else:
                r_amnt = rot_angles[i]
            tac2 = TF.rotate(tac2, r_amnt)
            tacs.append(tac2)
            # after rotating/flipping the images, we need to center-crop the rgb image
        return tacs

class ImageDiffStopCond(StopCond):
    def __init__(self,tester,window_len,buff_len,threshold=.2):
        super().__init__(tester)
        self.window_len = window_len
        self.window = []
        self.threshold=threshold

        self.update = True
        self.tac_buffer = []
        self.buff_len = buff_len
        self.previous_sim = []
        self.delay = 20
        assert self.delay<window_len, "window len must be larger than the delay"
    
    def should_stop(self, tac_img: np.ndarray, rot_aug=False):
        if self.update:
            sim_diff = self.pre_update(tac_img)
            if sim_diff > self.threshold:
                print("stop updating")
                self.update = False
            return False, sim_diff
        else:
            buffer_full, sim_diff = self.post_update(tac_img)
            if buffer_full:
                if sim_diff > self.threshold:
                    print("reach thres")
                    return True, sim_diff
                else:
                    print("resume update")
                    self.update = True
                    self.window = []
                    self.tac_buffer = []
                    return False, sim_diff
            else:
                return False, sim_diff

    def pre_update(self,tac_img):
        if len(self.window)<self.window_len:
            self.window.append(tac_img)
            return 0
        diffs = []
        for i in self.window:
            diff = np.sum(np.abs(i - tac_img))/(255.*np.product(i.shape))
            diffs.append(diff)
        med = np.median(diffs)
        self.window.pop(0)
        self.window.append(tac_img)
        return med
    
    def post_update(self, tac_img):
        if len(self.tac_buffer) < self.buff_len:
            buffer_full = False
            self.tac_buffer.append(tac_img)
        else:
            buffer_full = True
            self.tac_buffer.pop(0)
            self.tac_buffer.append(tac_img)

        diffs = []
        for i in self.window:
            diff = np.sum(np.abs(i - np.median(self.tac_buffer)))/(255.*np.product(i.shape))
            diffs.append(diff)
        med = np.median(diffs)
        return buffer_full,med



class SimilarityStopCond(StopCond):
    def __init__(self, tester: Tester, ref_embeddings, window_len = 10, threshold=.3):
        super().__init__(tester)
        assert len(ref_embeddings.shape) == 2, "shape of ref embedding needs to be B x embed_dim"
        self.ref_embeds = ref_embeddings
        self.threshold = threshold
        self.window_len = window_len
        self.window = []
        self.tac_buffer = []
        self.buff_len = window_len
        self.prev_val = None
        self.update = True
        self.sim_diff_rotation = 0

    def should_stop(self, tac_img: np.ndarray, do_aug=False):
        if self.update:
            sim_diff = self.pre_update(tac_img, do_aug)
            if sim_diff > self.threshold:
                print("stop updating")
                self.update = False
            return False, sim_diff
        else:
            buffer_full, sim_diff = self.post_update(tac_img)
            if buffer_full:
                if sim_diff > self.threshold:
                    print("reach thres")
                    return True, sim_diff
                else:
                    print("resume update")
                    self.update = True
                    self.window = []
                    self.tac_buffer = []
                    return False, sim_diff
            else:
                return False, sim_diff


    def pre_update(self, tac_img: np.ndarray,do_aug):
        if do_aug:
            # tac_img = [np.array(np.rot90(tac_img, 2*i)) for i in range(2)]
            tac_img = self.rot_aug(tac_img, 10, rot_angles=None, do_aug=False)
        # print(tac_img)
        # print(type(tac_img))
        # print(tac_img.shape)
        tac_embed = self.tester.query_network_tac(tac_img, do_aug=False, do_preproc=not do_aug)
        # self.sim_diff_rotation = np.linalg.norm(tac_embed[0]-tac_embed[1])
        tac_embed_means = np.mean(tac_embed,1)
        self.sim_diff_rotation = np.std(tac_embed_means)


        # dists = self.ref_embeds.dot(tac_embed.T)  # should be num_embeddings x 1
        # print(dists,np.median(dists))
        # return np.median(dists)
        if len(self.window) > self.window_len:
            self.window.pop(0)
        elif len(self.window)==0:
            self.window.append(np.array(tac_embed.mean(0)[None,...]))
            return 0

        self.window.append(np.array(tac_embed.mean(0)[None,...]))
        if len(self.window)>1:
            tac_embeds = np.concatenate(self.window)
        else:
            tac_embeds = self.window[0]

        dists = self.ref_embeds.dot(tac_embeds.T)  # should be num_embeddings x 1


        return np.median(dists)
    

    def post_update(self, tac_img):
        tac_embed = self.tester.query_network_tac(tac_img)

        if len(self.tac_buffer) < self.buff_len:
            buffer_full = False
            self.tac_buffer.append(tac_embed)
        else:
            buffer_full = True
            self.tac_buffer.pop(0)
            self.tac_buffer.append(tac_embed)

        tac_embeds = np.concatenate(self.tac_buffer)
        dists = self.ref_embeds.dot(tac_embeds.T).flatten()

        return buffer_full,np.median(dists)



class ImpulseStopCond(StopCond):
    def __init__(self, tester: Tester, window_len, threshold=.3):
        super().__init__(tester)
        self.window_len = window_len
        self.window = []
        self.threshold = threshold
        self.prev_val = None
        self.last = time.time()

    # @prof_dec
    def should_stop(self, tac_img: np.ndarray):
        tac_embed = self.tester.query_network_tac(tac_img)
        if len(self.window) < self.window_len:
            self.window.append(tac_embed)
            return False, 0
        ref_embeds = np.concatenate(self.window)
        self.window.pop(0)
        self.window.append(tac_embed)
        similarity = ref_embeds.dot(tac_embed.T)  # should be num_embeddings x 1
        # if self.visualize:
        #     self._visualize_decision(tac_img, similarity)
        if self.prev_val is None:
            self.prev_val = similarity.mean()
        sim_diff = abs(similarity.mean() - self.prev_val)
        if sim_diff > self.threshold:
            print("Prev val", self.prev_val)
            print("Stopped: ", similarity)
            return True, sim_diff
        self.prev_val = similarity.mean()
        # print("sim_diff",sim_diff)
        self.last = time.time()
        return False, sim_diff

    def _visualize_decision(self, tac_im, dists):
        if not self.visualize: return
        dist_mean = dists.mean()
        # plot the image, and the corresponding similarity
        cv2.putText(tac_im, f'{dist_mean:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        tac_im = cv2.resize(tac_im, (5 * tac_im.shape[1], 5 * tac_im.shape[0]))
        cv2.imshow(f'Tac image', tac_im)
        cv2.waitKey(10)


class ImpulseStopCondActive(ImpulseStopCond):
    def __init__(self, tester: Tester, window_len, buff_len, threshold=.3):
        super().__init__(tester, window_len, threshold)
        self.update = True
        self.tac_buffer = []
        self.buff_len = buff_len
        self.previous_sim = []
        self.delay = 20
        self.n_aug = 10
        self.sim_diff_rotation = 0
        assert self.delay<window_len, "window len must be larger than the delay"

    def should_stop(self, tac_img: np.ndarray, do_aug=False):
        if self.update:
            sim_diff = self.pre_update(tac_img, do_aug=do_aug)
            if sim_diff < self.threshold:
                print("stop updating")
                self.update = False
            return False, sim_diff
        else:
            buffer_full, sim_diff = self.post_update(tac_img)
            if buffer_full:
                if sim_diff < self.threshold:
                    print("reach thres")
                    return True, sim_diff
                else:
                    print("resume update")
                    self.update = True
                    self.window = []
                    self.tac_buffer = []
                    return False, sim_diff
            else:
                return False, sim_diff

    def pre_update(self, tac_img: np.ndarray, do_aug=False):
        if do_aug:
            # tac_img = [np.array(np.rot90(tac_img, 2*i)) for i in range(2)]
            tac_img = self.rot_aug(tac_img, self.n_aug, rot_angles=None, do_aug=False)
        tac_embed = self.tester.query_network_tac(tac_img, do_aug=False, do_preproc=not do_aug)
        
        if len(self.window) > self.window_len:
            self.window.pop(0)
        elif len(self.window)==0:
            self.window.append(np.array(tac_embed.mean(0)[None,...]))
            return 1
        if len(self.window)>1:
            ref_embeds = np.concatenate(self.window)
        else:
            ref_embeds = self.window[0]
        similarity = ref_embeds[:self.delay].dot(tac_embed.T)  # should be num_embeddings x 1

        self.sim_diff_rotation = np.linalg.norm(similarity[:,0]-similarity[:,1])
        print("sim difference between rotations: ",self.sim_diff_rotation)
        # if self.visualize:
        #     self._visualize_decision(tac_img, similarity)
        if self.prev_val is None:
            self.prev_val = np.median(similarity)

        sim_diff = abs(np.median(similarity) - self.prev_val)

        self.prev_val = np.median(similarity)
        self.previous_sim = similarity
        self.last = time.time()
        self.window.append(np.array(tac_embed.mean(0)[None,...]))
        # return sim_diff
        return self.prev_val

    def post_update(self, tac_img):
        tac_embed = self.tester.query_network_tac(tac_img)
        if len(self.tac_buffer) < self.buff_len:
            buffer_full = False
            self.tac_buffer.append(tac_embed)
        else:
            buffer_full = True
            self.tac_buffer.pop(0)
            self.tac_buffer.append(tac_embed)

        if len(self.window)>1:
            ref_embeds = np.concatenate(self.window)
        else:
            ref_embeds = self.window[0]
        tac_embeds = np.concatenate(self.tac_buffer)
        similarity = ref_embeds[:self.delay].dot(tac_embeds.T).flatten()
        # print("cur_similarity", np.median(similarity))
        # print("pre_val", self.prev_val)
        sim_diff = abs(np.median(similarity) - self.prev_val)
        # sim_diff = abs(np.median(similarity) - np.median(self.previous_sim))
        # sim_diff = np.median(abs(similarity - self.prev_val))

        # return buffer_full, sim_diff
        return buffer_full,np.median(similarity)


class EdgeStopCond(StopCond):
    """
    Source: https://github.com/facebookresearch/PyTouch/blob/main/pytouch/tasks/contact_area.py
    """
    def __init__(self, base, visualize=False):
        super().__init__(None, visualize)
        self.contact_area = ContactArea(base)

    def should_stop(self, tac_img):
        major, minor = self.contact_area(tac_img)

        return 0
