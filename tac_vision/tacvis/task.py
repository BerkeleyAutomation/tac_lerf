from tacvis.stop_condition import SimilarityStopCond, StopCond, ImpulseStopCond, \
    ImpulseStopCondActive,rot_aug,ImageDiffStopCond
from tacvis.tester import Tester
import numpy as np
import cv2
import os
import os.path as osp
from autolab_core import RigidTransform
import time
from PIL import Image
from tacvis.capture import DataCaptureUR5, l_p
import yaml
from digit_interface import Digit
from tacvis.lightning_modules import ContrastiveModule
import matplotlib.pyplot as plt
import json 
from sklearn.manifold import TSNE
import torchvision
import random
from scipy import ndimage

import cProfile, pstats, io
from pstats import SortKey

# from tacvis.contact_area import ContactArea
from pytouch.tasks import ContactArea

import pdb

def prof_dec(f):
    def new_f(*args,**kwargs):
        pr = cProfile.Profile()
        pr.enable()
        f(*args,**kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
    return new_f



class JEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JEncoder, self).default(obj)

class ManualStopTest():
    def __init__(self, params, save_dir, imagenet=False) -> None:
        self.tester = Tester(params)
        if imagenet:
            with open(params["encoder"]["model_yaml"], 'r') as stream:
                model_params = yaml.safe_load(stream)
            self.tester.model = ContrastiveModule(model_params).eval().cuda()

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir + '/images_rgb', exist_ok=True)
        self.d = Digit("D20161")  # Unique serial number
        self.d.connect()

    def slide_impulse(self):
        while True:
            if 'n' not in input('continue'):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                hl, = ax.plot([], [])
                index = 0
                tac_readings = []
                # self.stopper = ImpulseStopCond(self.tester,5,0.3)
                self.stopper = ImpulseStopCondActive(self.tester, 1, 0.3)
                self.stopper.buff_len = 20
                while True:
                    index += 1
                    tac_reading = self.d.get_frame(True)
                    stop, dist = self.stopper.should_stop(tac_reading)

                    hl.set_xdata(np.append(hl.get_xdata(), index))
                    hl.set_ydata(np.append(hl.get_ydata(), dist))
                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.01)

                    tac = tac_reading.copy()
                    cv2.putText(tac, f'dist: {dist:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                255)
                    new_tac = Image.fromarray(tac)
                    tac_readings.append(new_tac)
                    cv2.imshow('tac image', tac)
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27:
                        print("breaking")
                        cv2.destroyAllWindows()
                        break

                    if stop:
                        print("break due to the stopper")
                        Image.fromarray(tac).save(f"{self.save_dir}/image_stop_tac.jpg")
                        cv2.destroyAllWindows()
                        break
                        # fp = open(f"{self.save_dir}/out.gif", "wb")
                tac_readings[0].save("out.gif", save_all=True, append_images=tac_readings[1:],
                                     duration=100, loop=0)
                # fp.close()
                plt.close('all')
            else:
                break


class slidePrimitive():
    def __init__(self, params, save_dir, debug=False) -> None:
        self.tester = Tester(params)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir + '/images_rgb', exist_ok=True)
        self.capture = DataCaptureUR5(self.save_dir)
        self.debug = debug
        self.const_z = None
    
    def vis_tsne(self,load_dir):
        tac_list = []
        for filename in os.listdir(load_dir):
            if 'tac' in filename and ".jpg" in filename and "new" not in filename and "stop" not in filename:
                tac = np.asarray(Image.open(f'{load_dir}/{filename}'))
                tac_list.append(tac)
        tacs = np.stack(tac_list,axis=0).transpose(0,3,1,2)
        tac_embedding = self.tester.query_network_tac(tac_list)
        tsne = TSNE(metric='cosine', init='pca')
        vis_vectors = tsne.fit(tac_embedding).embedding_
        fig, axs = plt.subplots(1)
        fig.set_size_inches(16, 4)
        self.tester.scatter_images(vis_vectors, tacs, axs, fig, tac_embedding)
        plt.show()

    def take_reference_image(self, rgb_flag=True):
        new_dir = osp.join(self.save_dir, f"images_set_{len(os.listdir(self.save_dir)) + 1}")
        os.makedirs(new_dir)
        self.capture.robot.start_freedrive()
        input("Press enter to take a picture")
        trans = self.capture.robot.get_pose().translation
        self.capture.robot.stop_freedrive()
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
            Image.fromarray(global_rgb).save(f"{new_dir}/image_reference_rgb.png")
            return global_rgb
        else:
            digit_pose = l_p([trans[0], trans[1], self.capture.HOME_Z])
            self.capture.robot.move_pose(digit_pose, interp='tcp', vel=0.5, acc=2)
            self.capture.robot.move_until_contact([0, 0, -0.11, 0, 0, 0], 20, acc=.4)  # was 18
            global_tac = self.capture.d.frames(True)
            Image.fromarray(global_tac).save(f"{new_dir}/image_reference_tac.png")
            return global_tac

    def move_linear(self, start, end, vel, stopper: StopCond, thres=20, acc=0.25):
        assert len(start) == 3 and len(end) == 3, "dimension of start and goal must be 3"
        assert len(vel) == 3, "vel is for x,y,z direction and the first nonzero one would be used"

        if abs(end[0] - start[0]) > abs(end[1] - start[1]):
            print("generataing based on x")
            direc_x = vel[0] * np.sign((end[0] - start[0] + 1e-6))
            direc_y = np.clip((end[1] - start[1]) / (end[0] - start[0] + 1e-6), -10, 10) * direc_x
        elif abs(end[1] - start[1]) > abs(end[0] - start[0]):
            print("generataing based on y")
            direc_y = vel[1] * np.sign((end[1] - start[1] + 1e-6))
            direc_x = np.clip((end[0] - start[0]) / (end[1] - start[1] + 1e-6), -10, 10) * direc_y

        dire = [direc_x, direc_y, 0, 0, 0, 0]
        print("dire", dire)
        start_pose = l_p([start[0], start[1], -0.15])
        self.capture.robot.move_pose(start_pose, vel=0.2, acc=2)
        if self.const_z is None:
            self.capture.robot.move_until_contact([0, 0, -0.08, 0, 0, 0], 16, acc=.1)
        else:
            start_pose.translation[2] = self.const_z
            self.capture.robot.move_pose(start_pose, vel=0.2, acc=2)

        tac_reading = self.capture.d.get_frame(True)
        tac = tac_reading.copy()
        tac = cv2.resize(tac, (5 * tac.shape[1], 5 * tac.shape[0]))
        cv2.imshow('tac image', tac)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print("breaking")
            cv2.destroyAllWindows()

        if input("start") != '':
            return
        self.capture.robot.ur_c.speedL(dire, acceleration=acc)
        time.sleep(.5)
        startforce = np.array(self.capture.robot.ur_r.getActualTCPForce())

        if self.debug:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        x_data = []
        y_data =[]
        dif_data = []
        # hl, = ax.plot([], [])
        index = 0
        tac_readings = []
        new_tac_readings = []
        tac_embeddings = {}

        stop_flag = False

        # TODO remove
        # pr = cProfile.Profile()
        # pr.enable()

        while True:
            index += 1
            force = np.array(self.capture.robot.ur_r.getActualTCPForce())
            cur_force = np.linalg.norm((startforce - force).dot(dire))
            tac_reading = self.capture.d.get_frame(True)
            tac_readings.append(Image.fromarray(tac_reading))
            
            tac_embeddings[index] = self.tester.query_network_tac(tac_reading)
            stop, dist = stopper.should_stop(tac_reading, True)

            if not stopper.update:
                self.capture.robot.ur_c.speedL(np.array(dire/np.linalg.norm(dire))*0.0002, acceleration=acc)
                # self.capture.robot.ur_c.speedStop()
                if not stop_flag:
                    time.sleep(2)
                stop_flag = True
            elif stop_flag:
                self.capture.robot.ur_c.speedL(dire, acceleration=acc)
                stop_flag = False

            # hl.set_xdata(np.append(hl.get_xdata(), index))
            # hl.set_ydata(np.append(hl.get_ydata(), dist))
            x_data.append(index)
            y_data.append(dist)
            if stopper.sim_diff_rotation is not None:
                dif_data.append(stopper.sim_diff_rotation)
                if self.debug:
                    ax.plot(x_data,dif_data,'r')
            if self.debug:
                ax.plot(x_data,y_data,'b')
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.01)

            tac = tac_reading.copy()
            cv2.putText(tac, f'force,dist: {cur_force:.2f},{dist:.2f}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
            new_tac = Image.fromarray(tac)
            new_tac_readings.append(new_tac)
            # new_tac.save(f"{self.save_dir}/image_new_tac_{index}.jpg")
            tac = cv2.resize(tac, (5 * tac.shape[1], 5 * tac.shape[0]))
            cv2.imshow('tac image', tac)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                print("breaking")
                cv2.destroyAllWindows()
                break

            if stop:
                print("break due to the stopper")
                Image.fromarray(tac).save(f"{self.save_dir}/image_stop_tac.jpg")
                tac = tac_reading.copy()
                cv2.putText(tac, f'force,dist,stop: {cur_force:.2f},{dist:.2f}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
                new_tac = Image.fromarray(tac)
                new_tac.save(f"{self.save_dir}/image_new_tac_stop{index}.jpg")
                tac_readings.append(new_tac)
                break

            if cur_force > thres:
                print("break due to exceeding force limit")
                cv2.destroyAllWindows()
                break
            if np.linalg.norm(self.capture.robot.get_pose().translation[:2] - end[:2]) < 0.01:
                print("break due to reaching the end")
                # cv2.destroyAllWindows()
                break

            time.sleep(0.008)

        # TODO remove
        # pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

        self.capture.robot.ur_c.speedStop()
        time.sleep(0.5)
        cur_pos = self.capture.robot.get_pose().translation
        up_pos = l_p([cur_pos[0], cur_pos[1], self.capture.HOME_Z])
        self.capture.robot.move_pose(up_pos, interp='tcp', vel=0.1, acc=1)
        new_tac_readings[0].save(f"{self.save_dir}/out.gif", save_all=True,
                             append_images=new_tac_readings[1:], duration=100, loop=0)
        os.mkdir(os.path.join(self.save_dir, 'image_tac'))
        os.mkdir(os.path.join(self.save_dir, 'image_new_tac'))
        for index, (tac_im, new_tac_im) in enumerate(zip(tac_readings, new_tac_readings)):
            tac_im.save(f"{self.save_dir}/image_tac/{index}.jpg")
            new_tac_im.save(f"{self.save_dir}/image_new_tac/{index}.jpg")
        if not self.debug:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x_data,y_data,'b')
            # ax.plot(x_data,dif_data,'r')
        plt.savefig(f"{self.save_dir}/out.jpg")
        plt.close('all')

        np.save(f"{self.save_dir}/dist.npy",y_data)

        with open(f"{self.save_dir}/tac_embeddings.json", "w") as outfile:
            json.dump(tac_embeddings, outfile,sort_keys=True, indent=2, cls=JEncoder) 


class GuidedSearch(slidePrimitive):
    def __init__(self, params, save_dir) -> None:
        super().__init__(params, save_dir)
        self.cache_embed = np.load("/home/ravenhuang/tac_vis/tac_vision/tac_embeddings.npy")
        self.index_list = np.load("/home/ravenhuang/tac_vis/tac_vision/index_list.npy")
    
    def translate_aug(self, image):
        if image.shape[-1] == 3:
            height, width = image.shape[:2]
        else:
            height, width = image.shape[-2:]
  
        quarter_height, quarter_width =random.uniform(-height/40,height / 40) , random.uniform(-width/20,width / 20)
        
        T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
        
        # We use warpAffine to transform
        # the image using the matrix, T
        img_translation = cv2.warpAffine(image, T, (width, height))
        return img_translation
    
    def cache_embeddings(self, load_dir = "/home/ravenhuang/tac_vis/tac_vision/data/ur_data/" ):
        tac_dir = f"{load_dir}images_tac"
        file_list = os.listdir(tac_dir)

        tac_list = []     
        index_list = []
        for filename in file_list:
            if "tac" in filename:
                tac = np.asarray(Image.open(f'{tac_dir}/{filename}'))
                tac_list.append(tac)
                index_list.append(filename)

        tac_embedding = []
        for i in np.arange(start=0,stop=len(tac_list),step=64):
            tac_embedding.append(self.tester.query_network_tac(tac_list[i:np.min([i+64,len(tac_list)])]))

        tac_embedding = np.concatenate(tac_embedding,axis=0)
        np.save("./tac_embeddings.npy",tac_embedding)
        np.save("./index_list.npy",index_list)

    def vis_statistic(self, cached_score):
        tac_dir = "/home/ravenhuang/tac_vis/tac_vision/data/ur_data/images_tac"
        rgb_dir = "/home/ravenhuang/tac_vis/tac_vision/data/ur_data/images_rgb"

        cached_score_partition = np.argsort(cached_score[0])
        topk = 5
        ls_tac_files = self.index_list[cached_score_partition[:topk]]
        ms_tac_files = self.index_list[cached_score_partition[-topk:]]
        ls_rgb_files =[l.replace('tac', 'rgb') for l in ls_tac_files]
        ms_rgb_files =[m.replace('tac', 'rgb') for m in ms_tac_files]

        fig,axs = plt.subplots(4,topk)
        for i in range(topk):
            ms_tac_dir = f"{tac_dir}/{ms_tac_files[i]}"
            ls_tac_dir = f"{tac_dir}/{ls_tac_files[i]}"
            ms_rgb_dir = f"{rgb_dir}/{ms_rgb_files[i]}"
            ls_rgb_dir = f"{rgb_dir}/{ls_rgb_files[i]}"
            ms_tac = np.asarray(Image.open(ms_tac_dir))
            ms_rgb = np.asarray(Image.open(ms_rgb_dir))
            ls_tac = np.asarray(Image.open(ls_tac_dir))
            ls_rgb = np.asarray(Image.open(ls_rgb_dir))
            axs[0,i].imshow(ms_tac)
            axs[0,i].text(30, 30, cached_score[0][cached_score_partition[-topk+i]], bbox=dict(fill=False, edgecolor='red', linewidth=2))
            axs[1,i].imshow(ms_rgb)
            axs[2,i].imshow(ls_tac)
            axs[2,i].text(30, 30, cached_score[0][cached_score_partition[i]], bbox=dict(fill=False, edgecolor='red', linewidth=2))
            axs[3,i].imshow(ls_rgb)
        plt.savefig(f"{self.save_dir}/cache.jpg")
        plt.show()

        plt.hist(cached_score[0],density=True)  # density=False would make counts
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.savefig(f"{self.save_dir}/hist.jpg")
        plt.show()

    def slide_till_find(self, reference_file=None, rgb_flag=True, thres=0.7):
        # self.capture.home()
        for _ in range(10):
            self.capture.d.get_frame(True)

        if reference_file is None:
            reference_img = self.take_reference_image(rgb_flag)
        else:
            reference_img = np.asarray(Image.open(reference_file))
        
        reference_imgs = []
        reference_imgs.append(reference_img)
        rgb_batches = self.tester.preprocess_rgb(reference_imgs, 0.15, [128, 128])

        # for _ in range(5):
        #     reference_imgs.append(self.translate_aug(reference_img))

       
        # plt.imshow(reference_img)
        plt.imshow(rgb_batches[0].cpu().numpy().transpose(1,2,0))
        plt.savefig(f"{self.save_dir}/reference_img.jpg")
        plt.show()

        if rgb_flag:
            ref_embedding = self.tester.query_network_rgb(reference_imgs, 0.15, [128, 128])
        else:
            ref_embedding = self.tester.query_network_tac(reference_img)

        if thres<0: #use the adaptive trheshold
            cached_score = ref_embedding.dot(self.cache_embed.T)
            cached_score_partition = np.sort(cached_score)
            thres = np.max(cached_score_partition[:,-400:])-.2

            self.vis_statistic(cached_score)

        print("thres",thres)
        stopper = SimilarityStopCond(self.tester, ref_embedding, 2, thres)
        stopper.buff_len = 10

        self.capture.robot.start_freedrive()
        cmd = input("enter to loc the start point")
        if cmd != "":
            return 
        start_point = self.capture.robot.get_pose().translation
        input("enter to loc the end point")
        end_point = self.capture.robot.get_pose().translation
        self.capture.robot.stop_freedrive()
        self.move_linear(start_point, end_point, [0.005, 0.005, 0.005], stopper)

    def manual_slide(self, thres):
        self.capture.robot.stop_freedrive()
        self.capture.home()
        reference_img = self.take_reference_image(True)
        ref_embedding = self.tester.query_network_rgb(reference_img, 0.15, [128, 128])
        cv2.imshow("Reference image", reference_img)
        cv2.waitKey(1)
        stopper = SimilarityStopCond(self.tester, ref_embedding, thres)
        self.capture.robot.start_freedrive()
        while True:
            tac_reading = self.capture.d.get_frame(True)
            stop, dist = stopper.should_stop(tac_reading)

            tac = tac_reading.copy()
            cv2.putText(tac, f'dist: {dist:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
            tac = cv2.resize(tac, (5 * tac.shape[1], 5 * tac.shape[0]))
            cv2.imshow('tac image', tac)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                print("breaking")
                cv2.destroyAllWindows()
                break

            if stop:
                print("break due to the stopper")
                Image.fromarray(tac).save(f"{self.save_dir}/image_stop_tac.jpg")
                cv2.destroyAllWindows()
                break


class ImpulseSearch(slidePrimitive):
    def __init__(self, params, save_dir, pixel_diff=False, debug=False) -> None:
        super().__init__(params, save_dir, debug=debug)
        self.pixel_diff = pixel_diff
        # self.do_aug = params['do_aug']
    # @prof_dec
    def slide_till_find(self, thres=0.7, window=20, buff=20):
        # self.capture.home()
        for _ in range(10):
            self.capture.d.get_frame(True)

        # stopper = ImpulseStopCond(self.tester,window_len = 10,threshold = thres)
        if self.pixel_diff:
            stopper = ImageDiffStopCond(self.tester, window_len=window, buff_len=20, threshold=thres)
        else:
            stopper = ImpulseStopCondActive(self.tester, window_len=window, buff_len=20, threshold=thres)
        # stopper.buff_len = window

        self.capture.robot.start_freedrive()
        input("enter to loc the start point")
        start_point = self.capture.robot.get_pose().translation
        input("enter to loc the end point")
        end_point = self.capture.robot.get_pose().translation
        self.capture.robot.stop_freedrive()
        self.move_linear(start_point, end_point, [0.01, 0.01, 0.01], stopper)

    def manual_slide(self, thres):
        self.capture.robot.stop_freedrive()
        self.capture.home()

        stopper = ImpulseStopCond(self.tester, window_len=20, threshold=thres)
        self.capture.robot.start_freedrive()
        input("start")

        while True:
            tac_reading = self.capture.d.get_frame(True)
            stop, dist = stopper.should_stop(tac_reading, do_aug=True)

            tac = tac_reading.copy()
            cv2.putText(tac, f'dist: {dist:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
            tac = cv2.resize(tac, (5 * tac.shape[1], 5 * tac.shape[0]))
            cv2.imshow('tac image', tac)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                print("breaking")
                cv2.destroyAllWindows()
                break

            if stop:
                print("break due to the stopper")
                Image.fromarray(tac).save(f"{self.save_dir}/image_stop_tac.jpg")
                cv2.destroyAllWindows()
                break


class RotationSearch(slidePrimitive):
    def __init__(self, params, save_dir, edge_baseline =False) -> None:
        super().__init__(params, save_dir)
        # self.do_aug = params['do_aug']
        self.edge_baseline = edge_baseline
        if self.edge_baseline:
            for _ in range(10):
                self.edge_base = self.capture.d.get_frame(True)

    def cache_rgb_embeddings(self,reference_img):
        h,w =  reference_img.shape[:2]
        offset = np.min([w,h])
        start = 0
        end = offset/2
        num = 10

        xtranses = []

        xtrans = np.linspace(-start,-end,num) # this is 0,1 direction in image 
        xtranses.append(xtrans)

        xtrans = np.linspace(start,end,num) # this is 0,-1 direction in image
        xtranses.append(xtrans)


        transed_imgs = []
        for xtra in xtranses:
            for tx in xtra: 
                T = np.float32([[1, 0, tx], [0, 1, 0]])
                transed_img = cv2.warpAffine(reference_img, T, (w, h))
                transed_imgs.append(transed_img)
        return transed_imgs

    def tac_translate(self,tac,rotation,rgb_embeddings):
        tac = rot_aug(np.array(tac),1,[-rotation])[0]
        tac_embedding = self.tester.query_network_tac(tac, do_aug=False, do_preproc=False)
        
        scores = rgb_embeddings.dot(tac_embedding.T)
        scores = np.split(scores,2)
        sim_score = np.max(scores,1)
            
        rot_rad = rotation*np.pi/180
        rot_mat = np.array([[np.cos(rot_rad),-np.sin(rot_rad)],[np.sin(rot_rad),np.cos(rot_rad)]])
        idx = np.argmax(sim_score)
        if idx==0:
            direc = np.array([0,1])
        if idx==1:
            direc = np.array([0,-1])

        final_direc = rot_mat.dot(direc.T).tolist() + [0,0,0,0]
        final_direc /= np.linalg.norm(final_direc)

        print(f"index {idx}")             
        return final_direc

    def tac_translate_conf(self,tac,rotation,transed_imgs):
        tac_imgs = [tac for i in range(len(transed_imgs))]
        all_rotations, probs = self.tester.query_rotation_network_batch(transed_imgs,im_scale=0.15,rgb_size=[128,128],tac_im = tac_imgs)
        scores = np.std(probs,-1)
        scores = np.split(scores,2)

        sim_score = np.max(scores,1)
        tran_idx = np.argmax(scores,1)
        max_rot = [all_rotations[tran_idx[0]], all_rotations[tran_idx[1]+len(transed_imgs)//2]]

        idx = np.argmax(sim_score)
        if idx==0:
            if tran_idx[idx]==0:
                direc = np.array([0,0])
            else:
                direc = np.array([0,1])
        if idx==1:
            if tran_idx[idx]==0:
                direc = np.array([0,0])
            else:
                direc = np.array([0,-1])

        newrot_rad = max_rot[idx]*np.pi/180
        newrot_mat = np.array([[np.cos(newrot_rad),-np.sin(newrot_rad)],[np.sin(newrot_rad),np.cos(newrot_rad)]])
        # final_direc = rot_mat.dot(direc.T).tolist() + [0,0,0,0]
        final_direc = newrot_mat.dot(direc.T).tolist() + [0,0,0,0]
        final_direc /= (np.linalg.norm(final_direc) + 1e-8)

        print(f"index {idx}")             
        return final_direc, idx,  max_rot[idx]

    def slide_till_find(self,reference_file=None, rgb_flag=True, thres=0.7):
        # self.capture.home()
        for _ in range(10):
            self.capture.d.get_frame(True)

        
        if reference_file is None:
            reference_img = self.take_reference_image(rgb_flag)
        else:
            reference_img = np.asarray(Image.open(reference_file))
        # plt.imshow(reference_img)
        # plt.savefig(f"{self.save_dir}/reference_img.jpg")
        # plt.show()

        stopper = ImpulseStopCondActive(self.tester, window_len=40,buff_len = 20, threshold=thres)
        # stopper.buff_len = 20

        self.capture.robot.start_freedrive()
        cmd = input("enter to loc the start point")
        if cmd != "":
            return 
        start_point = self.capture.robot.get_pose().translation
        input("enter to loc the end point")
        end_point = self.capture.robot.get_pose().translation
        self.capture.robot.stop_freedrive()
        self.move_linear(start_point, end_point, [0.001, 0.001, 0.001], stopper, reference_img)

    def rot2direc(self,vel,rot):
        #rot in degree with respect to the horizontal edge
        #vel is the velocity in the moving direction
        rot_rad = rot*np.pi/180
        x = vel * np.cos(rot_rad)
        y = vel * np.sin(rot_rad)

        direc = np.array([x,y,0,0,0,0])
        return direc/np.linalg.norm(direc)

    def move_linear(self, start, end, vel, stopper, reference_img, thres=20, acc=0.25):

        assert len(start) == 3 and len(end) == 3, "dimension of start and goal must be 3"
        assert len(vel) == 3, "vel is for x,y,z direction and the first nonzero one would be used"

        if self.edge_baseline:
            contact_area = ContactArea(self.edge_base)

        if abs(end[0] - start[0]) > abs(end[1] - start[1]):
            print("generataing based on x")
            direc_x = vel[0] * np.sign((end[0] - start[0] + 1e-6))
            direc_y = np.clip((end[1] - start[1]) / (end[0] - start[0] + 1e-6), -10, 10) * direc_x
        elif abs(end[1] - start[1]) > abs(end[0] - start[0]):
            print("generataing based on y")
            direc_y = vel[1] * np.sign((end[1] - start[1] + 1e-6))
            direc_x = np.clip((end[0] - start[0]) / (end[1] - start[1] + 1e-6), -10, 10) * direc_y

        dire = [direc_x, direc_y, 0, 0, 0, 0]
        print("dire", dire)
        # start_pose = l_p([start[0], start[1], -0.15])
        start_pose = l_p([start[0], start[1], -0.2])
        self.capture.robot.move_pose(start_pose, vel=0.2, acc=2)
        self.capture.robot.move_until_contact([0, 0, -0.1, 0, 0, 0], 20, acc=.1)

        tac_reading = self.capture.d.get_frame(True)
        tac = tac_reading.copy()
        tac = cv2.resize(tac, (5 * tac.shape[1], 5 * tac.shape[0]))
        cv2.imshow('tac image', tac)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print("breaking")
            cv2.destroyAllWindows()

        if input("start") != '':
            return
        # self.capture.robot.ur_c.speedL(dire, acceleration=acc)
        time.sleep(.5)
        startforce = np.array(self.capture.robot.ur_r.getActualTCPForce())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_data = []
        y_data = []
        y2_data = []
        # hl, = ax.plot([], [])
        index = 0
        tac_readings = []
        tac_embeddings = {}

        stop_flag = False
        dist=0
        stop=False
        vel = 0.003
        prev_direc = np.array([-1.,1.,0,0,0,0])
        prev_direc /= np.linalg.norm(prev_direc)
        prev_trans = np.array([-1.,0.,0,0,0,0])
        prev_trans /= np.linalg.norm(prev_trans)
        prev_rot = 0
        rgb_embeddings = self.cache_rgb_embeddings(reference_img)
        try:
            while True:
                index += 1
                force = np.array(self.capture.robot.ur_r.getActualTCPForce())
                cur_force = np.linalg.norm((startforce - force).dot(dire))
                tac_reading = self.capture.d.get_frame(True)
                

                if self.edge_baseline:
                    try:
                        major, _ = contact_area(tac_reading)
                        vec = major[1] - major[0]
                        # rotation = np.arctan(-vec[1]/vec[0]) * 360 / np.pi
                        rotation = 180 - np.arctan(vec[1]/vec[0]) * 180 / np.pi
                    except:
                        rotation = prev_rot
                else:
                    rotation, prob = self.tester.query_rotation_network(reference_img,im_scale=0.15,rgb_size=[128,128],tac_im = tac_reading)
                print(f"{index},{rotation}")

                prev_rot = rotation

                direc = self.rot2direc(1,rotation)
                rot_direc_raw = direc.copy()

                if not stop_flag:
                    trans_direc, trans_idx, rot = self.tac_translate_conf(tac_reading,rotation,rgb_embeddings)
                    trans_direc_non_smooth = trans_direc.copy()

                if np.sign(direc.dot(prev_direc)) > 1e-8:
                    print("align with prev")
                    direc *= np.sign(direc.dot(prev_direc))
                else:
                    sign = direc[0].copy()
                    direc[0] *= -sign #is the sign of x not y!!!
                    direc[1] *= -sign

                smoothing = 0.7
                direc = prev_direc * smoothing + (1 - smoothing) * direc
                prev_direc = direc.copy()

                smoothing = 0.1
                trans_direc = prev_trans * smoothing + (1 - smoothing) * trans_direc
                prev_trans = trans_direc.copy()
                
                all_direc = 1*direc+ 1.*trans_direc               
                all_direc /= (np.linalg.norm(all_direc) + 1e-8)
                all_direc *= vel

                self.capture.robot.ur_c.speedL(all_direc,acceleration=acc)

                x_data.append(index)
                y_data.append(rotation)
                y2_data.append(dist)

                ax.plot(x_data, y_data, 'b')
                ax.plot(x_data, y2_data, 'r')
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.01)

                tac = tac_reading.copy()
                cv2.putText(tac, f'rot,dist: {rotation:.2f},{dist:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
                # cv2.putText(tac, f'rot_idx,idx: {rot}, {trans_idx}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
                cv2.line(tac, (140,140), (int(140+trans_direc_non_smooth[0]*30),int(140-trans_direc_non_smooth[1]*30)), (255, 0, 0), 3) #need the negative sign due to cv2 vis
                cv2.line(tac, (140,140), (int(140+rot_direc_raw[0]*20),int(140-rot_direc_raw[1]*20)), (0, 255, 0), 3) #need the negative sign due to cv2 vis
                cv2.circle(tac, (140, 140), 5, (0, 255, 0), 5)

                cv2.line(tac, (90,140), (int(90+direc[0]*20),int(140-direc[1]*20)), (0, 255, 255), 3) #need the negative sign due to cv2 vis
                cv2.circle(tac, (90, 140), 5, (0, 255, 255), 5)

                new_tac = Image.fromarray(tac)
                new_tac.save(f"{self.save_dir}/image_new_tac_{index}.jpg")
                tac_readings.append(new_tac)
                tac = cv2.resize(tac, (5 * tac.shape[1], 5 * tac.shape[0]))
                cv2.imshow('tac image', tac)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    print("breaking")
                    cv2.destroyAllWindows()
                    break

                if cur_force > thres:
                    print("break due to exceeding force limit")
                    cv2.destroyAllWindows()
                    break
                if np.linalg.norm(self.capture.robot.get_pose().translation[:2] - end[:2]) < 0.01:
                    print("break due to reaching the end")
                    # cv2.destroyAllWindows()
                    break
                if stop:
                    print("break due to the stopper")
                    Image.fromarray(tac).save(f"{self.save_dir}/image_stop_tac.jpg")
                    tac = tac_reading.copy()
                    cv2.putText(tac, f'force,dist,stop: {cur_force:.2f},{dist:.2f}', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
                    new_tac = Image.fromarray(tac)
                    new_tac.save(f"{self.save_dir}/image_new_tac_stop{index}.jpg")
                    tac_readings.append(new_tac)
                    break

                time.sleep(0.008)
        except KeyboardInterrupt:
            np.save(f"{self.save_dir}/rotation.npy",y_data)
            self.capture.robot.ur_c.speedStop()
            time.sleep(0.5)
            cur_pos = self.capture.robot.get_pose().translation
            up_pos = l_p([cur_pos[0], cur_pos[1], self.capture.HOME_Z])
            self.capture.robot.move_pose(up_pos, interp='tcp', vel=0.1, acc=1)
            tac_readings[0].save(f"{self.save_dir}/out.gif", save_all=True,
                                append_images=tac_readings[1:], duration=100, loop=0)
            for index,tac_im in enumerate(tac_readings):
                tac_im.save(f"{self.save_dir}/image_tac_{index}.jpg")
            plt.savefig(f"{self.save_dir}/out.jpg")
            plt.close('all')

        with open(f"{self.save_dir}/tac_embeddings.json", "w") as outfile:
            json.dump(tac_embeddings, outfile, sort_keys=True, indent=2, cls=JEncoder)


class FindAndSlide(GuidedSearch, RotationSearch):
    def __init__(self, params, save_dir, debug=False) -> None:
        GuidedSearch.__init__(self, params, save_dir)
        self.colors = ['b','g','r','c','m','y']
        self.prev_direc = np.array([-1.,1.,0,0,0,0])
        self.prev_direc /= np.linalg.norm(self.prev_direc)
        self.prev_trans = np.array([-1.,0.,0,0,0,0])
        self.prev_trans /= np.linalg.norm(self.prev_trans)

    def slide_till_find(self,reference_file=None, refer_num=2, servo_refer=None):
        reference_imgs = []
        if reference_file is None:
            for _ in range(refer_num):
                reference_img = self.take_reference_image(True)
                reference_imgs.append(reference_img)
        else:
            for rfile in reference_file:
                reference_imgs.append( np.array(Image.open(rfile)))
        
        if servo_refer is None:
            self.servo_img = self.take_reference_image(True)
        else:
            self.servo_img =  np.array(Image.open(servo_refer))
        self.servo_embeddings = self.cache_rgb_embeddings(self.servo_img)#bad naming here, servo_embeddings are not embeddings but translated rgb image 
        
        rgb_batches = self.tester.preprocess_rgb(reference_imgs, 0.15, [128, 128])
        # for i,rgb in enumerate(rgb_batches.cpu().numpy()):
        #     plt.imshow(rgb.transpose(1,2,0))
        #     plt.savefig(f"{self.save_dir}/reference_img_{i}.jpg")
        #     plt.show()

        ref_embedding = self.tester.query_network_rgb(rgb_batches, 0.15, [128, 128], preprocess=False)

        cached_score = ref_embedding.dot(self.cache_embed.T)
        cached_score_partition = np.sort(cached_score,-1)
        thres = [np.median(cs[-400:]) for cs in cached_score_partition]
        # for cs in cached_score:
        #     self.vis_statistic([cs])

        print("thres",thres)
        stoppers = []
        for i,r_embed in enumerate(ref_embedding):
            stopper = SimilarityStopCond(self.tester, r_embed[None,...], 2, thres[i])
            stopper.buff_len = 10
            stoppers.append(stopper)

        self.capture.robot.start_freedrive()
        cmd = input("enter to loc the start point")
        if cmd != "":
            return
        start_point = self.capture.robot.get_pose().translation
        input("enter to loc the end point")
        end_point = self.capture.robot.get_pose().translation
        self.capture.robot.stop_freedrive()
        self.move_linear(start_point, end_point, [0.005, 0.005, 0.005], stoppers)
    
    def servo_dire(self,tac_reading,vel):
        rotation, prob = self.tester.query_rotation_network(self.servo_img,im_scale=0.15,rgb_size=[128,128],tac_im = tac_reading)
        direc = self.rot2direc(1,rotation)
        rot_direc_raw = direc.copy()

        trans_direc, trans_idx, rot = self.tac_translate_conf(tac_reading,rotation,self.servo_embeddings)
        trans_direc_non_smooth = trans_direc.copy()

        if np.sign(direc.dot(self.prev_direc)) > 1e-8:
            print("align with prev")
            direc *= np.sign(direc.dot(self.prev_direc))
        else:
            sign = direc[0].copy()
            direc[0] *= -sign #is the sign of x not y!!!
            direc[1] *= -sign

        smoothing = 0.7
        direc = self.prev_direc * smoothing + (1 - smoothing) * direc
        self.prev_direc = direc.copy()

        smoothing = 0.1
        trans_direc = self.prev_trans * smoothing + (1 - smoothing) * trans_direc
        self.prev_trans = trans_direc.copy()
        
        all_direc = 1*direc+ 1.*trans_direc
        
        
        all_direc /= (np.linalg.norm(all_direc) + 1e-8)
        all_direc *= vel

        return direc, trans_direc, all_direc
    
    def move_linear(self, start, end, vel, stoppers: StopCond, thres=20, acc=0.25):
        assert len(start) == 3 and len(end) == 3, "dimension of start and goal must be 3"
        assert len(vel) == 3, "vel is for x,y,z direction and the first nonzero one would be used"

        if abs(end[0] - start[0]) > abs(end[1] - start[1]):
            print("generataing based on x")
            direc_x = vel[0] * np.sign((end[0] - start[0] + 1e-6))
            direc_y = np.clip((end[1] - start[1]) / (end[0] - start[0] + 1e-6), -10, 10) * direc_x
        elif abs(end[1] - start[1]) > abs(end[0] - start[0]):
            print("generataing based on y")
            direc_y = vel[1] * np.sign((end[1] - start[1] + 1e-6))
            direc_x = np.clip((end[0] - start[0]) / (end[1] - start[1] + 1e-6), -10, 10) * direc_y

        init_dire = [direc_x, direc_y, 0, 0, 0, 0]
        self.prev_direc = np.array(init_dire)
        print("dire", init_dire)
        start_pose = l_p([start[0], start[1], -0.15])
        self.capture.robot.move_pose(start_pose, vel=0.2, acc=2)
        self.capture.robot.move_until_contact([0, 0, -0.1, 0, 0, 0], 16, acc=.15)

        tac_reading = self.capture.d.get_frame(True)
        tac = tac_reading.copy()
        tac = cv2.resize(tac, (5 * tac.shape[1], 5 * tac.shape[0]))
        cv2.imshow('tac image', tac)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print("breaking")
            cv2.destroyAllWindows()

        if input("start") != '':
            return
        print("xdd")
        self.capture.robot.ur_c.speedL(init_dire, acceleration=acc)
        time.sleep(.5)
        startforce = np.array(self.capture.robot.ur_r.getActualTCPForce())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_data = []
        y_data =[[] for _ in range(len(stoppers))]
        
        index = 0
        tac_readings = []
        new_tac_readings = []
        tac_embeddings = {}

        stop_flag = False
        servo_mode = False #if it finds one of the reference, enter servoing mode
        print("xdd")
        while True:
            index += 1
            tac_reading = self.capture.d.get_frame(True)
            tac_readings.append(tac)
            tac_embeddings[index] = self.tester.query_network_tac(tac_reading)
            # print("here")
            if not servo_mode:
                dire,rot_dire, trans_direc = init_dire.copy(),init_dire.copy(),init_dire.copy()
            else:
                print("enter into servo mode")
                rot_dire, trans_direc, dire = self.servo_dire(tac_reading,vel[0])
                self.capture.robot.ur_c.speedL(dire, acceleration=acc)

            force = np.array(self.capture.robot.ur_r.getActualTCPForce())
            cur_force = np.linalg.norm((startforce - force).dot(dire))

            stop, dist = [],[]
            for stopper in stoppers:
                sp, dt = stopper.should_stop(tac_reading, True)
                stop.append(sp)
                dist.append(dt)

            update = False
            for stopper in stoppers:
                update = update or stopper.update
            if not update:
                self.capture.robot.ur_c.speedL(np.array(dire/np.linalg.norm(dire))*0.0002, acceleration=acc)
                # self.capture.robot.ur_c.speedStop()
                if not stop_flag:
                    time.sleep(2)
                stop_flag = True
            elif stop_flag:
                self.capture.robot.ur_c.speedL(dire, acceleration=acc)
                stop_flag = False
            # hl.set_xdata(np.append(hl.get._xdata(), index))
            # hl.set_ydata(np.append(hl.get_ydata(), dist))
            x_data.append(index)
            for i,d in enumerate(dist):
                y_data[i].append(d)
                ax.plot(x_data,y_data[i],self.colors[i])
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
            # print("lol")
            tac = tac_reading.copy()
            for _,d in enumerate(dist):
                cv2.putText(tac, f'dist: {d:.2f}', (10, 50+30*_),cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
            cv2.line(tac, (140,140), (int(140+trans_direc[0]*30),int(140-trans_direc[1]*30)), (255, 0, 0), 3) #need the negative sign due to cv2 vis
            cv2.line(tac, (140,140), (int(140+rot_dire[0]*20),int(140-rot_dire[1]*20)), (0, 255, 0), 3) #need the negative sign due to cv2 vis
            cv2.circle(tac, (140, 140), 5, (0, 255, 0), 5)
            cv2.line(tac, (90,140), (int(90+dire[0]*20),int(140-dire[1]*20)), (0, 255, 255), 3) #need the negative sign due to cv2 vis
            cv2.circle(tac, (90, 140), 5, (0, 255, 255), 5)
            # print('asdf')
            new_tac = Image.fromarray(tac)
            new_tac_readings.append(new_tac)
            tac = cv2.resize(tac, (5 * tac.shape[1], 5 * tac.shape[0]))
            cv2.imshow('tac image', tac)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                print("breaking")
                cv2.destroyAllWindows()
                break

            if np.any(stop):
                servo_mode = True
                stop_idx = np.where(stop)[0][0]
                print(f"break to the stopper {stop_idx}")
                stoppers.pop(stop_idx) #if found one refere should pop the stopper condition
                y_data =[[] for _ in range(len(stoppers))]

                Image.fromarray(tac).save(f"{self.save_dir}/image_stop_tac_{stop_idx}.jpg")
                tac = tac_reading.copy()
                for _,d in enumerate(dist):
                    cv2.putText(tac, f'dist: {d:.2f}', (10, 50+30*_),cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
                new_tac = Image.fromarray(tac)
                new_tac.save(f"{self.save_dir}/image_new_tac_stop{index}.jpg")
                if len(stoppers)==0:
                    break

            if cur_force > thres:
                print("break due to exceeding force limit")
                cv2.destroyAllWindows()
                break
            if np.linalg.norm(self.capture.robot.get_pose().translation[:2] - end[:2]) < 0.01:
                print("break due to reaching the end")
                # cv2.destroyAllWindows()
                break

            time.sleep(0.008)

        self.capture.robot.ur_c.speedStop()
        time.sleep(0.5)
        cur_pos = self.capture.robot.get_pose().translation
        up_pos = l_p([cur_pos[0], cur_pos[1], self.capture.HOME_Z])
        self.capture.robot.move_pose(up_pos, interp='tcp', vel=0.1, acc=1)
        tac_readings = [Image.fromarray(a) for a in tac_readings]
        tac_readings[0].save(f"{self.save_dir}/out.gif", save_all=True,
                             append_images=tac_readings[1:], duration=100, loop=0)
        for index, (tac_im, new_tac_im) in enumerate(zip(tac_readings, new_tac_readings)):
            tac_im.save(f"{self.save_dir}/image_tac_{index}.jpg")
            new_tac_im.save(f"{self.save_dir}/image_new_tac_{index}.jpg")
        plt.savefig(f"{self.save_dir}/out.jpg")
        plt.close('all')

        np.save(f"{self.save_dir}/dist.npy",y_data)

        with open(f"{self.save_dir}/tac_embeddings.json", "w") as outfile:
            json.dump(tac_embeddings, outfile,sort_keys=True, indent=2, cls=JEncoder) 
        

def contact_area_test(save_dir, params):
    tester = Tester(params)
    reference_file="/home/ravenhuang/tac_vis/tac_vision/output_task/rotation_test_task/trial_31/images_set_1/image_reference_rgb.jpg"
    reference_img = np.asarray(Image.open(reference_file))
    capture = DataCaptureUR5(save_dir, robot=False)
    input('hit enter to take base')
    for _ in range(10):
        base = capture.d.get_frame(True)

    contact_area = ContactArea(base, draw_poly=True)

    while True:
        # input('hit enter to take contact')
        contact = capture.d.get_frame(True)
        try:
            
            major, minor = contact_area(contact)
        except:
            pass
        
        # vec = major[1] - major[0]
        # rotation = np.arctan(-vec[1]/vec[0]) * 360 / np.pi

        # rotation_net, _ = tester.query_rotation_network(reference_img,im_scale=0.15,rgb_size=[128,128],tac_im = contact)

        # print(vec)
        # print(major)
        # print(rotation)
        # print(rotation_net)
        cv2.imshow('howdy', contact)
        cv2.waitKey(1)
        # plt.show()

    breakpoint()



def tac_translate(tester,tac,reference_img):
    ori_tac = tac.copy()
    sc = StopCond(tester)

    h,w =  reference_img.shape[:2]
    offset = np.min([w,h])
    start = 0
    end = offset/2
    num = 10

    xtranses = []

    xtrans = np.linspace(-start,-end,num) # this is 0,1 direction in image 
    xtranses.append(xtrans)

    xtrans = np.linspace(start,end,num) # this is 0,-1 direction in image
    xtranses.append(xtrans)


    transed_imgs = []
    for xtra in xtranses:
        for tx in xtra: 
            T = np.float32([[1, 0, tx], [0, 1, 0]])
            transed_img = cv2.warpAffine(reference_img, T, (w, h))
            transed_imgs.append(transed_img)
            # plt.imshow(transed_img)
            # plt.show()
            # fig, ax = plt.subplots(len(transed_imgs))
            # import pdb;pdb.set_trace()
            # for i,tac_img in enumerate(transed_imgs):
            #     ax[i].imshow(tac_img)
            #     ax[i].text(100,100, scores[i], bbox=dict(fill=False, edgecolor='red', linewidth=2))
            # plt.show()
    tac_imgs = [tac for i in range(len(transed_imgs))]
    rotations, probs = tester.query_rotation_network_batch(transed_imgs,im_scale=0.15,rgb_size=[128,128],tac_im = tac_imgs)
    scores = np.std(probs,-1)

    rotation, prob = tester.query_rotation_network(reference_img,im_scale=0.15,rgb_size=[128,128],tac_im = tac)
    rgb_batches = tester.preprocess_rgb(transed_imgs, 0.15, [128,128] , True).cpu().numpy()
    # # plt.imshow(tac);plt.show()
    tac = sc.rot_aug(np.array(tac),1,[-rotation])[0]
    # plt.imshow(tac.permute(1,2,0));plt.show()
    # tac_embedding = tester.query_network_tac(tac, do_aug=False, do_preproc=False)
    # tac_embedding = tester.query_network_tac(tac, do_aug=False)
    # rgb_embeddings = tester.query_network_rgb(transed_imgs, 0.15, [128, 128])
    # scores = rgb_embeddings.dot(tac_embedding.T)
    scores = np.split(scores,len(scores)//num)
    # import pdb;pdb.set_trace()

    sim_score = []
    for s in scores:
        s = np.sort(s)[::-1]
        sim_score.append(np.median(s[:int(len(s)*0.2)]))

    # sim_score = np.max(scores,1)
    tran_idx = np.argmax(scores,1)
    # for i,idx in enumerate(tran_idx):
    #     plt.imshow(rgb_batches[idx+i*num].transpose(1,2,0))
    #     if i==0:
    #         plt.text(50,50,f"left")
    #     else:
    #         plt.text(50,50,f"right")
    #     plt.show()
    # for i,sscore in enumerate(scores):
    #     plt.plot(range(len(sscore)),sscore)
    #     if i==0:
    #         plt.text(0,0,f"left")
    #     else:
    #         plt.text(0,0,f"right")
    #     plt.show()
    
    max_rot = [rotations[tran_idx[0]], rotations[tran_idx[1]+num]]
    newtacs = []
    for r in max_rot:
        print(r)
        newtac = sc.rot_aug(np.array(ori_tac),1,[-r])[0]
        newtacs.append(newtac)
    #     plt.imshow(newtac.permute(1,2,0));plt.show()

    rot_rad = rotation*np.pi/180
    rot_mat = np.array([[np.cos(rot_rad),-np.sin(rot_rad)],[np.sin(rot_rad),np.cos(rot_rad)]])
    idx = np.argmax(sim_score)
    if idx==0:
        # direc = [-1,-1,0,0,0,0]
        direc = np.array([0,1])
    if idx==1:
        # direc = [-1,1,0,0,0,0]
        direc = np.array([0,-1])
    

    plt.imshow(tac.permute(1,2,0))
    plt.arrow(150,150,int(-direc[1]*20),int(direc[0]*20),
                                color=(0, 1, 0),
                                width=5,)
    plt.show()
    
    plt.imshow(newtacs[idx].permute(1,2,0))
    plt.arrow(150,150,int(-direc[1]*20),int(direc[0]*20),
                                color=(0, 1, 0),
                                width=5,)

    plt.show()
    
    final_direc = rot_mat.dot(direc.T).tolist() + [0,0,0,0]
    final_direc /= np.linalg.norm(final_direc)

    plt.imshow(ori_tac)
    plt.arrow(50,50,int(final_direc[0]*20),int(-final_direc[1]*20),color=(0, 1, 0),
                                width=5)
    plt.show()

    newrot_rad = max_rot[idx]*np.pi/180
    newrot_mat = np.array([[np.cos(newrot_rad),-np.sin(newrot_rad)],[np.sin(newrot_rad),np.cos(newrot_rad)]])
    final_direc = newrot_mat.dot(direc.T).tolist() + [0,0,0,0]
    final_direc /= np.linalg.norm(final_direc)

    plt.imshow(ori_tac)
    plt.arrow(50,50,int(final_direc[0]*20),int(-final_direc[1]*20),color=(0, 1, 0),
                                width=5)
    plt.show()
    # print(f"index {[sim_score_1,sim_score_2,sim_score_3,sim_score_4]}")   
    print(f"index {idx}")
    print(final_direc*0.001)             
    return final_direc
        

if __name__ == '__main__':

    with open('config/test_ur5.yaml', 'r') as stream:
        params = yaml.safe_load(stream)
    
    # tester = Tester(params)
    # tac = Image.open("/home/ravenhuang/tac_vis/tac_vision/output_task/rotation_test_task/trial_47/image_tac_209.jpg")
    # reference_img = np.array(Image.open("/home/ravenhuang/tac_vis/tac_vision/output_task/guided_slide_task/trial_33/images_set_36/image_reference_rgb.jpg"))
    # dire = tac_translate(tester,tac,reference_img)
    # exit()

    # contact_area_test("./output_task/guided_slide_fold", params)
    # 1/0

    # save_dir = "./output_task/guided_servo_test"
    # save_dir = "./output_task/guided_slide_fold" 
    # save_dir = "./output_task/guided_slide_task_button"
    # save_dir = "./output_task_0904_needle2/slide_task"
    # save_dir = "./output_task/rotation_test_task"
    # save_dir = "./output_task/needle_1"
    save_dir = "./output_task/test_pixel_diff"

    # Task = GuidedSearch(params,save_dir) # this is image guided search, remember to change the save+dir above to make sure the output is saved in the correct folder
    #when pixel_diff is true, is using the baseline
    Task = ImpulseSearch(params, save_dir, pixel_diff=True, debug=False)
    # Task = RotationSearch(params, save_dir)
    # Task = FindAndSlide(params,save_dir) 

    for _ in range(10):
        Task.capture.d.get_frame(True)

    # tac = Image.open("/home/ravenhuang/tac_vis/tac_vision/output_task/rotation_test_task/trial_45/image_new_tac_46.jpg")
    # reference_img = np.array(Image.open("/home/ravenhuang/tac_vis/tac_vision/output_task/rotation_test_task/trial_31/images_set_1/image_reference_rgb.jpg"))
    # rotation, prob = Task.tester.query_rotation_network(reference_img,im_scale=0.15,rgb_size=[128,128],tac_im = tac)

    # print(rotation)
    # exit()

    # Task.vis_tsne( "./output_task/slide_task/trial_43")
    # tester.test()
    ind = len(os.listdir(Task.save_dir))
    while True:
        if input('continue') != 'n':
            Task.save_dir = f"{save_dir}/trial_{ind}"

            os.makedirs(Task.save_dir, exist_ok=True)
            
            #when thres is <0, it will use adaptive thres, if no reference file, it will take one at the beginning of each experiment. Remember to check the vis to make sure it's the correct feature
            #reference image I have been using edge, better: /home/ravenhuang/tac_vis/tac_vision/output_task/guided_slide_task_button/images_rgb/image_reference_rgb.jpg
            #reference image I have been using edge:/home/ravenhuang/tac_vis/tac_vision/output_task/guided_slide_task/trial_33/images_set_36/image_reference_rgb.jpg
            #reference image I have been using zipper:/home/ravenhuang/tac_vis/tac_vision/output_task/guided_slide_task_multifab/images_rgb/image_reference_rgb.jpg
            #servo refer /home/ravenhuang/tac_vis/tac_vision/output_task/rotation_test_task/trial_31/images_set_1/image_reference_rgb.jpg
            # references = ["/home/ravenhuang/tac_vis/tac_vision/output_task/guided_slide_task_button/images_rgb/image_reference_rgb.jpg","/home/ravenhuang/tac_vis/tac_vision/output_task/guided_slide_task_button/images_rgb/image_reference_rgb.jpg"]
            # Task.slide_till_find(reference_file=references,servo_refer="/home/ravenhuang/tac_vis/tac_vision/output_task/rotation_test_task/trial_31/images_set_1/image_reference_rgb.jpg")

            # Task.slide_till_find(thres=-1)
            # Task.slide_till_find(reference_file=None,thres=-1)
            # Task.slide_till_find(thres=0.3, window=40)
            Task.slide_till_find(thres=0.7, window=40)
            # Task.slide_till_find(reference_file="/home/ravenhuang/tac_vis/tac_vision/output_task/rotation_test_task/trial_31/images_set_1/image_reference_rgb.jpg",thres=-1)
            # Task.vis_tsne(Task.save_dir)
            ind += 1
            # Task.manual_slide(thres = 0.3)
        else:
            break

    # Task = ManualStopTest(params,"./output_task/manual_slide_task",imagenet=False)
    # Task.slide_impulse()
