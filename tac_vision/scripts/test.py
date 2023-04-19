from tacvis.models import DEVICE, Encoder,compute_similarity_heatmap
from tacvis.dataset import PREPROC_IMG
from tacvis.capture import DataCapture,l_p,r_p
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import time
'''
tests to run:
1. towels vs towels+flip augmentations: ancientfog vs summer-silence-189
    summer-silence much less activated, peaks on an edge not the corner
2. no reg vs reg: colorful-dream vs coolwave and restfulmorning
        coolwave way better than colorfuldream, peaks at correct location
        restfulmorning also better, but fuzzier and peaks along the whole edge
3. newfocus with no flipping: 
    no flipping much worse
4. 128 vs 32: coolwave+restfulmorning vs colorfulbrook
'''
# old towels, no flilpping
# model_dir = 'output/ancient-fog-150'
# chkpt = 100

# old towels with 128, flipping
# model_dir='output/summer-silence-189'
# chkpt=300

# newfocus with 128 + reg
# model_dir = 'output/colorful-brook-191'
# chkpt = 300

# 128+reg + no flip
# model_dir = 'output/eternal-field-192'
# chkpt=300

# 128 + reg + dropout 10%
# model_dir='output/vivid-microwave-193'
# chkpt=300

#128 + reg + dropout 20%
# model_dir='output/still-waterfall-194'
# chkpt=300


# first time using newfocus
# model_dir = 'output/colorful-dream-173'
# chkpt = 300

# newfocus with no flipping
# model_dir = 'output/eternal-field-192'
# chkpt=300

# newfocus with reg
# model_dir='output/cool-wave-179'
# chkpt=300

# newfocus with reg and dropout
# model_dir='output/restful-morning-180'
# chkpt=300

#good grayscale one
# GRAYSCALE=True 
# model_dir='output/visionary-sun-227'
# chkpt=500

#"the one"
# model_dir='output/olive-planet-224'
# chkpt=500

#big hue variation, independent aug
#pretty good
# model_dir='output/rare-plant-218'
# chkpt=500

#LESS SCALE AUG
# model_dir='output/decent-yogurt-204'
# chkpt=500

#big hue variation, paired aug
#this one is pretty noisy
# model_dir='output/resilient-wildflower-219'
# chkpt=500

#grayscale, lots of independent aug
# model_dir='output/olive-flower-228'
# chkpt = 500
# GRAYSCALE = True

#grayscale, lots of independent aug, less reg
# model_dir='output/peachy-thunder-229'
# chkpt = 500
# GRAYSCALE = True

#grayscale, lots of independent aug, 64dim
# model_dir='output/worthy-glitter-231'
# chkpt = 500
# GRAYSCALE = True
#actually good one
# model_dir='output/revived-sky-238'
# chkpt = 500
# GRAYSCALE = True

model_dir= 'output/pleasant-vortex-316'
chkpt=500
GRAYSCALE=True



# model_dir='output_ae/ae_ae+contrastive_deep'
# chkpt=80
# GRAYSCALE=True

if 'GRAYSCALE' not in globals():
    GRAYSCALE=False
RES=(4096,2160)
LOAD=True
SAVE=True
out_dir='test_out/'
heatmap_dir='data/heatmap_tests'
heatmap_idx = len(os.listdir(heatmap_dir)) + 1
idx = len(os.listdir(out_dir)) + 1

if __name__=='__main__':
    with open('config/train.yaml', 'r') as stream:
        params = yaml.safe_load(stream)
    img_encoder = Encoder(params)
    tac_encoder = Encoder(params)
    tac_encoder.load(f'{model_dir}/tac_encoder_{chkpt}.pt')
    img_encoder.load(f'{model_dir}/img_encoder_{chkpt}.pt')
    if LOAD:
        #load pic and tactile
        pic=np.load(f'{heatmap_dir}/img0.npy')
        tactile=np.load(f'{heatmap_dir}/tac0.npy')
        if GRAYSCALE:
            pic = np.mean(pic,axis=2).astype(np.uint8)
            #convert to 3 channel
            pic = np.stack([pic,pic,pic],axis=2)
        scales=[.55]
        heatmaps,best_crops = compute_similarity_heatmap(pic,[tactile],scales,img_encoder,tac_encoder,stride=RES[0]//100)
        for i,h in enumerate(heatmaps):
            fig,axs=plt.subplots(1,5)
            fig.set_size_inches(16,4)
            axs[0].imshow(pic)
            axs[1].imshow(h)
            axs[2].imshow(best_crops[i])
            # axs[3].imshow(rgb_gt)
            #rotat tactile image 90 deg
            tactile=np.rot90(tactile,k=1)
            axs[4].imshow(tactile)
            plt.savefig(f"{out_dir}/scale_{scales[i]}_id{idx}.jpg",dpi=300)
            plt.show()
            idx+=1
        exit()
    capture = DataCapture('data/data')
    #take a tactile reading and save it
    while input("Test? [y]/n") != 'n':
        capture.set_res(RES)
        capture.set_zoom(100)
        capture.set_focus(24)
        capture.iface.go_pose_plan_single("left", 
                    l_p([capture.HOME_X,capture.HOME_Y-capture.CAM_Y_OFFSET/2,capture.HOME_Z+.05]),table_z=.1,mode='Manipulation1')
        capture.iface.sync()
        time.sleep(.5)
        pic = capture.web.frames(True)
        print("pic",pic.shape)

        #move down to touch
        capture.iface.go_pose("left", l_p([capture.HOME_X,capture.HOME_Y,capture.HOME_Z]),linear=False)
        capture.iface.sync()
        capture.set_zoom(500)
        capture.set_focus(55)
        capture.set_res((1280,720))
        rgb_gt = capture.web.frames(True)
        digit_pose=capture.iface.y.left.get_pose()
        
        digit_pose.translation -= [0,capture.CAM_Y_OFFSET,0]
        capture.iface.go_pose("left", digit_pose,linear=False)
        capture.iface.sync()
        n_touch_readings=1
        tac_frames=[]
        for i in range(n_touch_readings):
            if i==0:
                digit_pose.translation -= [0,0,capture.TOUCH_DISTANCE]
            else:
                #move down by .02
                digit_pose.translation -= [0,0,.02]
            capture.iface.go_pose("left", digit_pose,linear=True)
            capture.iface.sync()
            tactile = capture.d.get_frame()
            tac_frames.append(tactile)
            #move back up
            digit_pose.translation += [0,0,.02]
            capture.iface.go_pose("left", digit_pose,linear=True)
            capture.iface.sync()
        if SAVE:
            np.save(f'{heatmap_dir}/img{heatmap_idx}.npy',pic)
            np.save(f'{heatmap_dir}/tac{heatmap_idx}.npy',tactile)
            heatmap_idx+=1
        if GRAYSCALE:
            pic = np.mean(pic,axis=2).astype(np.uint8)
            #convert to 3 channel
            pic = np.stack([pic,pic,pic],axis=2)
        #compute the similarity heatmap
        scales=[.55]
        heatmaps,best_crops = compute_similarity_heatmap(pic,tac_frames,scales,img_encoder,tac_encoder,stride=RES[0]//100)
        for i,h in enumerate(heatmaps):
            fig,axs=plt.subplots(1,5)
            fig.set_size_inches(16,4)
            axs[0].imshow(pic)
            axs[1].imshow(h)
            axs[2].imshow(best_crops[i])
            axs[3].imshow(rgb_gt)
            #rotat tactile image 90 deg
            tactile=np.rot90(tactile,k=1)
            axs[4].imshow(tactile)
            plt.savefig(f"{out_dir}/scale_{scales[i]}_id{idx}.jpg",dpi=300)
            plt.show()
            idx+=1