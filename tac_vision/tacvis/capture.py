
from autolab_core import RigidTransform
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import threading
import os.path as osp
from ur5py.ur5 import UR5Robot
from PIL import Image

from perception import WebcamSensor
from digit_interface import Digit
import time

GRIP_DOWN_R = np.diag([1, -1, -1])
l_tcp_frame = "l_tcp"
r_tcp_frame = "r_tcp"
base_frame = "base_link"

def l_p(trans, rot=GRIP_DOWN_R):
    return RigidTransform(
        translation=trans,
        rotation=rot,
        from_frame=l_tcp_frame,
        to_frame=base_frame,
    )


def r_p(trans, rot=GRIP_DOWN_R):
    return RigidTransform(
        translation=trans,
        rotation=rot,
        from_frame=r_tcp_frame,
        to_frame=base_frame,
    )
    
class AsyncWrite(threading.Thread):
    def __init__(self, rgb, tactile, i, data_folder):
        # calling superclass init
        threading.Thread.__init__(self)
        self.rgb = rgb
        self.tactile = tactile
        self.i = i
        self.OUTPUT_DIR = data_folder

    def run(self):
        Image.fromarray(self.rgb).save(f"{self.OUTPUT_DIR}/images_rgb/image_{self.i}_rgb.jpg")
        Image.fromarray(self.tactile).save(f"{self.OUTPUT_DIR}/images_tac/image_{self.i}_tac.jpg")

class AsyncWriteTac(threading.Thread):
    def __init__(self, tactile, i, data_folder):
        # calling superclass init
        threading.Thread.__init__(self)
        self.tactile = tactile
        self.i = i
        self.OUTPUT_DIR = data_folder

    def run(self):
        Image.fromarray(self.tactile).save(f"{self.OUTPUT_DIR}/images_tac_{self.i}.jpg")
        
class DataCaptureUR5:
    # HOME_X=0.1
    HOME_X=0.08
    HOME_Y=0.625
    HOME_Z=-0.15
    HOME_JOINTS = [1.745178461074829, -2.035919491444723, 
            -1.9630869070636194, -0.7158778349505823, 1.581896424293518, 0.2129390835762024]
    CAM_OFFSET = np.array([.056,0,0])
    HIGH_RES=(4096,2160)
    RES=(1920,1080)
    #default params below:
    # FOCUS = 21
    FOCUS = 24
    ZOOM = 150
    TRANS_RAND = .01#random translation to add to samples
    # TRANS_RAND = .0#random translation to add to samples
    ROT_RAND = np.deg2rad(40)#random rotation to add to samples
    # ROT_RAND = np.deg2rad(0)#random rotation to add to samples
    def __init__(self,output_dir,robot= True,digit_tcp=RigidTransform(translation=[0,0,.07],from_frame='tcp',to_frame='base_link')):
        self.OUTPUT_DIR = output_dir
        if robot:
            self.robot = UR5Robot(gripper=False)
            self.robot.set_tcp(digit_tcp)
        #home the robot
        self.d = Digit("D20161") # Unique serial number
        self.d.connect()
        self.web_id=0
        # self.web_id=6
        self.web = WebcamSensor(device_id=self.web_id)
        self.web._adjust_exposure=False
        self.web.start()
        self.set_res(self.RES)
        self.idx = len(os.listdir(osp.join(self.OUTPUT_DIR, "images_rgb")))
        command = [ 
            "v4l2-ctl",
            f"-d /dev/video{self.web_id}",
            "-c exposure_auto=1",
            "-c exposure_auto_priority=1",
            "-c exposure_absolute=60",
            "-c white_balance_temperature_auto=1",
            f"-c zoom_absolute={self.ZOOM}",
        ]
        print("turning off focus")
        os.system(f"v4l2-ctl -d /dev/video{self.web_id} -c focus_auto=0")
        print("setting focus")
        os.system(f"v4l2-ctl -d /dev/video{self.web_id} -c focus_absolute={self.FOCUS}")
        print("sending exposure")
        os.system(" ".join(command))
    
    def home(self):
        self.robot.move_joint(self.HOME_JOINTS,vel=.4)

    def set_zoom(self,zoom):        
        cmd = f"v4l2-ctl -d /dev/video{self.web_id} -c zoom_absolute={zoom}"
        os.system(cmd)

    def set_focus(self,focus):
        cmd = f"v4l2-ctl -d /dev/video{self.web_id}  -c focus_auto=0 -c focus_absolute={focus}"
        os.system(cmd)
            
    def set_res(self,res):
        '''
        width,height
        '''
        self.web.res = res
        self.web._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.web._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  res[0])
        self.web._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        
    def collect_grid(self,x_n,y_n,x_size,y_size):
        self.home()
        self.robot.start_freedrive()
        input("go to start point")
        start = self.robot.get_pose().translation
        input("go to end point")
        end = self.robot.get_pose().translation
        self.robot.stop_freedrive()

        for i,x in enumerate(np.linspace(start[0],end[0],x_n)):
            ys = np.linspace(start[1],end[1],y_n)
            if i%2==1:
                #reverse the ys
                ys=ys[::-1]
            for y in ys:
                dp=np.random.uniform((-self.TRANS_RAND,-self.TRANS_RAND,-self.ROT_RAND),
                                            (self.TRANS_RAND,self.TRANS_RAND,self.ROT_RAND))
                pose = l_p([x+dp[0],y+dp[1],self.HOME_Z])
                pose.rotation = RigidTransform.z_axis_rotation(dp[2])@pose.rotation
                self.robot.move_pose(pose,interp='tcp',vel=.5,acc=.15)
                while not self.robot.is_stopped():
                    time.sleep(.008)
                im,tac=self.collect_pair()
                writer = AsyncWrite(im,tac, self.idx, self.OUTPUT_DIR)
                writer.start()
                self.idx += 1

        
    def collect_pair(self):
        digit_pose = self.robot.get_pose()
        time.sleep(2)
        rgb = self.web.frames(True)
        digit_pose = digit_pose*RigidTransform(translation=self.CAM_OFFSET+[0,0,.08],
                        from_frame = digit_pose.from_frame,to_frame = digit_pose.from_frame)
        self.robot.move_pose(digit_pose,interp='tcp',vel=0.5,acc=2)
        self.robot.move_until_contact([0,0,-0.11,0,0,0],25,acc=.4)#was 18
        time.sleep(1)
        tactile = self.d.get_frame(True)
        #move back up
        self.robot.move_pose(digit_pose,'tcp',.7,2)
        return rgb,tactile

    def manual_collect(self):
        for _ in range(10):
            self.d.get_frame(True)
        new_dir = osp.join(self.OUTPUT_DIR, f"images_set_{len(os.listdir(self.OUTPUT_DIR))+1}")
        os.makedirs(new_dir) 
        self.robot.start_freedrive()
        input("Press enter to take a picture")
        trans = self.robot.get_pose().translation
        self.robot.stop_freedrive()
        #use the x,y from this commanded pose
        camera_pose = l_p([trans[0],trans[1],self.HOME_Z + 0.15])
        # camera_pose.rotation = RigidTransform.z_axis_rotation(np.deg2rad(90))@camera_pose.rotation
        camera_pose = camera_pose*RigidTransform(translation=-self.CAM_OFFSET,
                        from_frame = camera_pose.from_frame,to_frame = camera_pose.from_frame)
        self.set_res(self.HIGH_RES)
        os.system(f"v4l2-ctl -d /dev/video{self.web_id} -c focus_absolute=13")
        self.robot.move_pose(camera_pose,interp='tcp',vel=0.2,acc=.2)
        time.sleep(.2)
        #capture the global image here
        global_rgb = self.web.frames(True)
        Image.fromarray(global_rgb).save(f"{new_dir}/image_global.jpg")
        #next freedrive to take tac readings
        self.robot.start_freedrive()
        idx = 0
        while True:
            cmd = input("Enter to take tac img, anything else to break")
            self.robot.stop_freedrive()
            if cmd != "":
                break
            trans = self.robot.get_pose().translation
            tac_pose = l_p([trans[0],trans[1],self.HOME_Z])
            # tac_pose.rotation = RigidTransform.z_axis_rotation(np.deg2rad(90))@camera_pose.rotation
            #we move the robot by cam offset before calling collect pair
            tac_pose = tac_pose*RigidTransform(translation=-self.CAM_OFFSET,
                        from_frame = tac_pose.from_frame,to_frame = tac_pose.from_frame)
            self.robot.move_pose(tac_pose,interp='tcp',vel=.2,acc=.2)
            rgb,tac = self.collect_pair()
            Image.fromarray(rgb).save(f"{new_dir}/image_rgb_{idx}.jpg")
            Image.fromarray(tac).save(f"{new_dir}/image_tac_{idx}.jpg")
            idx+=1
            self.robot.start_freedrive()
        return new_dir

    def test_collect(self,num,x_size,y_size,fixed=False,rotate=False,large_capture=True):
        new_dir = osp.join(self.OUTPUT_DIR, f"images_set_{len(os.listdir(self.OUTPUT_DIR))+1}")
        os.makedirs(new_dir) 
        if large_capture:
            camera_pose = l_p([self.HOME_X,self.HOME_Y-0.03,self.HOME_Z+0.15])   
            camera_pose.rotation = RigidTransform.z_axis_rotation(np.deg2rad(90))@camera_pose.rotation
            self.set_res(self.HIGH_RES)
            print("setting focus for global")
            os.system(f"v4l2-ctl -d /dev/video{self.web_id} -c focus_absolute=13")
            self.robot.move_pose(camera_pose,interp='tcp',vel=0.5,acc=.2)
            while not self.robot.is_stopped():
                time.sleep(.008)
            time.sleep(1)
            global_rgb = self.web.frames(True)
            Image.fromarray(global_rgb).save(f"{new_dir}/image_global.jpg")
            
        self.set_res(self.RES)
        print("setting focus for local")
        os.system(f"v4l2-ctl -d /dev/video{self.web_id} -c focus_absolute={self.FOCUS}")
        if not fixed:
            samples = np.random.uniform([self.HOME_X-x_size/2,self.HOME_Y-y_size/2],[self.HOME_X+x_size/2,self.HOME_Y+y_size/2],(num,2))
        else:
            samples = np.array([self.HOME_X,self.HOME_Y])[None,:]
        for idx,sample in enumerate(samples):     
            web_pose =l_p([sample[0],sample[1],self.HOME_Z])
            if rotate:
                dp=np.random.uniform((-self.TRANS_RAND,-self.TRANS_RAND,-self.ROT_RAND),
                                                (self.TRANS_RAND,self.TRANS_RAND,self.ROT_RAND))
                web_pose.rotation = RigidTransform.z_axis_rotation(dp[2])@web_pose.rotation
            self.robot.move_pose(web_pose,interp='tcp',vel=0.5,acc=1)
            
            if not large_capture:
                self.set_res(self.HIGH_RES)
                while not self.robot.is_stopped():
                    time.sleep(.008)
                global_rgb = self.web.frames(True)
                time.sleep(1)
                Image.fromarray(global_rgb).save(f"{new_dir}/image_global.jpg")
                self.set_res(self.RES)
                
            while not self.robot.is_stopped():
                time.sleep(.008)
            rgb,tac = self.collect_pair()
            Image.fromarray(rgb).save(f"{new_dir}/image_rgb_{idx}.jpg")
            Image.fromarray(tac).save(f"{new_dir}/image_tac_{idx}.jpg")
            
        return new_dir
    
    
        