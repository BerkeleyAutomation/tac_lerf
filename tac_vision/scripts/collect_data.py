from autolab_core import RigidTransform
import numpy as np
import matplotlib.pyplot as plt
from tacvis.capture import DataCaptureUR5,l_p
import os 
from PIL import Image

OUTPUT_DIR = "data/test_data"

def collect():
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    #also make the images_rgb and images_tac folders
    os.makedirs(f"{OUTPUT_DIR}/images_rgb",exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images_tac",exist_ok=True)
    # capture=DataCaptureYuMi(OUTPUT_DIR)
    capture=DataCaptureUR5(OUTPUT_DIR)
    # print("pose",capture.robot.get_pose())
    # print("joints",capture.robot.get_joints())
    capture.home()
    while input("Collect? [y]/n")!='n':
        capture.collect_grid(x_n=30,y_n=15,x_size=.43,y_size=.23)
        # capture.collect_grid(x_n=40,y_n=30,x_size=.23,y_size=.18)
        capture.home()

def calibrate_offset():
    capture=DataCaptureUR5(OUTPUT_DIR)
    capture.home()
    while True:
        offset = input("enter offset")
        offset = float(offset)
        if abs(offset)>.1:
            print("too high")
            continue
        capture.robot.move_pose(l_p((capture.HOME_X,capture.HOME_Y,capture.HOME_Z)))
        capture.CAM_OFFSET=np.array([offset,0,0])
        rgb,tac=capture.collect_pair()
        fig,ax=plt.subplots(1,2)
        ax[0].imshow(rgb)
        ax[1].imshow(tac)
        ax[0].plot(1920/2,1080/2,'r*')
        plt.show()

def collect_background():
    capture=DataCaptureUR5(OUTPUT_DIR)
    capture.home()
    backgrounds = [capture.d.get_frame() for _ in range(100)]
    mean_background = np.mean(backgrounds, axis=0).astype(np.uint8)
    plt.imshow(mean_background)
    plt.show()
    img = Image.fromarray(mean_background)
    img.save(f'{OUTPUT_DIR}/tac_background.jpg')
    print("Saved background image")


if __name__ == "__main__":
    # calibrate_offset()
    collect()
    # collect_background()