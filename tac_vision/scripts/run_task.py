

import os
import yaml
import argparse
from tacvis.task import ImpulseSearch,RotationSearch,GuidedSearch,FindAndSlide

def run_task(Task,thres,window=None,reference_file=None):
    for _ in range(10):
        Task.capture.d.get_frame(True)

    ind = len(os.listdir(Task.save_dir))
    while True:
        if input('continue') != 'n':
            Task.save_dir = f"{save_dir}/trial_{ind}"
            os.makedirs(Task.save_dir, exist_ok=True)
            if reference_file is None:
                if window is not None:
                    Task.slide_till_find(thres=thres, window=window)
                else:
                    Task.slide_till_find(thres=thres)
            else:
                Task.slide_till_find(reference_file=reference_file,thres=thres)
            ind += 1
        else:
            break


if __name__ == '__main__':
    reference_edge = "/home/ravenhuang/tac_vis/tac_vision/output_task/guided_slide_task_button/images_rgb/image_reference_rgb.jpg"
    reference_zipper = "/home/ravenhuang/tac_vis/tac_vision/output_task/guided_slide_task_multifab/images_rgb/image_reference_rgb.jpg"
    servo_refer = "/home/ravenhuang/tac_vis/tac_vision/output_task/rotation_test_task/trial_31/images_set_1/image_reference_rgb.jpg"
    
    with open('config/test_ur5.yaml', 'r') as stream:
        params = yaml.safe_load(stream)
    
    parser = argparse.ArgumentParser(description="Run the task")
    parser.add_argument("--task", type=str)
    parser.add_argument("--thres", type=float, default=-1)
    args = parser.parse_args()
    
    if "anomaly" in args.task:
        if "bs" in args.task:
            save_dir = './output_task/anomaly/test_pixel_diff'
            Task = ImpulseSearch(params, save_dir, pixel_diff=True)
        else:
            save_dir = './output_task/anomaly/similarity'
            Task = ImpulseSearch(params, save_dir, pixel_diff=False)
        Task.const_z = -.287
        run_task(Task,args.thres,40)

    elif "guided" in args.task:
        if "bs" in args.task:
            save_dir = './output_task/guided/bs'
            Task = ImpulseSearch(params, save_dir)
            run_task(Task,args.thres)
        else:
            save_dir = './output_task/guided/refer'
            Task = GuidedSearch(params,save_dir)
            run_task(Task,args.thres,reference_file = reference_edge)

    elif "servo" in args.task:
        if "bs" in args.task:
            save_dir = './output_task/servo/bs'
            Task = RotationSearch(params, save_dir,edge_baseline=True)
        else:
            save_dir = './output_task/servo/rot'
            Task = RotationSearch(params,save_dir)
        
        run_task(Task,args.thres,reference_file =servo_refer)
