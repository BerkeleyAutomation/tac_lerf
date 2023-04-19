import yaml
from tacvis.tester import Tester
            


                
if __name__=='__main__':
    with open('config/test_ur5.yaml', 'r') as stream:
        params = yaml.safe_load(stream)
    
    tester = Tester(params)
    tester.test()
    # tester.test_live('/home/ravenhuang/tac_vis/tac_vision/output_test/heatmap_test_wr/images_set_4', True)
    # tester.test_live("/home/ravenhuang/tac_vis/tac_vision/output_test_live/rotation/images_set_9", True)
    # tester.test_live(None, True)

