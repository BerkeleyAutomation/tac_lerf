from digit_interface import Digit
from perception import WebcamSensor
import pprint
from digit_interface.digit_handler import DigitHandler
import matplotlib.pyplot as plot
import time
from PIL import Image


digits = DigitHandler.list_digits()
print("Connected DIGIT's to Host:")
pprint.pprint(digits)

# d = Digit("D20001") # Unique serial number
d = Digit("D20165") # Unique serial number
d.connect()
d.show_view()
frame = d.get_frame()
# d.disconnect()
# plot.imshow(frame)
# plot.show()

web = WebcamSensor(device_id=1)
web.start()
image = web.frames(True)
# import pdb;pdb.set_trace()

frams = []
for i in range(10):
    frame = d.save_frame(f'./tac_{i}.png')
    image = web.frames(True)
    Image.fromarray(image).save(f'./rgb_{i}.png')
    time.sleep(0.1)

d.disconnect()