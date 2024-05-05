from matplotlib import pyplot as plt
import matplotlib.artist as artist
import numpy as np
import cv2
from ipywidgets import widgets
import IPython.display as Disp


# %matplotlib notebook
# bbox_select(dir_= "revised_videos/frames/0679_frames/f00", frame=10)
# matplotlib.pyplot.connect() 



# bbox_select(dir_="revised_videos/frames/0679_frames/f00", frame=30)
frame = input("Input the video you want to use (Select from 679, 687, 688, 689)")
im = cv2.imread("revised_videos/frames/0679_frames/f0016.jpg")
# im = cv2.imread("revised_videos/frames/2363_frames/f0016.jpg")

coordinates = []


def on_click(event):
    coordinates.append((event.xdata, event.ydata))
    print('you clicked', event.button, event.xdata, event.ydata)

    if len(coordinates) == 8:
        f = open("background_keypoints_angle1.txt", "w")
        for i in range(8):
            f.write(str(coordinates[i][0]) + "," + str(coordinates[i][1])+ "\n")

        fig.canvas.mpl_disconnect(cid)
    return coordinates

fig, ax = plt.subplots()
ax.imshow(im)

cid = fig.canvas.mpl_connect('button_press_event', on_click)

# for i in range(10):
#     ax.scatter(np.random.random(), np.random.random(), picker=True, pickradius=10)
plt.show()