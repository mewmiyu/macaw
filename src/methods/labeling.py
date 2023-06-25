import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import numpy as np

class Labeler:

    def __init__(self):
        self.coords_x = []
        self.coords_y = []


    def labeling(self):
        images, _ , filenames = utils.load_data('data')
        images_coords = []
        for i,image in enumerate(images):
            self.coords_x = []
            self.coords_y = []
            self.image = image.permute(1,2,0)
            self.title = filenames[i]
            fig = plt.figure()
            plt.imshow(self.image, zorder=1)
            cid = fig.canvas.mpl_connect('button_press_event', self.onclick)            
            cid2 = fig.canvas.mpl_connect('key_press_event', self.on_press)
            plt.title(self.title)
            plt.show()


    def on_press(self, event):
        print('press', event.key)
        if event.key == 'x':
            print(self.coords_x, self.coords_y)
            plt.close()
        if event.key == 'r':
            self.coords_x = []
            self.coords_y = []
            plt.clf()
            plt.imshow(self.image, zorder=1)
            plt.title(self.title)
            plt.draw()
        if event.key == 'l':
            exit()
        if event.key == 'b':
            minx = np.min(self.coords_x)
            miny = np.min(self.coords_y)
            maxx = np.max(self.coords_x)
            maxy = np.max(self.coords_y)
            plt.clf()
            plt.scatter(self.coords_x, self.coords_y,zorder=2)
            plt.imshow(self.image, zorder=1)
            plt.plot([minx, minx],[miny, maxy], "red",zorder=2)
            plt.plot([maxx, maxx],[miny, maxy], "red",zorder=2)
            plt.plot([minx, maxx],[miny, miny], "red",zorder=2)
            plt.plot([minx, maxx],[maxy, maxy], "red",zorder=2)
            plt.title(self.title)
            plt.draw()
                    
    def onclick(self, event):
        ix, iy = event.xdata, event.ydata
        print (f'x = {ix}, y = {iy}')
        if ix != None and iy != None:
            self.coords_x.append(ix)
            self.coords_y.append(iy)
            plt.clf()
            plt.scatter(self.coords_x, self.coords_y,zorder=2)
            plt.imshow(self.image, zorder=1)
            plt.title(self.title)
            plt.draw()