import numpy as np
from visualization import YardVisualizer
def generate_demo_image():
    visualizer = YardVisualizer(tiers= 6 , rows= 4, bays=10)

    yard3d = np.zeros((6, 4, 10), dtype=np.float32)
    yard3d[0,0,0] = 12
    yard3d[1,0,0] = 5
    yard3d[2,0,0] = 4
    yard3d[3,0,0] = 3

    yard3d[0,3,4] = 8
    yard3d[1,3,4] = 4
    yard3d[2,3,4] = 3

    yard3d[0, 2, 6] = 8
    yard3d[1, 2, 6] = 4

    yard3d[0, 2, 7] = 10
    yard3d[0, 2, 8] = 9
    yard3d[1, 2, 8] = 7

    yard3d[0, 0, 7] = 9
    yard3d[1, 0, 7] = 4
    yard3d[0, 0, 8] = 11
    yard3d[1, 0, 8] = 4
    yard3d[0, 0, 6] = 12
    yard3d[1, 0, 6] = 6
    yard3d[0, 2, 5] = 11
    yard3d[0, 2, 4] = 12
    yard3d[0, 2, 3] = 5


    visualizer.set_yard( yard=yard3d)
    visualizer.draw_yard()

generate_demo_image()