import math
import time

import numpy as np
import pybullet as p
import pybullet_data


def create_plane():
    # planeId = p.loadURDF("plane.urdf")
    plane_id = p.loadURDF("plane100.urdf", useMaximalCoordinates=True)
    return plane_id

def create_cube(side_len, start_pos, start_orn, rgba):
    shift = [0, 0, 0]
    lengths = [side_len] * 3

    vis_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                 halfExtents=lengths,
                                 rgbaColor=rgba,
                                 specularColor=[0.4, 0.4, 0],
                                 visualFramePosition=shift)

    col_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                    halfExtents=lengths,
                                    collisionFramePosition=shift)

    cube_id = p.createMultiBody(baseMass=1,
                                baseInertialFramePosition=[0, 0, 0],
                                baseCollisionShapeIndex=col_id,
                                baseVisualShapeIndex=vis_id,
                                basePosition=start_pos,
                                baseOrientation=start_orn,
                                useMaximalCoordinates=True)

    return cube_id

##### CAMERA SETTINGS #####
camTargetPos = [0, 0, 0.5]
cameraUp = [0, 0, 1]
cameraPos = [1, 1, 1]

pitch = -10.0

yaw = 0
roll = 0
upAxisIndex = 2
camDistance = 2
pixelWidth = 320
pixelHeight = 200
nearPlane = 0.01
farPlane = 100

fov = 60
###########################

def step():
    p.stepSimulation()

    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                     roll, upAxisIndex)
    aspect = pixelWidth / pixelHeight
    projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
    img_arr = p.getCameraImage(pixelWidth,
                               pixelHeight,
                               viewMatrix,
                               projectionMatrix,
                               shadow=1,
                               lightDirection=[1, 1, 1],
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)

    w, h, rgba, depth, seg = img_arr

    np_img_arr = np.reshape(rgba, (h, w, 4))
    # Normalize to [0, 1]
    # np_img_arr = np_img_arr * (1. / 255.)

    return np_img_arr

def setup(gui=True, gravity=0, **cube_params):
    mode = p.GUI if gui else p.DIRECT
    background_options = '--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0'

    p.connect(mode, options=background_options)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, gravity)

    # plane_id = create_plane()
    cube_id = create_cube(**cube_params)

    return cube_id


DEFAULT_CUBE_PARAMS = {
        'side_len': 0.1,
        'start_pos': [0, 0, 1],
        'start_orn': p.getQuaternionFromEuler([math.pi/4]*3)
}

if __name__ == '__main__':
    pass
