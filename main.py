import math
import time

import numpy as np
import pybullet as p
import pybullet_data


def create_plane():
    # planeId = p.loadURDF("plane.urdf")
    plane_id = p.loadURDF("plane100.urdf", useMaximalCoordinates=True)
    return plane_id

def create_cube(side_len, start_pos, start_orn):
    shift = [0, 0, 0]
    lengths = [side_len] * 3

    vis_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                 halfExtents=lengths,
                                 rgbaColor=[1, 1, 1, 1],
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
camTargetPos = [0, 0, 0]
cameraUp = [0, 0, 1]
cameraPos = [1, 1, 1]

pitch = -10.0

yaw = 0
roll = 0
upAxisIndex = 2
camDistance = 4
pixelWidth = 320
pixelHeight = 200
nearPlane = 0.01
farPlane = 100

fov = 60
###########################

if __name__ == '__main__':
    physicsClient = p.connect(p.GUI) # p.DIRECT for non-graphical version

    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally
    plane_id = create_plane()

    start_pos = [0, 0, 1]
    start_orn = p.getQuaternionFromEuler([math.pi/4]*3)
    side_len = 0.1

    cube_id = create_cube(side_len, start_pos, start_orn)
    p.setGravity(0,0,-10)

    NUM_STEPS = 10000
    for i in range(NUM_STEPS):
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
        w = img_arr[0]  # width of the image, in pixels
        h = img_arr[1]  # height of the image, in pixels
        rgb = img_arr[2]  # color data RGB
        dep = img_arr[3]  # depth data

        np_img_arr = np.reshape(rgb, (h, w, 4))
        # Normalize to [0, 1]
        np_img_arr = np_img_arr * (1. / 255.)

        # Save the images into some kind of buffer here

        time.sleep(1./240.)

    # Useful function for ground truth cube position/orientation
    cubePos, cubeOrn = p.getBasePositionAndOrientation(cube_id)
    # print(cubePos,cubeOrn)

    p.disconnect()
