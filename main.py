import math
import time

import pybullet as p
import pybullet_data


def create_plane():
    # planeId = p.loadURDF("plane.urdf")
    plane_id = p.loadURDF("plane100.urdf", useMaximalCoordinates=True)
    return plane_id

def create_cube():
    shift = [0, 0, 0]
    lengths = [0.1, 0.1, 0.1]
    start_pos = [0, 0, 1]
    pi4 = math.pi/4
    start_orn = p.getQuaternionFromEuler([pi4, pi4, pi4])

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


if __name__ == '__main__':
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-1)
    create_plane()
    cube_id = create_cube()
    #set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    for i in range (10000):
        p.stepSimulation()
        time.sleep(1./240.)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(cube_id)
    print(cubePos,cubeOrn)
    p.disconnect()
