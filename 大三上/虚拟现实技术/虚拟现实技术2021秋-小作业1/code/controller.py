import time
import glfw
import numpy as np
from transform import (
    modelTransform,
    viewTransform,
    perspectiveProjection,
    orthogonalProjection,
)


SPEED_T = 1
SPEED_R = 100
SPEED_S = 1


class StateController:
    def __init__(self, window, winSize):
        self.window = window
        self.winWidth, self.winHeight = winSize

        self.offsets = [0.0, 0.0, 0.0]
        self.angles = [0.0, 0.0, 0.0]
        self.scales = [1.0, 1.0, 1.0]
        self.projectionType = "perspective"
        self.elevation = 45.0
        self.azimuth = 135.0
        self.distance = 5.0
        self.orthoScale = 1.0
        self.readyToSwitch = False

        self.time = time.time()

    def update(self):
        deltaTime = time.time() - self.time
        self.time = time.time()

        # translation offsets
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            self.offsets[0] += deltaTime * SPEED_T
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.offsets[0] -= deltaTime * SPEED_T
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.offsets[1] += deltaTime * SPEED_T
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.offsets[1] -= deltaTime * SPEED_T
        if glfw.get_key(self.window, glfw.KEY_Z) == glfw.PRESS:
            self.offsets[2] += deltaTime * SPEED_T
        if glfw.get_key(self.window, glfw.KEY_X) == glfw.PRESS:
            self.offsets[2] -= deltaTime * SPEED_T

        # rotation angles
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            self.angles[0] += deltaTime * SPEED_R
        if glfw.get_key(self.window, glfw.KEY_R) == glfw.PRESS:
            self.angles[0] -= deltaTime * SPEED_R
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.angles[1] += deltaTime * SPEED_R
        if glfw.get_key(self.window, glfw.KEY_F) == glfw.PRESS:
            self.angles[1] -= deltaTime * SPEED_R
        if glfw.get_key(self.window, glfw.KEY_C) == glfw.PRESS:
            self.angles[2] += deltaTime * SPEED_R
        if glfw.get_key(self.window, glfw.KEY_V) == glfw.PRESS:
            self.angles[2] -= deltaTime * SPEED_R

        # scaling factors
        if glfw.get_key(self.window, glfw.KEY_T) == glfw.PRESS:
            self.scales[0] += deltaTime * SPEED_S
        if glfw.get_key(self.window, glfw.KEY_Y) == glfw.PRESS:
            self.scales[0] -= deltaTime * SPEED_S
        if glfw.get_key(self.window, glfw.KEY_G) == glfw.PRESS:
            self.scales[1] += deltaTime * SPEED_S
        if glfw.get_key(self.window, glfw.KEY_H) == glfw.PRESS:
            self.scales[1] -= deltaTime * SPEED_S
        if glfw.get_key(self.window, glfw.KEY_B) == glfw.PRESS:
            self.scales[2] += deltaTime * SPEED_S
        if glfw.get_key(self.window, glfw.KEY_N) == glfw.PRESS:
            self.scales[2] -= deltaTime * SPEED_S

        # switch projection type
        if glfw.get_key(self.window, glfw.KEY_1) == glfw.PRESS:
            self.readyToSwitch = True
        if glfw.get_key(self.window, glfw.KEY_1) == glfw.RELEASE and self.readyToSwitch:
            self.readyToSwitch = False
            self.projectionType = (
                "perspective" if self.projectionType == "orthogonal" else "orthogonal"
            )

        # camera elevation, azimuth and distance
        if glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS:
            self.elevation += deltaTime * SPEED_R
        if glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS:
            self.elevation -= deltaTime * SPEED_R
        # restrict the camera elevation to be in (-90, 90)
        self.elevation = np.clip(self.elevation, -90 + 1e-5, 90 - 1e-5)
        if glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
            self.azimuth += deltaTime * SPEED_R
        if glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
            self.azimuth -= deltaTime * SPEED_R
        if glfw.get_key(self.window, glfw.KEY_O) == glfw.PRESS:
            if self.projectionType == "perspective":
                self.distance += deltaTime * SPEED_T
            else:
                self.orthoScale += deltaTime * SPEED_S
        if glfw.get_key(self.window, glfw.KEY_P) == glfw.PRESS:
            if self.projectionType == "perspective":
                self.distance -= deltaTime * SPEED_T
            else:
                self.orthoScale -= deltaTime * SPEED_S

    def getMVP(self):
        mMat = modelTransform(self.offsets, self.angles, self.scales)
        # the camera always looks at world origin
        elevation, azimuth = np.radians(self.elevation), np.radians(self.azimuth)
        camera_pos = (
            self.distance * np.cos(elevation) * np.sin(azimuth),
            self.distance * np.sin(elevation),
            -self.distance * np.cos(elevation) * np.cos(azimuth),
        )
        vMat = viewTransform(camera_pos, (0, 0, 0), (0, 1, 0))
        if self.projectionType == "orthogonal":
            pMat = orthogonalProjection(
                -2 * self.orthoScale,
                2 * self.orthoScale,
                -2 * self.orthoScale,
                2 * self.orthoScale,
                0.1,
                100,
            )
        else:
            pMat = perspectiveProjection(45, self.winWidth / self.winHeight, 0.1, 100)
        MVP = pMat @ vMat @ mMat
        MVP = np.ascontiguousarray(MVP.astype(np.float32))
        return MVP
