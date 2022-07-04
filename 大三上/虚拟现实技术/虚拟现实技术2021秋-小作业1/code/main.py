import os
import sys

if sys.platform == "darwin":
    os.environ["DYLD_LIBRARY_PATH"] = "libglfw/darwin"

from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

from PIL import Image
from shader import Shader
from assets.model import vertexBufferData, uvBufferData
from controller import StateController


WIN_HEIGHT = 800
WIN_WIDTH = 800
ASPECT_RATIO = WIN_WIDTH / WIN_HEIGHT


class WindowContext:
    def __init__(self, width, height):
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(
            width, height, "VR2021A-Assignment1", None, None
        )
        glfw.make_context_current(self.window)

    def loop(self, func, *args, **kwargs):
        while not glfw.window_should_close(self.window):
            func(*args, **kwargs)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.terminate()


def setupImageTexture(image_path):
    image = Image.open(image_path).convert("RGB")
    tBuffer = image.transpose(Image.FLIP_TOP_BOTTOM).tobytes()
    tHeight = image.height
    tWidth = image.width
    textureGLID = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureGLID)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, tWidth, tHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, tBuffer
    )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    return textureGLID


if __name__ == "__main__":
    context = WindowContext(WIN_WIDTH, WIN_HEIGHT)
    controller = StateController(context.window, (WIN_WIDTH, WIN_HEIGHT))

    glClearColor(0.0, 0, 0.5, 0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    vertex = glGenVertexArrays(1)
    glBindVertexArray(vertex)

    shader = Shader("glsl/vertex.glsl", "glsl/fragment.glsl")

    MVP_ID = glGetUniformLocation(shader.program, "MVP")
    TEXTURE_ID = glGetUniformLocation(shader.program, "textureSampler")
    textureGLID = setupImageTexture("assets/texture.png")

    vertexBuffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
    glBufferData(
        GL_ARRAY_BUFFER,
        len(vertexBufferData) * 4,
        (GLfloat * len(vertexBufferData))(*vertexBufferData),
        GL_STATIC_DRAW,
    )

    uvBuffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, uvBuffer)
    glBufferData(
        GL_ARRAY_BUFFER,
        len(uvBufferData) * 4,
        (GLfloat * len(uvBufferData))(*uvBufferData),
        GL_STATIC_DRAW,
    )

    def update():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader.program)

        controller.update()
        MVP = controller.getMVP()
        glUniformMatrix4fv(MVP_ID, 1, GL_TRUE, MVP)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, textureGLID)
        glUniform1i(TEXTURE_ID, 0)

        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, uvBuffer)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)

        glDrawArrays(GL_TRIANGLES, 0, len(vertexBufferData) // 3)

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

    context.loop(update)
