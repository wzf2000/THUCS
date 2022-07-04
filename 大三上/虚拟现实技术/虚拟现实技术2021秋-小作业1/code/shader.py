"""
Code adapted from https://github.com/jcteng/python-opengl-tutorial/blob/master/utils/shaderLoader.py
"""

from OpenGL import GL as gl


def checkOpenGLError():
    err = gl.glGetError()
    if err != gl.GL_NO_ERROR:
        print("GLERROR:", gl.gluErrorString(err))


class Shader(object):
    def __init__(self, vertexShaderPath, fragmentShaderPath):
        with open(vertexShaderPath, "rb") as f:
            vertexShaderSrc = f.read()
        with open(fragmentShaderPath, "rb") as f:
            fragmentShaderSrc = f.read()
        self.initShader(vertexShaderSrc, fragmentShaderSrc)

    def initShader(self, vertexShaderSrc, fragmentShaderSrc):
        self.program = gl.glCreateProgram()
        checkOpenGLError()

        self.vs = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(self.vs, vertexShaderSrc)
        gl.glCompileShader(self.vs)
        if gl.GL_TRUE != gl.glGetShaderiv(self.vs, gl.GL_COMPILE_STATUS):
            err = gl.glGetShaderInfoLog(self.vs)
            raise Exception(err)
        gl.glAttachShader(self.program, self.vs)
        checkOpenGLError()

        self.fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(self.fs, fragmentShaderSrc)
        gl.glCompileShader(self.fs)
        if gl.GL_TRUE != gl.glGetShaderiv(self.fs, gl.GL_COMPILE_STATUS):
            err = gl.glGetShaderInfoLog(self.fs)
            raise Exception(err)
        gl.glAttachShader(self.program, self.fs)
        checkOpenGLError()

        gl.glLinkProgram(self.program)
        if gl.GL_TRUE != gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS):
            err = gl.glGetShaderInfoLog(self.vs)
            raise Exception(err)
        checkOpenGLError()
