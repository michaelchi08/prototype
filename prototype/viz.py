#!/usr/bin/env python3
import math

import numpy as np

import pygame
from pygame.locals import OPENGL
from pygame.locals import DOUBLEBUF
from pygame.locals import RESIZABLE
from pygame.locals import HWSURFACE

from OpenGL.GL import glBegin
from OpenGL.GL import glEnd
from OpenGL.GL import glColor3f
from OpenGL.GL import glClear
from OpenGL.GL import glClearColor
from OpenGL.GL import glVertex3f
from OpenGL.GL import glTranslatef
from OpenGL.GL import glRotatef
from OpenGL.GL import glViewport
from OpenGL.GL import glMatrixMode
from OpenGL.GL import glLoadIdentity
from OpenGL.GL import GL_LINES
from OpenGL.GL import GL_PROJECTION
from OpenGL.GL import GL_MODELVIEW
from OpenGL.GL import GL_COLOR_BUFFER_BIT
from OpenGL.GL import GL_DEPTH_BUFFER_BIT

from OpenGL.GLU import gluPerspective


class VizNavMode(object):
    PAN = 1
    ROTATE = 2


class Viz(object):
    def __init__(self):
        self.navigation_mode = VizNavMode.PAN
        self.window_width = 800
        self.window_height = 600
        self.screen = None

        self.world_background_color = (0.1, 0.1, 0.1)

        self.camera_position = [0, 0, 0]

        self.mouse_drag = False
        self.rotate_mode = False

    def setup(self):
        self.camera_position = [0.0, -2.0, -10.0]

        pygame.init()
        display = (self.window_width, self.window_height)
        pygame.display.set_mode(
            display,
            DOUBLEBUF | RESIZABLE | HWSURFACE | OPENGL
        )
        gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
        glTranslatef(
            self.camera_position[0],
            self.camera_position[1],
            self.camera_position[2]
        )

    def resize(self, width, height):
        self.window_width = width
        self.window_height = height

        self.screen = pygame.display.set_mode(
            (width, height),
            DOUBLEBUF | RESIZABLE | HWSURFACE | OPENGL
        )
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(width)/height, 0.1, 100.0)
        glTranslatef(
            self.camera_position[0],
            self.camera_position[1],
            self.camera_position[2]
        )
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def event_handler(self):
        x = 0.0
        y = 0.0
        z = 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            elif event.type == pygame.VIDEORESIZE:
                self.resize(*event.dict['size'])

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_drag = True

                # scroll up
                if event.dict['button'] == 4:
                    z += 0.1

                # scroll down
                elif event.dict['button'] == 5:
                    z -= 0.1

            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_drag = False

            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_drag:
                    p = pygame.mouse.get_rel()
                    if math.fabs(p[0]) < 20.0 and math.fabs(p[1]) < 20.0:
                        x = p[0] * 0.1
                        y = p[1] * -0.1

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x -= 0.1
                elif event.key == pygame.K_RIGHT:
                    x += 0.1
                elif event.key == pygame.K_UP:
                    y += 0.1
                elif event.key == pygame.K_DOWN:
                    y -= 0.1
                elif event.key == pygame.K_PAGEUP:
                    z += 0.1
                elif event.key == pygame.K_PAGEDOWN:
                    z -= 0.1
                elif event.key == pygame.K_r:
                    if self.rotate_mode is False:
                        self.rotate_mode = True
                        print("rotate_mode: ON")
                    else:
                        self.rotate_mode = False
                        print("rotate_mode: OFF")
                elif event.key == pygame.K_q:
                    pygame.quit()
                    quit()

            if self.rotate_mode:
                glRotatef(10.0, 0, x, 0)
            else:
                glTranslatef(x, y, z)

    def clear_scene(self):
        r, g, b = self.world_background_color
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(r, g, b, 1.0)

    def draw_ground(self):
        glColor3f(0.7, 0.7, 0.7)
        glBegin(GL_LINES)
        for i in np.arange(-2.5, 2.75, 0.25):
            glVertex3f(i, 0, 2.5)
            glVertex3f(i, 0, -2.5)
            glVertex3f(2.5, 0, i)
            glVertex3f(-2.5, 0, i)
        glEnd()

    def run(self):
        self.setup()

        while True:
            self.event_handler()
            self.clear_scene()
            self.draw_ground()

            pygame.display.flip()
            pygame.time.wait(10)


if __name__ == "__main__":
    viz = Viz()
    viz.setup()
    viz.run()
