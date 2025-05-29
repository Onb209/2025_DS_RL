import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np


class OrbitCamera:
    """ Camera control class for orbit-like interaction. """
    def __init__(self, distance=50.0, theta=62, phi=64):
        self.distance = distance
        self.theta = theta
        self.phi = phi
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Center of view
        self.sensitivity = 0.3
        self.zoom_speed = 0.8
        self.pan_speed = 0.02
        self.dragging = False
        self.panning = False
        self.last_mouse = None

    def update(self):
        """ Computes camera position from spherical coordinates. """
        x = self.distance * np.sin(np.radians(self.phi)) * np.cos(np.radians(self.theta))
        y = self.distance * np.cos(np.radians(self.phi))
        z = self.distance * np.sin(np.radians(self.phi)) * np.sin(np.radians(self.theta))
        eye = np.array([x, y, z])  # Camera position

        # Use gluLookAt to set camera position
        gluLookAt(*eye, *self.target, 0, 1, 0)

    def handle_event(self, event):
        """ Handles user input for orbit controls. """
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click drag for rotation
                self.dragging = True
                self.last_mouse = pygame.mouse.get_pos()
            elif event.button == 3:  # Right click drag for panning
                self.panning = True
                self.last_mouse = pygame.mouse.get_pos()
            elif event.button == 4:  # Scroll Up (Zoom In)
                self.distance = max(1.5, self.distance - self.zoom_speed)
            elif event.button == 5:  # Scroll Down (Zoom Out)
                self.distance += self.zoom_speed

        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
            elif event.button == 3:
                self.panning = False

        elif event.type == MOUSEMOTION:
            if self.dragging:
                self._rotate(event)
            elif self.panning:
                self._pan(event)

    def _rotate(self, event):
        """ Rotates the camera using left mouse drag. """
        x, y = pygame.mouse.get_pos()
        dx, dy = x - self.last_mouse[0], y - self.last_mouse[1]
        self.theta += dx * self.sensitivity
        self.phi = min(170, max(10, self.phi - dy * self.sensitivity))
        self.last_mouse = (x, y)

    def _pan(self, event):
        """ Moves the camera target using right mouse drag. """
        x, y = pygame.mouse.get_pos()
        dx, dy = x - self.last_mouse[0], y - self.last_mouse[1]
        self.last_mouse = (x, y)

        # Compute the forward vector (direction from target to camera)
        forward = np.array([
            np.sin(np.radians(self.phi)) * np.cos(np.radians(self.theta)),
            np.cos(np.radians(self.phi)),
            np.sin(np.radians(self.phi)) * np.sin(np.radians(self.theta))
        ])
        # The world up vector remains the same
        up = np.array([0, 1, 0])
        # Compute the right vector: note the order for a proper right-handed system
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        # Recompute the true up vector
        true_up = np.cross(right, forward)
        true_up /= np.linalg.norm(true_up)

        # Adjust the target.
        # Here we subtract the right offset so that dragging right moves the view to the right,
        # and add the up offset (note: dy is added since screen Y increases downward).
        self.target += dx * self.pan_speed * right
        self.target += dy * self.pan_speed * true_up

