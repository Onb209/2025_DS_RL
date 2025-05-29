from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_cube(width, height, depth):
    """ Generates cube vertices and faces. """
    vertices = np.array([
        [-width / 2, -height / 2, -depth / 2],  # Back Bottom Left
        [ width / 2, -height / 2, -depth / 2],  # Back Bottom Right
        [ width / 2,  height / 2, -depth / 2],  # Back Top Right
        [-width / 2,  height / 2, -depth / 2],  # Back Top Left
        [-width / 2, -height / 2,  depth / 2],  # Front Bottom Left
        [ width / 2, -height / 2,  depth / 2],  # Front Bottom Right
        [ width / 2,  height / 2,  depth / 2],  # Front Top Right
        [-width / 2,  height / 2,  depth / 2]   # Front Top Left
    ], dtype=np.float32)

    faces = np.array([
        [4, 5, 6], [4, 6, 7],  # Front (+Z)
        [1, 0, 3], [1, 3, 2],  # Back (-Z)
        [5, 1, 2], [5, 2, 6],  # Right (+X)
        [0, 4, 7], [0, 7, 3],  # Left (-X)
        [3, 7, 6], [3, 6, 2],  # Top (+Y)
        [0, 1, 5], [0, 5, 4]   # Bottom (-Y)
    ], dtype=np.uint32)

    normals = np.array([
        [0, 0, 1], [0, 0, 1],  # Front
        [0, 0, -1],[0, 0, -1], # Back
        [1, 0, 0], [1, 0, 0],  # Right
        [-1, 0, 0],[-1, 0, 0], # Left
        [0, 1, 0], [0, 1, 0],  # Top
        [0, -1, 0], [0, -1, 0]  # Bottom
    ], dtype=np.float32)
    
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0], # Back face
        [4, 5], [5, 6], [6, 7], [7, 4], # Front face
        [0, 4], [1, 5], [2, 6], [3, 7], # Connect front and back faces
        # -----------------------------------------------------
        # TODO (3) : Need to Add Additional Edges to make rigid
        #    3-------2
        #   /|      /|
        #  7-------6 |      <- Back face (0,1,2,3)
        #  | |     | |      
        #  | 0-----|-1      <- Front face (4,5,6,7)
        #  |/      |/
        #  4-------5   
        # -----------------------------------------------------
        [0, 2], [1, 3], [4, 6], [5, 7], 
        [1, 6], [2, 5], [0, 7], [3, 4], 
        [0, 5], [1, 4], [2, 7], [3, 6],         
    ], dtype=np.uint32)

    return vertices, faces, edges, normals


class Cube:
    def __init__(self, width=1.0, height=1.0, depth=1.0, positions=[0, 0, 0], rotation=[0, 0, 0], color=(0, 1, 0), wireframe=False):
        # Mesh 
        self.color = color
        self.vertices, self.faces, self.edges, self.normals = generate_cube(width, height, depth)
        self.wireframe = wireframe  
        
        self.init_rot = R.from_euler('XYZ', rotation, degrees=True)
        self.init_pos = np.array(positions, dtype=np.float32) + self.init_rot.apply(self.vertices)
        
        
        # Physics
        self.curr_pos = self.init_pos.copy()
        self.prev_pos = self.curr_pos.copy()
        self.vel = np.zeros_like(self.curr_pos)
        
        self.inv_mass = 1.0
        self.restitution = 0.1
        self.friction = 0.1
        
    
    def reset(self):
        self.curr_pos = self.init_pos.copy()
        self.prev_pos = self.curr_pos.copy()
        self.vel = np.zeros_like(self.curr_pos)
        
    def draw(self):
        glColor4f(*self.color)
        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            for i , edge in enumerate(self.edges):
                glBegin(GL_LINE_LOOP)
                for vertex in edge:
                    glVertex3fv(self.curr_pos[vertex])
                glEnd()
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glBegin(GL_TRIANGLES)
            self.recompute_normals()
            for i, face in enumerate(self.faces):
                glNormal3fv(self.normals[i])  # Use correct face normal
                for vertex in face:
                    glVertex3fv(self.curr_pos[vertex])
            glEnd()
            self.draw_shadow()
    
    # =================================
    # Below functions are for Rendering
    # =================================
    def recompute_normals(self):
        new_normals = []
        for face in self.faces:
            v0, v1, v2 = self.curr_pos[face[0]], self.curr_pos[face[1]], self.curr_pos[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            normal /= np.linalg.norm(normal)
            new_normals.append(normal)

        self.normals = np.array(new_normals, dtype=np.float32)
    
    def draw_shadow(self, light_dir=(10, 15, 10)):
        if np.max(self.curr_pos[:, 1]) < 0:
            return
        
        glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glColor4f(0.2, 0.2, 0.2, 0.9)  

        # Define ground plane: y = 0 → (A, B, C, D) = (0, 1, 0, 0)
        # Ax + By + Cz + D = 0
        A, B, C, D = 0, 1, 0, 0 

        # Convert light direction into a homogeneous coordinate
        Lx, Ly, Lz, Lw = light_dir[0], light_dir[1], light_dir[2], 0 

        # Compute shadow projection matrix
        dot = A * Lx + B * Ly + C * Lz + D * Lw
        shadow_mat = np.array([
            [dot - Lx * A,    -Lx * B,    -Lx * C,    -Lx * D],
            [   -Ly * A,  dot - Ly * B,    -Ly * C,    -Ly * D],
            [   -Lz * A,    -Lz * B,  dot - Lz * C,    -Lz * D],
            [   -Lw * A,    -Lw * B,    -Lw * C,  dot - Lw * D]
        ], dtype=np.float32).T 

        glPushMatrix()
        glTranslatef(0, 0.01, 0)  # Slightly above the ground
        glMultMatrixf(shadow_mat) 
        
        # Draw the shadow
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            for vertex in face:
                glVertex3fv(self.curr_pos[vertex])
        glEnd()

        glPopMatrix()

        # Restore OpenGL state
        glEnable(GL_LIGHTING)  
        glPopAttrib()
    

class Plane:
    def __init__(self, size=10, color=(0.3, 0.3, 0.3, 1.0)):
        self.normal = np.array([0, 1, 0], dtype=np.float32)
        self.origin = np.array([0, 0, 0], dtype=np.float32)
        
        self.size = size
        self.color = color

    def draw(self):
        glEnable(GL_BLEND)
        glColor4f(*self.color)
        glBegin(GL_QUADS)
        glNormal3fv(self.normal)
        glVertex3f(-self.size, 0, -self.size)
        glVertex3f(self.size, 0, -self.size)
        glVertex3f(self.size, 0, self.size)
        glVertex3f(-self.size, 0, self.size)
        glEnd()
        
        self.draw_grid()
        
        glDisable(GL_BLEND)
        
        self.draw_axes()
    
    def draw_grid(self, step=10.0):
        
        glColor4f(1.0, 1.0, 1.0, 0.9)
        glPushMatrix()
        glTranslatef(0, 0.01, 0)
        glBegin(GL_LINES)
        for i in np.arange(-self.size, self.size + step, step):
            glVertex3f(i, 0, -self.size)
            glVertex3f(i, 0, self.size)

            glVertex3f(-self.size, 0, i)
            glVertex3f(self.size, 0, i)
        glEnd()
        glPopMatrix()
    
    def draw_axes(self, length=5.0):
        
        glLineWidth(2.0)
        glPushMatrix()
        glTranslatef(0, 0.02, 0)

        glBegin(GL_LINES)        
        # X-Axis (Red)
        glColor3f(1, 0, 0)  # Red
        glVertex3f(0, 0, 0)
        glVertex3f(length, 0, 0)

        # Y-Axis (Green)
        glColor3f(0, 1, 0)  # Green
        glVertex3f(0, 0, 0)
        glVertex3f(0, length, 0)

        # Z-Axis (Blue)
        glColor3f(0, 0, 1)  # Blue
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, length)
        glEnd()
        glPopMatrix()
        glLineWidth(1.0)
        
        