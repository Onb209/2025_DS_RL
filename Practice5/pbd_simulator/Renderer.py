import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class Renderer:
    def __init__(self, world, camera, width=800, height=600,):
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("PBD")

        # OpenGL Setup
        glEnable(GL_DEPTH_TEST)
        glClearColor(135/255, 206/255, 235/255, 1.0)
        
        # Set Up 3D Projection
        glMatrixMode(GL_PROJECTION)
        gluPerspective(30, width / height, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

        # Camera
        self.camera = camera

        # Lighting
        self.set_lighting()

        # UI Panel Setup
        self.font = pygame.font.Font(None, 24)  # Default Pygame font

        # World to Render
        self.world = world  
        

        self.tracking = False
        self.user_torque = np.array([0.0, 0.0])
        self.acc_reward =  0.0

    def handle_events(self):
        for event in pygame.event.get():
            self.camera.handle_event(event)  

            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    self.world.playing = not self.world.playing
            
                if event.key == K_ESCAPE:
                    self.world.running = False
                
                if event.key == K_r:
                    self.world.playing = False
                    self.user_torque *= 0.0
                    self.acc_reward *= 0.0
                    self.world.reset()
                        
                if event.key == K_e:
                    for obj in self.world.get_objects():
                        obj.wireframe = not obj.wireframe
                        
                if event.key == K_d and pygame.key.get_mods() & KMOD_CTRL:
                    self.world.simulation.detach_all()
                    
                if event.key == K_q:
                   self.user_torque -= 20.0
                   print(self.user_torque)
                if event.key == K_w:
                   self.user_torque += 20.0
                   print(self.user_torque)

                if event.key == K_f:
                    self.tracking = not self.tracking
                

            if event.type == QUIT:
                self.world.running = False

    def render(self,):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        if self.tracking:
            cube1_center = self.world.get_objects()[0].curr_pos.mean(axis=-2)
            self.camera.target = cube1_center
            self.camera.target[1] = 3.0

        self.camera.update()

        # Draw Objects
        for obj in self.world.get_objects():
            obj.draw()
        
        # Draw Ground
        self.world.ground.draw()

        self.draw_ui()
        pygame.display.flip()

    def set_lighting(self):
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        
        light_position = [15, 20, 15, 0]  
        light_ambient = [0.3, 0.3, 0.3, 1.0]
        light_diffuse = [0.8, 0.8, 0.8, 1.0]
        light_specular = [0.1, 0.1, 0.1, 1.0]

        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

        glEnable(GL_COLOR_MATERIAL) 
        # glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE)

    def draw_ui(self,):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.screen.get_width(), 0, self.screen.get_height())
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        keymap_texts = [
            "Controls:",
            "---------------------------------------------------",
            "[Left Mouse] Rotate Camera",
            "[Right Mouse] Pan Camera",
            "[Scroll] Zoom In/Out",
            "[E] Visualize Edges",
            "[CTRL + D] Detach All",
            "[Q/W] Apply Torque",
            "[F] Tracking Mode",
            "---------------------------------------------------",
            "[SPACE] Start / Stop Simulation",
            "[R] Reset Simulation",
            "[ESC] Quit"
        ]

        y_offset = 20
        for text in keymap_texts:
            self.render_text(text, 20, self.screen.get_height() - y_offset)
            y_offset += 25  # Move down for the next line
        
        sim_time = self.world.sim_time
        if sim_time is not None:
            self.render_text(f"Simulation Time: {format_time(sim_time)}", self.screen.get_width() - 250, self.screen.get_height() - 20)

        self.render_text(f"Total reward: {self.acc_reward:.2f}", self.screen.get_width() - 250, self.screen.get_height() - 40)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def render_text(self, text, x, y):
        font_surface = self.font.render(text, True, (0.0*255, 0.0*255, 0.0*255))  # White text
        text_data = pygame.image.tostring(font_surface, "RGBA", True)
        text_width, text_height = font_surface.get_size()

        # Generate texture
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_width, text_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Reset OpenGL color to white
        glPushAttrib(GL_CURRENT_BIT)  # Save the current color state
        glColor3f(1.0, 1.0, 1.0)  # Set color to white

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)

        # Draw a quad with the text texture
        glBindTexture(GL_TEXTURE_2D, texture)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1)
        glVertex2f(x, y)
        glTexCoord2f(1, 1)
        glVertex2f(x + text_width, y)
        glTexCoord2f(1, 0)
        glVertex2f(x + text_width, y - text_height)
        glTexCoord2f(0, 0)
        glVertex2f(x, y - text_height)
        glEnd()

        # Restore previous color state
        glPopAttrib()

        # Clean up
        glDeleteTextures([texture])
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)

        
def format_time(seconds):
    minutes = int(seconds) // 60
    sec = int(seconds) % 60
    ms = int((seconds - int(seconds)) * 1000)  # Extract milliseconds
    return f"{minutes:02}:{sec:02}:{ms:03}"
