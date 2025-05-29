from pbd_simulator.Simulation import PBDSimulation  
from pbd_simulator.Renderer import Renderer
from pbd_simulator.Controls import OrbitCamera  
from pbd_simulator.Objects import Cube, Plane
from pbd_simulator.Constraints import *

def initWorld(world, train_mode=False):
    # ===========================
    # Simulation, Renderer
    # ===========================
    
    world.simulation = PBDSimulation(
        world=world,
        gravity=(0, -9.8, 0),
        time_step=1/30,
        substeps=4
        
    )
    
    world.renderer = Renderer(
        world=world,
        camera=OrbitCamera(
            distance=70.0, 
            theta=80.0, 
            phi=70.0, 
        )
    )
    
    # =============================
    # Create Objects
    # =============================    
    
    # ---------------------------------
    # Set Ground
    # ---------------------------------
    ground = Plane(size=100, color=(0.5, 0.5, 0.5, 0.9))
    world.set_ground(ground)

    # ---------------------------------
    # Draw Cube
    # ---------------------------------
    #    3-------2
    #   /|      /|
    #  7-------6 |      <- Back face (0,1,2,3)
    #  | |     | |      
    #  | 0-----|-1      <- Front face (4,5,6,7)
    #  |/      |/
    #  4-------5       
    # ---------------------------------
    
    cube1 = Cube(width=1.0, height=2.0, depth=1.0, positions=[0, 4.5, 0], rotation=[0, 0, 0], color=(1.0, 0.0, 0.0, 1.0))
    world.add_object(cube1)
    cube2 = Cube(width=1.0, height=2.0, depth=1.0, positions=[0, 2.5, 0], rotation=[0, 0, 0], color=(0.0, 1.0, 0.0, 1.0))
    world.add_object(cube2)
    cube3 = Cube(width=2.0, height=1.0, depth=1.0, positions=[0.5, 1.0, 0], rotation=[0, 0, 0], color=(0.2, 0.5, 0.84, 1.0))    
    world.add_object(cube3)
    
    if not train_mode:
        cube4 = Cube(width=2.0, height=10.0, depth=30.0, positions=[19.0, 5+122.5, 0], rotation=[0, 0, 0], color=(0.3, 0.3, 0.3, 1.0))    
        cube4.restitution = 1.0
        world.add_object(cube4)
    

    # =============================
    # Generate Constraints
    # =============================
    
    attach_comp = 0.00001
    rigid_comp = 0.00000001
    hinge_comp = 0.000000
    world.simulation.collision_compliance = 0.00000001

    # ---------------------------------
    # a. Attach Constraint
    # ---------------------------------
    world.simulation.add_constraint(AttachmentConstraint(cube1, 2, cube1.curr_pos[2], compliance=attach_comp))


    # ---------------------------------
    # b. Rigid Constraint
    # ---------------------------------
    for cube in world.get_objects():
        if cube is not None:
            for edge in cube.edges:
                rest_length = np.linalg.norm(cube.vertices[edge[0]] - cube.vertices[edge[1]])
                world.simulation.add_constraint(DistanceConstraint(cube, edge[0], cube, edge[1], rest_length, rigid_comp))
                
    
    # ---------------------------------
    # c. Hinge Constraint
    # ---------------------------------
    world.simulation.add_constraint(DistanceConstraint(cube1, 5, cube2, 6, rest_length=0.0, compliance=hinge_comp))
    world.simulation.add_constraint(DistanceConstraint(cube1, 1, cube2, 2, rest_length=0.0, compliance=hinge_comp))
    world.simulation.add_constraint(DistanceConstraint(cube2, 4, cube3, 7, rest_length=0.0, compliance=hinge_comp))
    world.simulation.add_constraint(DistanceConstraint(cube2, 0, cube3, 3, rest_length=0.0, compliance=hinge_comp))



    # -------------------------------
    # d.  Joint limit constraint
    # #--------------------------------
    # world.simulation.add_constraint(MinDistanceConstraint(cube1, 6, cube2, 5, min_length=3.0, compliance=hinge_comp))
    # world.simulation.add_constraint(MinDistanceConstraint(cube1, 2, cube2, 1, min_length=3.0, compliance=hinge_comp))
    
    # world.simulation.add_constraint(MinDistanceConstraint(cube2, 1, cube3, 0, min_length=1.0, compliance=hinge_comp))
    # world.simulation.add_constraint(MinDistanceConstraint(cube2, 5, cube3, 4, min_length=1.0, compliance=hinge_comp))

    # world.simulation.add_constraint(MinDistanceConstraint(cube2, 7, cube3, 4, min_length=1.0, compliance=hinge_comp))
    # world.simulation.add_constraint(MinDistanceConstraint(cube2, 3, cube3, 0, min_length=1.0, compliance=hinge_comp))

    return world




class World:
    def __init__(self,):
        # Objects
        self.objects = []
        self.ground = None
        
        # Simulation
        self.simulation = None
        self.sim_time = 0.0
        self.playing = True

        # Rendering
        self.renderer = None
        self.running = True
    
    def set_ground(self, ground):
        self.ground = ground    

    def add_object(self, obj):
        if obj is not None:
            self.objects.append(obj)
    
    def get_objects(self):
        return self.objects    
    
    def reset(self):
        self.simulation.reset()
        self.sim_time = 0.0
    
    def step(self):
        if self.playing:
            self.simulation.step()
            self.sim_time += self.simulation.time_step

    def render(self, ):
        self.renderer.render()

    def handle_events(self):
        self.renderer.handle_events()
