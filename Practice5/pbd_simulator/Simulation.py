import numpy as np
from pbd_simulator.Constraints import *

class PBDSimulation:
    def __init__(self, 
                 world, 
                 gravity=(0, -9.8, 0), 
                 time_step=0.0333,
                 substeps=2):

        self.world = world
        self.constraints = []
        
        # Physics Parameters
        self.gravity = np.array(gravity, dtype=np.float32)
        self.time_step = time_step
        self.substeps = substeps
        self.h = self.time_step / self.substeps

        # Constraints parameters
        self.collision_compliance = 0.00000001
        self.attach_constraints = []


    ##### ========================= Added for RL ================================== #####

    def step_(self, action):
        collisions = self.check_collisions()
        for _ in range(self.substeps):
            contacts = self.generate_contacts(collisions)
            self.apply_action(action)
            self.integrate()
            self.solve_positions(contacts)
            self.update_velocities()
            self.solve_velocities(contacts)

    def apply_ext_force(self, obj, idx, force):
        acc = force*obj.inv_mass
        obj.vel[idx] += acc*self.h

    def apply_action(self, action):
        """
        # --------------------------------------------------
        # Implement torque application
        # --------------------------------------------------

        Apply torque to the hinge joints of the objects in the simulation.
        Parameters:
        action (np.ndarray): An array of torques to be applied to the hinge joints. 
                             The length of the array should match the number of degrees of freedom (DOF).
        Description:
        This method applies the specified torques to the hinge joints between the objects in the simulation.
        In this scene, our foot model consists of three objects connected by two hinge joints:
        - Joint 0: Connects cube1 and cube2
        - Joint 1: Connects cube2 and cube3
        For each torque in the action array, the method:
        1. Identifies the parent and child objects connected by the hinge joint.
        2. Computes the rotation axis of the hinge joint. The rotation axis is the edge of the parent object that connects the two hinge vertices.
        """
        num_dofs = action.shape[0]
        for dof_index in range(num_dofs):
            torque = action[dof_index]
            if dof_index == 0:
                parent_object = self.world.get_objects()[0]
                child_object = self.world.get_objects()[1]

                parent_joint = [1, 5]
                child_joint = [2, 6]

                parent_joint_opp = [0, 4]
                child_joint_opp = [3, 7]

            elif dof_index == 1:
                parent_object = self.world.get_objects()[1]
                child_object = self.world.get_objects()[2]

                parent_joint = [0, 4]
                child_joint = [3, 7]

                parent_joint_opp = [1, 5]
                child_joint_opp = [2, 6]

            rot_axis = parent_object.curr_pos[parent_joint[1]] - parent_object.curr_pos[parent_joint[0]]
            rot_axis = rot_axis / np.linalg.norm(rot_axis)

            """
            parent: the vertex of the parent object connected to the hinge joint
            child: the vertex of the child object connected to the hinge joint
            parent_opp: the opposite vertex of the parent object connected to the hinge joint
            child_opp: the opposite vertex of the child object connected to the hinge joint

            The force will be applied to the vertices of parent , parent_opp and child, child_opp
            Force applied on parent(child) and parrent_opp(child_opp) is in opposite direction.

            Since the hinge joint axis have 2 vertices, the for loop runs twice to apply the force on both vertices.
            """
            for i in range(2):
                parent = parent_object.curr_pos[parent_joint[i]]
                child = child_object.curr_pos[child_joint[i]]

                parent_opp = parent_object.curr_pos[parent_joint_opp[i]]
                child_opp = child_object.curr_pos[child_joint_opp[i]]

                parent_length = np.linalg.norm(parent - parent_opp)
                child_length = np.linalg.norm(child - child_opp)

                force_norm_parent = -torque / parent_length
                force_norm_child = torque / child_length

                parent_force_direction = np.cross(rot_axis,parent_opp - parent)
                parent_force_direction = parent_force_direction / np.linalg.norm(parent_force_direction)

                parent_force = force_norm_parent * parent_force_direction
                self.apply_ext_force(parent_object, parent_joint[i], -parent_force)
                self.apply_ext_force(parent_object, parent_joint_opp[i], parent_force)

                child_force_direction = np.cross(rot_axis,child_opp - child)
                child_force_direction = child_force_direction / np.linalg.norm(child_force_direction)

                child_force = force_norm_child * child_force_direction

                self.apply_ext_force(child_object, child_joint[i], -child_force)
                self.apply_ext_force(child_object, child_joint_opp[i], child_force)

    ##### ===================================================================== #####


    def add_constraint(self, constraint):
        if isinstance(constraint, Constraint):
            self.constraints.append(constraint)
        
        if isinstance(constraint, AttachmentConstraint):
            self.attach_constraints.append(constraint)

    def detach_all(self):
        for attach_constraint in self.attach_constraints:
            attach_constraint.anchor = None
    
    def reset(self,):
        for obj in self.world.get_objects():
            obj.curr_pos = obj.init_pos.copy()
            obj.prev_pos = obj.curr_pos.copy()
            obj.vel = np.zeros_like(obj.curr_pos)

        for constraint in self.constraints:
            constraint.reset()
    
    def step(self): 
        collisions = self.check_collisions()
        
        for _ in range(self.substeps):
            # --------------------------------------------------
            # Generate Contacts
            # --------------------------------------------------
            contacts = self.generate_contacts(collisions)

            # --------------------------------------------------
            # Integrate, Solve Constraints(Positions) and Update Velocities
            # - Place self.integrate(), self.solve_positions() and self.update_velocities() in right order
            # --------------------------------------------------
            self.integrate()
            self.solve_positions(contacts)
            self.update_velocities()
            
            # --------------------------------------------------
            # Solve Velocities
            # --------------------------------------------------
            self.solve_velocities(contacts)
            
                    
    def check_collisions(self):
        # Broad Phase Collision Detection for large number of objects.
        return []
    
    def generate_contacts(self, collisions):
        contacts = []
        
        # Dynamic - Dynamic Collisions
        for collision in collisions:
            pass
        
        # --------------------------------------------------
        # Generate Ground Collision Constraints
        # - Find the vertices in contact with the ground
        # - Generate Ground Collision Constraints
        # --------------------------------------------------
        for obj in self.world.get_objects():
            contact_vertices = np.where(obj.curr_pos[:, 1] < 0)[0]
            for id in contact_vertices:
                contacts.append(GroundCollisionConstraint(obj, id, compliance=self.collision_compliance))
            
        return contacts
    
    def integrate(self):
        # --------------------------------------------------
        # Integrate
        # - Update previous position, velocity and current position
        # --------------------------------------------------
        for obj in self.world.get_objects():
            obj.prev_pos = obj.curr_pos.copy()
            obj.vel += self.gravity * self.h
            obj.curr_pos += obj.vel * self.h

    def solve_positions(self, contacts=None):        
        for constraint in self.constraints:
            constraint.solve(self.h)
        
        if contacts is None:
            return
        
        for contact in contacts:
            contact.solve(self.h)
            
    def update_velocities(self):
        # --------------------------------------------------
        # Update Velocities
        # - Update velocities of objects based on the current and previous positions
        # --------------------------------------------------
        for obj in self.world.get_objects():
            obj.vel = (obj.curr_pos - obj.prev_pos) / self.h        
        
    def solve_velocities(self, contacts):
        for contact in contacts:
            contact.solve_velocity()
            
        
