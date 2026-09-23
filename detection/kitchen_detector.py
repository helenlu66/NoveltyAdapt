import numpy as np
from detection.detector import Detector
from typing import *
from scipy.spatial.transform import Rotation as R
from mimicgen.envs.robosuite.coffee import Coffee_Drawer_Novelty

class KitchenDetector(Detector):
    def __init__(self, env, return_int=False):
        super().__init__(env, return_int)
        # predicate mappings
        self.predicates = {
            'pickupable': {
                'func':self.pickupable,
                'params':['tabletop_object']
                }, 
            'directly-on-table': {
                'func':self.directly_on_table,
                'params':['tabletop_object', 'table']
            }, 
            'exclusively-occupying-gripper': {
                'func':self.exclusively_occupying_gripper,
                'params':['tabletop_object', 'gripper']
            }, 
            'in': {
                'func':self.inside,
                'params':['tabletop_object', 'pot']
            }, 
            'covered': {
                'func':self.covered,
                'params':['pot']
            }, 
            'on': {
                'func':self.on,
                'params':['on_off_object']
            },
            'ontop': {
                'func':self.ontop,
                'params':['tabletop_object', 'tabletop_object']
            },
            'free': {
                'func':self.free,
                'params':['gripper']
            }, 
        }
        # mapping from relevant object types to objects in the environment
        self.object_types = {
            'tabletop_object':['lid', 'bread', 'pot', 'switch', 'stove'], 
            'on_off_object':['switch', 'stove'],
            'gripper':['gripper'], 
            'lid':['lid'], 
            'bread':['bread'], 
            'pot':['pot'], 
            'switch':['switch'], 
            'stove':['stove'], 
            'table':['table']
        }

        # this is a hack to map the grounded object to their pddl format. This is needed because the grounded object is in the format e.g. 'mug' while the pddl object is in the format 'mug1'
        self.grounded_object_to_pddl_object = {'table':'table1', 'gripper':'gripper1', 'lid':'lid1', 'bread':'bread1', 'pot':'pot1', 'switch':'switch1', 'stove':'stove1'}

        self.grounded_tabletop_object_to_kitchen_class_object = {'lid':self.env.lid, 'bread':self.env.bread_ingredient, 'pot':self.env.pot_object, 'stove':self.env.stove_object_1, 'switch':self.env.button_object_1, 'gripper':self.env.robots[0].gripper}

    
    def pickupable(self, tabletop_obj:str) -> bool:
        """Returns True if the object can be picked up.
        Args:
            tabletop_obj (str): the tabletop object
            
        Returns:
            bool: True if the object is small enough for the gripper to pick up
        """
        # hardcoding non-fixtures to be pickupable
        assert self._is_type(tabletop_obj, 'tabletop_object')
        if self._is_type(tabletop_obj, 'on_off_object'):
            return False
        return True


    def directly_on_table(self, tabletop_obj:str, table:str) -> bool:
        """Returns True if the object is directly on the table.

        Args:
            tabletop_obj (str): the tabletop object
            table (str): the name of the table object

        Returns:
            bool: True if the object is directly on the table
        """
        assert self._is_type(tabletop_obj, 'tabletop_object') and self._is_type(table, 'table')
        # the two on-off-objects are always directly on the table
        if self._is_type(tabletop_obj, 'on_off_object'):
            return True
        obj = self.grounded_tabletop_object_to_kitchen_class_object[tabletop_obj]
        return self.env.check_contact(obj, 'table_collision')


    def exclusively_occupying_gripper(self, tabletop_obj:str, gripper:str) -> bool:
        """Returns True if the object is exclusively occupying the gripper.

        Args:
            tabletop_obj (str): the tabletop object
            gripper (str): the gripper object

        Returns:
            bool: True if the object is exclusively occupying the gripper
        """
        assert self._is_type(tabletop_obj, 'tabletop_object') and self._is_type(gripper, 'gripper')
        gripper = self.grounded_tabletop_object_to_kitchen_class_object[gripper]
        tabletop_obj_contact_geoms = self.grounded_tabletop_object_to_kitchen_class_object[tabletop_obj].contact_geoms
        return self.env._check_grasp(gripper, tabletop_obj_contact_geoms)

    def on(self, on_off_object:str) -> bool:
        """Returns True if the object is turned on.

        Args:
            on_off_object (str): the on-off object
            table (str): the table object

        Returns:
            bool: True if the object is on the table
        """
        assert self._is_type(on_off_object, 'on_off_object')
        if self._is_type(on_off_object, 'stove'):
            return self.env.buttons_on[1] # the stove is on if the button is on
        elif self._is_type(on_off_object, 'switch'):
            return self.env.buttons_on[1]
        else:
            raise ValueError(f"Invalid on-off object: {on_off_object}")
    
    def ontop(self, tabletop_obj1:str, tabletop_obj2:str) -> bool:
        """Returns True if the first object is on top of the second object.

        Args:
            tabletop_obj1 (str): the first tabletop object
            tabletop_obj2 (str): the second tabletop object

        Returns:
            bool: True if the first object is on top of the second object
        """
        assert self._is_type(tabletop_obj1, 'tabletop_object') and self._is_type(tabletop_obj2, 'tabletop_object')
        obj1 = self.grounded_tabletop_object_to_kitchen_class_object[tabletop_obj1]
        if self._is_type(tabletop_obj2, 'pot'):
            if self.env.check_contact(obj1, [
                'PotObject_body_1', 
                'PotObject_body_2', 
                'PotObject_body_3', 
                'PotObject_body_4', 
                #'PotObject_handle_left_1', 
                #'PotObject_handle_right_1'
                ]): # approximate the ontop relation with contact with the sides of the pot
                return True
            else:
                # if the first object is lid
                if self._is_type(tabletop_obj1, 'lid'):
                    lid = self.grounded_tabletop_object_to_kitchen_class_object['lid']
                    pot = self.grounded_tabletop_object_to_kitchen_class_object['pot']
                    lid_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(lid.root_body)]
                    pot_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(pot.root_body)]
                    dist = np.linalg.norm(lid_pos - pot_pos)
                    return lid_pos[2] > pot_pos[2] and dist < 0.10
                return False
        elif self._is_type(tabletop_obj2, 'stove'):
            # check if obj1 is close to the stove
            stove_pos = self.env.sim.data.body_xpos[self.env.object_body_ids['stove_1']]
            obj1_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(obj1.root_body)]
            dist = np.linalg.norm(stove_pos - obj1_pos)
            return dist < 0.080
        else:
            # obj2 = self.grounded_tabletop_object_to_kitchen_class_object[tabletop_obj2]
            # # check if obj1 is only in contact with obj2 and not any other tabletop object
            # contact_with_obj2 = self.env.check_contact(obj1, obj2)
            # obj1_higher = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(obj1.root_body)][2] > self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(obj2.root_body)][2]
            # obj1_not_on_table = self.directly_on_table(tabletop_obj1, 'table')
            # return contact_with_obj2 and obj1_higher and obj1_not_on_table
            return False
    
    def inside(self, tabletop_obj:str, pot:str) -> bool:
        """Returns True if the object is inside the pot.

        Args:
            tabletop_obj (str): the tabletop object
            pot (str): the container object

        Returns:
            bool: True if the object is inside the pot
        """
        assert self._is_type(tabletop_obj, 'tabletop_object') and self._is_type(pot, 'pot')
        obj = self.grounded_tabletop_object_to_kitchen_class_object[tabletop_obj]
        return self.env.check_contact(obj, 'PotObject_body_0')


    def covered(self, pot:str) -> bool:
        """Returns True if the pot is covered.

        Args:
            container (_type_): the container object

        Returns:
            bool: True if the container is open
        """
        assert self._is_type(pot, 'pot')
        # pot is covered if there the lid is on top of the pot
        lid = self.grounded_tabletop_object_to_kitchen_class_object['lid']
        pot = self.grounded_tabletop_object_to_kitchen_class_object['pot']
        lid_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(lid.root_body)]
        pot_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(pot.root_body)]
        dist = np.linalg.norm(lid_pos - pot_pos)
        # check if the lid is ontop of the pot. If it is, then the pot is covered
        lid_ontop = self.ontop('lid', 'pot')
        if lid_ontop:
            return True
        return dist < 0.11


    def free(self, gripper) -> bool:
        """Returns True if the gripper is free.

        Args:
            gripper (_type_): the gripper object

        Returns:
            bool: True if the gripper is free
        """
        for obj in self.object_types['tabletop_object']:
            if self.exclusively_occupying_gripper(obj, gripper):
                return False
        return True
    
    
    def verify_env(self, env) -> bool:
        """Verify that the environment is the correct environment class.

        Args:
            env (MujocoEnv): the environment
        Returns:
            bool: True if the environment is correct
        """
        while hasattr(env, 'env'):
            env = env.env
            if isinstance(env, Coffee_Drawer_Novelty):
                return True
        return False
