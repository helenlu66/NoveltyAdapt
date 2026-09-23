from execution.executor import *
from detection.kitchen_detector import KitchenDetector
from learning.learning_utils import *


class PickUpFromTabletop(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="pick-up-from-tabletop")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the pick up from tabletop operator in the kitchen simulation environment

        Args:
            detector (Detector): the detector for the kitchen domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        #print("Pretending to executing pick-up-from-tabletop operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator.name)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"

        #print('executed pick-up-from-tabletop successfully')
        return True
        
    
class FreeGripperFromFixedObject(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="free-gripper-from-large-object")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the free gripper from large object operator in the kitchen simulation environment

        Args:
            detector (Detector): the detector for the kitchen domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing free-gripper-from-large-object operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the free-gripper-from-large-object loop
        

class PlaceOnTable(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="place-on-table")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the place on table operator in the kitchen simulation environment
        Args:
            detector (Detector): the detector for the kitchen domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing place-on-table operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the open-drawer loop
    
class PlaceInPot(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="place-in-pot")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the place in pot operator in the kitchen simulation environment

        Args:
            detector (Detector): the detector for the kitchen domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing place-in-pot operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the place-in-pot loop

class PlaceOnObject(Executor): 
    def __init__(self):
        super().__init__("coded", operator_name="place-on-object")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the place on object operator in the kitchen simulation environment

        Args:
            detector (Detector): the detector for the kitchen domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing place-on-object operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the place-on-object loop


class TurnOnStove(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="turn-on-stove")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the turn on stove operator in the kitchen simulation environment

        Args:
            detector (Detector): the detector for the kitchen domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator.ident())
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

class PotOnStove(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="pot-on-stove")
    
    def check_success(self, detector:Detector):
        """check if the pot is on the stove

        Args:
            detector (Detector): the detector for the kitchen domain

        Returns:
            bool: True if the pot is on the stove
        """
        env = detector.get_env()
        binary_obs = detector.detect_binary_states(env)
        on_stove = binary_obs["ontop pot1 stove1"]
        free_gripper = binary_obs["free gripper1"]
        lid_on = binary_obs["ontop lid1 pot1"] and binary_obs["covered pot1"]
        success = on_stove and free_gripper and lid_on
        if self.verbose:
            print(f"{self.name}: {success}")
        return success
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """spawn the pot on the stove

        Args:
            detector (Detector): the detector for the kitchen domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        env = detector.get_env()
        if self.check_success(detector):
            steps = 0
            while steps < 2:
                env.step(np.array([0,0,0,0])) # for some reason the env needs this to update the state
                if render:
                    env.render()
                steps += 1
            return True
        else:
            env._place_pot_on_stove()
            steps = 0
            while steps < 2:
                env.step(np.array([0,0,0,0])) # for some reason the env needs this to update the state
                if render:
                    env.render()
                steps += 1
            success = self.check_success(detector)
            if self.verbose:
                print('execution success:', success)
            return success

KITCHEN_EXECUTORS = {
    "pick-up-from-tabletop": PickUpFromTabletop(),
    "free-gripper-from-fixed-object": FreeGripperFromFixedObject(),
    "place-on-table": PlaceOnTable(),
    "place-in-pot": PlaceInPot(),
    "place-on-object": PotOnStove(),
    "turn-on-stove": TurnOnStove()
}

if __name__ == "__main__":
    domain = "kitchen"
    config = load_config("config.yaml")
    config['simulation']['has_renderer'] = True
    env = load_env(domain, config['simulation'])
    camera_id = 3
    print(env.sim.model.camera_names)
    print(env.sim.model.camera_names[camera_id])
    env.viewer.set_camera(camera_id=camera_id)
    env.reset()
    detector = KitchenDetector(env)

    # get the grounded operator
    plan = load_plan(config['planning'][domain])
    grounded_operator = find_grounded_operator_from_plan(
        plan, 'place-on-object')
    # load the executor
    executor = KITCHEN_EXECUTORS['place-on-object']
    success = executor.execute(detector, grounded_operator, render=True)