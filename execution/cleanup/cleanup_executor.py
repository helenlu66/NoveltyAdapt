from execution.executor import *
from detection.cleanup_detector import CleanupDetector
from learning.learning_utils import *


class PickUpFromTabletop(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="pick-up-from-tabletop")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action):
        """execute the pick up from tabletop operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing pick-up-from-tabletop operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the pick-up-from-tabletop loop
        
    
class FreeGripperFromLargeObject(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="free-gripper-from-large-object")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action):
        """execute the free gripper from large object operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing free-gripper-from-large-object operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the free-gripper-from-large-object loop
        

class OpenDrawer(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="open-drawer")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action):
        """execute the open drawer operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing open-drawer operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the open-drawer loop
    
class CloseDrawer(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="close-drawer")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action):
        """execute the close drawer operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing close-drawer operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the close-drawer loop

class PlaceInDrawerFromGripper(Executor): 
    def __init__(self):
        super().__init__("coded", operator_name="place-in-drawer-from-gripper")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action):
        """execute the place in drawer from gripper operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing place-in-drawer-from-gripper operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the place-in-drawer-from-gripper loop


class PourOutFromMug(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="pour-out-from-mug")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the pour out contents operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator.ident())
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: change the state of the world such that the gripper is holding the mug above the tabletop and the cube has been poured onto the tabletop
        
        env._place_mug_in_gripper()
        
        
        step = 0
        while step < 20:
            # close the gripper a bit more
            env.step([0, 0, 0, 0, 0, 0, 1])
            if render:
                env.render()
            step += 1
        step = 0
        while step < 15:
            # turn the mug upside down a bit more
            env.step([0.5, 0, 0.5, 0, 0.75, 0, 0])
            if render:
                env.render()
            step += 1
        step = 0
        while step < 10:
            # turn the mug upside down a bit more
            env.step([0, 0, 0, 0, 0, 0, 0])
            if render:
                env.render()
            step += 1
        # check if the operator has been executed successfully
        binary_obs = detector.detect_binary_states(env)
        success =  all(check_condition_satisfied(effect, binary_obs) for effect in grounded_operator.effects)
        print(f"Execution of pour-out-from-mug successful: {success}")
        return success

CLEANUP_EXECUTORS = {
    "pick-up-from-tabletop": PickUpFromTabletop(),
    "free-gripper-from-large-object": FreeGripperFromLargeObject(),
    "open-drawer": OpenDrawer(),
    "close-drawer": CloseDrawer(),
    "place-in-drawer-from-gripper": PlaceInDrawerFromGripper(),
    "pour-out-from-mug": PourOutFromMug()
}

if __name__ == "__main__":
    domain = "cleanup"
    config = load_config("config.yaml")
    env = load_env(domain, config['simulation'])
    env.reset()
    detector = CleanupDetector(env)

    # get the grounded operator
    plan = load_plan(config['planning'][domain])
    grounded_operator = find_grounded_operator_from_plan(
        plan, 'pour-out-from-mug')
    # load the executor
    executor = CLEANUP_EXECUTORS['pour-out-from-mug']
    success = executor.execute(detector, grounded_operator, render=True)

    print(f"Execution of pour-out-from-mug successful: {success}")

    step = 0
    while step < 50:
        # input("Press enter to continue...")
        env.step([0, 0, 0, 0, 0, 0, 1])
        env.render()
        step += 1