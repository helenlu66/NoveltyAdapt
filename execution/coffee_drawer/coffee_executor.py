from execution.executor import *
from detection.coffee_drawer_detector import CoffeeDrawerDetector


class PickUpFromTabletopExecutor(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="pick-up-from-tabletop")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the pick up from tabletop operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing pick-up-from-tabletop operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()
        
        #TODO: code the execution of the pick-up-from-tabletop loop

        # render the env after execution
        if render:
            env.render()
        

class OpenCoffeePodHolder(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="open-coffee-pod-holder")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the open coffee pod holder operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing open-coffee-pod-holder operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the open-coffee-pod-holder loop

        # render the env after execution
        if render:
            env.render()
        
    
class CloseCoffeePodHolder(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="close-coffee-pod-holder")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the close coffee pod holder operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing close-coffee-pod-holder operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the close-coffee-pod-holder loop

        # render the env after execution
        if render:
            env.render()
         

class FreeGripperFromLargeObject(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="free-gripper-from-large-object")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the free gripper from large object operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing free-gripper-from-large-object operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        # the gripper starts off not holding anything in the env so this is a hacky way to simulate the operator
        # render the env after execution
        if render:
            env.render()
        return True
        
    
class PlacePodInHolderFromGripper(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="place-pod-in-holder-from-gripper")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the place pod in holder from gripper operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing place-pod-in-holder-from-gripper operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the place-pod-in-holder-from-gripper loop

        # render the env after execution
        if render:
            env.render()
        

class PlaceMugUnderHolderFromGripper(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="place-mug-under-holder-from-gripper")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the place mug under holder from gripper operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing place-mug-under-holder-from-gripper operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the place-mug-under-holder-from-gripper loop

        # render the env after execution
        if render:
            env.render()

class OpenDrawer(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="open-drawer")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the open drawer operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing open-drawer operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: change the state of the drawer to open

class OpenDrawerOpenCoffeePodHolder(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="open-drawer-open-coffee-pod-holder")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action=None, render=False):
        """execute the open drawer open coffee pod holder operator in the coffee
           simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults
                                     to False.
        """
        
        # if grounded_operator is not None:
        #     grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator.ident())
        #     assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        binary_obs = detector.detect_binary_states(env)
        if binary_obs["open drawer1"] and binary_obs["open coffee-pod-holder1"] and binary_obs["free gripper1"]:
            return True
        # change the state of the drawer to open and the coffee pod holder to open
        env._reset_open_drawer_open_lid()

        steps = 0
        while steps < 2:
            env.step(np.array([0,0,0,0])) # for some reason the env needs this to update the state
            if render:
                env.render()
            steps += 1
        binary_obs = detector.detect_binary_states(env)
        success=  binary_obs["open drawer1"] and binary_obs["open coffee-pod-holder1"] and binary_obs["free gripper1"]
        print(f"Open drawer open coffee pod holder successful: {success}")
        return success


COFFEE_EXECUTORS = {
    "pick-up-from-tabletop": PickUpFromTabletopExecutor(),
    "close-coffee-pod-holder": CloseCoffeePodHolder(),
    "place-pod-in-holder-from-gripper": PlacePodInHolderFromGripper(),
    "place-mug-under-holder-from-gripper": PlaceMugUnderHolderFromGripper(),
    "open-drawer": OpenDrawerOpenCoffeePodHolder(), # dirty hack to avoid implementing open-drawer
    "open-coffee-pod-holder": OpenDrawerOpenCoffeePodHolder(), # dirty hack to avoid implementing open-coffee-pod-holder
    "free-gripper-from-object-and-return-to-neutral": OpenDrawerOpenCoffeePodHolder(), # dirty hack to avoid implementing free-gripper-from-large-object
    "open-drawer-open-coffee-pod-holder": OpenDrawerOpenCoffeePodHolder()
}

if __name__ == "__main__":
    # Test the coffee executor

    # make the coffee novelty env
    domain = "coffee"
    config = load_config("config.yaml")
    env = load_env(domain, config['simulation'])
    env = VisualizationWrapper(env, indicator_configs=None)
    env.viewer.set_camera(camera_id=0)
    executor = COFFEE_EXECUTORS["open-drawer-open-coffee-pod-holder"]
    detector = CoffeeDrawerDetector(env)
    success = executor.execute(detector, None, render=True)
    print(f"Execution of open-drawer-open-coffee-pod-holder successful: {success}")
    env.render()
    for i in range(50):
        # input("Press enter to continue")
        env.step(np.array([0,0,0,0]))
        env.render()
    

    
