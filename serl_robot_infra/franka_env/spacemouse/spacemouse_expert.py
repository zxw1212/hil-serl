import time
import multiprocessing
import numpy as np
import inputs
from franka_env.spacemouse import pyspacemouse
from typing import Tuple


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6  # Using lists for compatibility
        self.latest_data["buttons"] = [0, 0, 0, 0]

        # Start a process to continuously read the SpaceMouse state
        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True
        self.process.start()

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read_all()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            if len(state) == 2:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw
                ]
                buttons = state[0].buttons + state[1].buttons
            elif len(state) == 1:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw
                ]
                buttons = state[0].buttons

            # Update the shared state
            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons
    
    def close(self):
        # pyspacemouse.close()
        self.process.terminate()

class JoystickExpert:
    """
    This class provides an interface to the Joystick/Gamepad.
    It continuously reads the joystick state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["buttons"] = [False, False]

        # Start a process to continuously read Joystick state
        self.process = multiprocessing.Process(target=self._read_joystick)
        self.process.daemon = True
        self.process.start()


    def _read_joystick(self):        
        # Scale factors for different axes
        scale = {
            'ABS_X': 0.1,    # Adjust these scale factors
            'ABS_Y': 0.1,    # to control sensitivity
            'ABS_RX': 0.3,
            'ABS_RY': 0.3,
            'ABS_Z': 0.04,
            'ABS_RZ': 0.04,
            'ABS_HAT0X': 0.3,
        }
        
        action = [0.0] * 6
        buttons = [False, False]
        
        while True:
            try:
                # Get fresh events
                events = inputs.get_gamepad()
          
                # Process events
                for event in events:
                    # Calculate relative changes based on the axis
                    if event.code == 'ABS_Y':
                        action[0] = -(event.state / 32768.0) * scale[event.code]
                        
                    elif event.code == 'ABS_X':
                        action[1] = -(event.state / 32768.0) * scale[event.code]
                        
                    elif event.code == 'ABS_Z':
                        action[2] = -(event.state / 255.0) * scale[event.code]

                    elif event.code == 'ABS_RZ':
                        action[2] = (event.state / 255.0) * scale[event.code]

                    elif event.code == 'ABS_RX':
                        action[3] = (event.state / 32768.0) * scale[event.code]
                        
                    elif event.code == 'ABS_RY':
                        action[4] = (event.state / 32768.0) * scale[event.code]

                    elif event.code == 'ABS_HAT0X':
                        action[5] = (event.state / 255.0) * scale[event.code]
                        
                    # Handle button events
                    elif event.code == 'BTN_TL':
                        buttons[0] = bool(event.state)
                    elif event.code == 'BTN_TR':
                        buttons[1] = bool(event.state)

                # Update the shared state
                self.latest_data["action"] = action
                self.latest_data["buttons"] = buttons
                
            except inputs.UnpluggedError:
                print("No controller found. Retrying...")
                time.sleep(1)

    def get_action(self):
        """Returns the latest action and button state from the Joystick."""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons
    
    def close(self):
        self.process.terminate()
