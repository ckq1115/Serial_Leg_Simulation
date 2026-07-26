import threading
from pynput import keyboard

class KeyboardHoldReader:
    def __init__(self):
        self.speed_axis = 0.0
        self.yaw_axis = 0.0
        self.hight_axis = 0.0
        self.jump_pressed = False
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            if key == keyboard.Key.up:
                with self.lock:
                    self.speed_axis = 1.0
            elif key == keyboard.Key.down:
                with self.lock:
                    self.speed_axis = -1.0
            elif key == keyboard.Key.left:
                with self.lock:
                    self.yaw_axis = 1.0
            elif key == keyboard.Key.right:
                with self.lock:
                    self.yaw_axis = -1.0
            elif key == keyboard.Key.shift_l:
                with self.lock:
                    self.hight_axis = 0.001
            elif key == keyboard.Key.shift_r:
                with self.lock:
                    self.hight_axis = -0.001
            elif key == keyboard.Key.space:
                with self.lock:
                    self.jump_pressed = True
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.up or key == keyboard.Key.down:
                with self.lock:
                    self.speed_axis = 0.0
            elif key == keyboard.Key.left or key == keyboard.Key.right:
                with self.lock:
                    self.yaw_axis = 0.0
            elif key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                with self.lock:
                    self.hight_axis = 0.0
            elif key == keyboard.Key.space:
                with self.lock:
                    self.jump_pressed = False
        except AttributeError:
            pass

    def read_axes(self):
        with self.lock:
            return self.speed_axis, self.yaw_axis, self.hight_axis, self.jump_pressed

    def stop(self):
        self.listener.stop()