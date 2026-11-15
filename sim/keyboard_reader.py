import threading
import os
import sys
import ctypes
from ctypes import wintypes
from pynput import keyboard

class KeyboardHoldReader:
    def __init__(self, consume_viewer_arrows=False, balance_hotkey=False):
        self.speed_axis = 0.0
        self.yaw_axis = 0.0
        self.hight_axis = 0.0
        self.jump_requested = False
        self.jump_held = False
        self.balance_toggle = self.balance_key_held = False
        self.lock = threading.Lock()
        self.consume_viewer_arrows = consume_viewer_arrows and sys.platform == "win32"
        options = {}
        self.filter_viewer_keys = sys.platform == "win32" and (consume_viewer_arrows or balance_hotkey)
        if self.filter_viewer_keys:
            self.user32 = ctypes.WinDLL("user32", use_last_error=True)
            self.user32.GetForegroundWindow.restype = wintypes.HWND
            self.user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
            self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD
            options["win32_event_filter"] = self._viewer_key_filter
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release, **options)
        self.listener.start()

    def _viewer_focused(self):
        pid = wintypes.DWORD()
        window = self.user32.GetForegroundWindow()
        self.user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
        return bool(window) and pid.value == os.getpid()

    def _viewer_key_filter(self, msg, data):
        arrows = {0x25: keyboard.Key.left, 0x26: keyboard.Key.up,
                  0x27: keyboard.Key.right, 0x28: keyboard.Key.down,
                  0x4A: keyboard.KeyCode.from_char("j"),
                  0x4D: keyboard.KeyCode.from_char("m")}
        if not self._viewer_focused():
            # Returning False filters this listener, without suppressing OS input.
            return False
        if data.vkCode in arrows and (self.consume_viewer_arrows or data.vkCode == 0x4D):
            key = arrows[data.vkCode]
            if msg in (0x101, 0x105):
                self.on_release(key)
            elif msg in (0x100, 0x104):
                self.on_press(key)
            # Only consume arrows, J and M for this process's foreground viewer. Other
            # keys, including Space/Backspace, still reach the native MuJoCo UI.
            self.listener.suppress_event()
        return True

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
            elif (getattr(key, "char", None) or "").lower() == "j":
                with self.lock:
                    if not self.jump_held:
                        self.jump_requested = True
                    self.jump_held = True
            elif (getattr(key, "char", None) or "").lower() == "m":
                with self.lock:
                    if not self.balance_key_held:
                        self.balance_toggle = True
                    self.balance_key_held = True
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
            elif (getattr(key, "char", None) or "").lower() == "j":
                with self.lock:
                    self.jump_held = False
            elif (getattr(key, "char", None) or "").lower() == "m":
                with self.lock:
                    self.balance_key_held = False
        except AttributeError:
            pass

    def read_axes(self):
        if self.consume_viewer_arrows and not self._viewer_focused():
            with self.lock:
                self.speed_axis = self.yaw_axis = self.hight_axis = 0.0
                self.jump_requested = self.jump_held = False
        with self.lock:
            jump_requested = self.jump_requested
            self.jump_requested = False
            return (self.speed_axis, self.yaw_axis, self.hight_axis, jump_requested)

    def consume_balance_toggle(self):
        with self.lock:
            toggle = self.balance_toggle
            self.balance_toggle = False
            if self.filter_viewer_keys and not self._viewer_focused():
                self.balance_key_held = False
                return False
            return toggle

    def stop(self):
        self.listener.stop()
