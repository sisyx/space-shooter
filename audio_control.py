import pygame.mixer
import threading
from pathlib import Path

class GameAudio:
    def __init__(self):
        self.audios_base = Path("audios")

        pygame.mixer.init()
        self.load_sounds()

    def load_sounds(self):
        self.sounds = {
            "shoot": pygame.mixer.Sound(self.audios_base / "shoot.mp3")
        }

    def play_sound(self, sound_name):
        if sound_name in self.sounds:
            thread = threading.Thread(target=self.sounds[sound_name].play)
            thread.daemon = True
            thread.start()

    def _play_shoot_sound(self):
        self.play_sound("shoot")

