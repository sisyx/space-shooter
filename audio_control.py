import pygame.mixer
import threading
from pathlib import Path

class GameAudio:
    def __init__(self, channels=16):
        self.audios_base = Path("audios")
        
        # Initialize mixer with more channels to support multiple simultaneous sounds
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(channels)  # Set number of mixing channels
        
        self.load_sounds()

    def load_sounds(self):
        self.sounds = {
            "shoot": pygame.mixer.Sound(self.audios_base / "shoot.mp3"),
            "explosion": pygame.mixer.Sound(self.audios_base / "explosion.mp3"),
        }

    def play_sound(self, sound_name):
        if sound_name in self.sounds:
            # Find an available channel or force play on any channel
            channel = pygame.mixer.find_channel()
            if channel is None:
                # If no free channel, use channel 0 (will interrupt whatever is playing there)
                channel = pygame.mixer.Channel(0)
            
            thread = threading.Thread(target=channel.play, args=(self.sounds[sound_name],))
            thread.daemon = True
            thread.start()

    def play_sound_immediate(self, sound_name):
        """Alternative method that doesn't use threading - might be more reliable"""
        if sound_name in self.sounds:
            channel = pygame.mixer.find_channel()
            if channel is None:
                # Force play on a channel if all are busy
                channel = pygame.mixer.Channel(len(pygame.mixer.get_init()) % pygame.mixer.get_num_channels())
            channel.play(self.sounds[sound_name])

    def _play_shoot_sound(self):
        self.play_sound("shoot")
    
    def _play_explosion_sound(self):
        self.play_sound("explosion")