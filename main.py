import curses
from dataclasses import dataclass
import random
import locale
import time

import os
import logging

# Add these BEFORE importing cv2 and mediapipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)

import cv2
cv2.setLogLevel(0)  # or cv2.LOG_LEVEL_SILENT if available

import mediapipe
import numpy as np

# Initialize Mediapipe
mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles


@dataclass
class Config:
    # Use ASCII fallbacks for better terminal compatibility
    player_char: str = "^"
    enemy_char: str = "V"
    player_step: int = 4
    tmp_player_step: int = 4
    simple_fire_char: str = "|"
    max_enemy_per_line = 8
    enemy_move_delay = 15
    bullet_hitbox = 2  # Fixed typo: bullte -> bullet
    target_fps = 30

class SpaceShooter:
    def __init__(self, stdscr, config=Config()):
        self.config = config
        self.stdscr = stdscr
        self.height, self.width = self.stdscr.getmaxyx()
        self.game_width = self.width // 2
        self.player_x = self.game_width // 2
        self.player_y = self.height - 10
        self.using_visual_commands = True
        self.cap = None
        self.recent_player_dirs = [0,0,0,0,0,0,0,0]
        
        # visual controller
        self.first_hand_pos = {
            "x": 0,
            "y": 0
        }
        self.second_hand_pos = {
            "x": 0,
            "y": 0
        }
        
        # Initialize curses properly
        self.setup_curses()
        
        self.shoots: list = []
        self.enemies: list = []
        self.loop_count = 0

        # score & player
        self.score = 0
        self.health = 5
       
        # Timing control
        self.frame_time = 1.0 / self.config.target_fps
        self.last_frame_time = time.time()

        # Initialize mediapipe hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def render_control(self):
        """Render Hands"""
        base_x = self.game_width
        base_y = 5

        global_first_hand_position = {
            "x": base_x + self.first_hand_pos["x"],
            "y": base_y + self.first_hand_pos["y"],
        }
        global_second_hand_position = {
            "x": base_x + self.second_hand_pos["x"],
            "y": base_y + self.second_hand_pos["y"],
        }
        self.safe_addstr(3, 3, f"{global_first_hand_position['y']}, {global_first_hand_position['x']}")
        self.safe_addstr(6, 3, f"{global_second_hand_position['y']}, {global_second_hand_position['x']}")

        self.safe_addch(global_first_hand_position["y"], global_first_hand_position["x"], "🤚")
        self.safe_addch(global_second_hand_position["y"], global_second_hand_position["x"], "✋")
        
        
    def setup_curses(self):
        """Properly initialize curses with better compatibility"""
        # Set locale for better character support
        try:
            locale.setlocale(locale.LC_ALL, '')
        except locale.Error:
            pass  # Continue with default locale if setting fails
        
        # Configure curses
        curses.curs_set(0)  # Hide cursor
        self.stdscr.nodelay(True)  # Non-blocking input
        self.stdscr.timeout(0)  # No timeout for getch
        
        # Initialize colors if available
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        
        # Set encoding for better Unicode support
        if hasattr(self.stdscr, 'encoding'):
            self.stdscr.encoding = 'utf-8'

    def safe_addch(self, y, x, char, attr=0):
        """Safely add character with bounds checking and fallback"""
        try:
            if 0 <= y < self.height and 0 <= x < self.width:
                self.stdscr.addch(y, x, char, attr)
        except curses.error:
            # If the character fails, try ASCII fallback
            fallback_chars = {
                "🏹": "^", "🍎": "V", "🤲": "|",
                "█": "#", "▓": ":", "░": "."
            }
            fallback = fallback_chars.get(char, "#")
            try:
                if 0 <= y < self.height and 0 <= x < self.width:
                    self.stdscr.addch(y, x, fallback, attr)
            except curses.error:
                pass  # Skip if even fallback fails

    def safe_addstr(self, y, x, text, attr=0):
        """Safely add string with bounds checking"""
        try:
            if 0 <= y < self.height and 0 <= x < self.width:
                # Truncate text if it would go beyond screen width
                max_len = self.width - x - 1
                if len(text) > max_len:
                    text = text[:max_len]
                self.stdscr.addstr(y, x, text, attr)
        except curses.error:
            pass

    def run(self):
        while self.health > 0:
            frame_start = time.time()
            
            if self.using_visual_commands and self.cap == None:
                self._run_camera()
            
            # Update game state
            self.update_game_state()
            
            # Render everything
            self.p()
            
            # Control frame rate
            self.control_frame_rate(frame_start)

            # Handle input
            self.handle_input()
            
            self.loop_count += 1
        
        if self.using_visual_commands:
            self._stop_camera()
    
    def _run_camera(self):

        self.cap = cv2.VideoCapture(0)
        _, frame = self.cap.read()
    
        self.initial_blank = np.zeros(frame.shape, dtype=np.uint8)

        self.using_visual_commands = True

    def _stop_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("Camera released and windows closed.")

    def _get_visual_commands(self):
        if not self.cap.isOpened():
            print("ERROR: Could not open Camera")
            self.using_visual_commands = False
            return
        else:
            self.using_visual_commands = True

        ret, frame = self.cap.read()
        blank = self.initial_blank.copy()

        if not ret:
            print("ERROR: Failed to capture frame")
            return 

        # Flip the frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process hands
        hand_results = self.hands.process(rgb_frame)

        if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) ==2:
            h, w, c = blank.shape

            first_hand = hand_results.multi_hand_landmarks[0]
            second_hand = hand_results.multi_hand_landmarks[1]
            first_hand_position = first_hand.landmark[0]
            second_hand_position = second_hand.landmark[0]
            self.first_hand_pos["x"] = int(first_hand_position.x * (self.width // 2))
            self.first_hand_pos["y"] = int(first_hand_position.y * self.height)
            self.second_hand_pos["x"] = int(second_hand_position.x * (self.width // 2))
            self.second_hand_pos["y"] = int(second_hand_position.y * self.height)

            first_x_y = (first_hand_position.x, first_hand_position.y)
            second_x_y = (second_hand_position.x, second_hand_position.y)

            vertical_height = abs(first_hand_position.y - second_hand_position.y)
            horizontal_length = abs(first_hand_position.x - second_hand_position.x)
            hypotenuse = np.sqrt(vertical_height**2 + horizontal_length**2)

            if horizontal_length > 0:
                _slope = (first_hand_position.y - second_hand_position.y) / (first_hand_position.x - second_hand_position.x)
                _sign = 1 if _slope > 0 else -1
                _sin = _sign * (vertical_height / hypotenuse)
                _dir = _sign
                
                # compute player x acceleration
                self.recent_player_dirs = [_sign] + self.recent_player_dirs
                if len(self.recent_player_dirs) > 8:
                    self.recent_player_dirs.pop()
                times_in_this_direction = 0
                for i in self.recent_player_dirs:
                    if i == _dir:
                        times_in_this_direction += 1
                    else: break
                _step_increase_rate = 1.1 ** (times_in_this_direction - 1)
                self.config.tmp_player_step = self.config.player_step * _step_increase_rate

                _tmp_step = int(self.config.tmp_player_step * _sin)
                new_player_x = self.player_x + _tmp_step

                if _sign == 1:
                    if new_player_x < self.game_width - self.config.tmp_player_step:
                        self.player_x += _tmp_step
                    else:
                        self.player_x = self.game_width - self.config.player_step
                else:
                    if self.config.tmp_player_step < new_player_x:
                        self.player_x += _tmp_step
                    else:
                        self.player_x = 0 + self.config.player_step

                self.fire_simple()

        self.stdscr.clear()
        self.stdscr.refresh()

    def handle_input(self):
        """Handle all input in one place"""
        key = self.stdscr.getch()
        if key == ord("q"):
            self.health = 0  # Exit game

        if self.using_visual_commands:
            if self.loop_count % 3 == 0:
                self._get_visual_commands()
            return

        elif key == curses.KEY_LEFT:
            if self.player_x > 1:
                self.player_x -= self.config.player_step
        elif key == curses.KEY_RIGHT:
            if self.player_x < self.game_width - 2:
                self.player_x += self.config.player_step
        elif key == curses.KEY_UP:
            if self.player_y > 1:
                self.player_y -= self.config.player_step
        elif key == curses.KEY_DOWN:
            if self.player_y < self.height - 2:
                self.player_y += self.config.player_step
        elif key == ord(" "):
            self.fire_simple()

    def update_game_state(self):
        """Update all game objects"""
        self.update_shoots()
        self.update_enemies()
        self.update_shoot_enemy()

    def update_shoots(self):
        """Update bullet positions and remove off-screen bullets"""
        for idx in range(len(self.shoots) - 1, -1, -1):
            shoot = self.shoots[idx]
            shoot["y"] -= 2
            if shoot["y"] <= 0:
                self.shoots.pop(idx)

    def update_enemies(self):
        """Update enemy positions and generate new enemies"""
        # Only move enemies and generate new ones based on the delay
        if self.loop_count % self.config.enemy_move_delay == 0:
            self.generate_enemy()
            
            # Move enemies down
            for idx in range(len(self.enemies) - 1, -1, -1):
                enemy = self.enemies[idx]
                enemy["y"] += 2
                if enemy["y"] >= self.height - 1:
                    self.enemies.pop(idx)
                    self.health -= 1

    def p(self):
        """Render everything using double buffering technique"""
        # Clear screen (but don't refresh yet)
        self.stdscr.erase()
        
        # Draw player
        self.safe_addch(self.player_y, self.player_x, self.config.player_char, 
                       curses.color_pair(2) if curses.has_colors() else 0)
        
        # Draw bullets
        for shoot in self.shoots:
            self.safe_addch(shoot["y"], shoot["x"], self.config.simple_fire_char,
                           curses.color_pair(3) if curses.has_colors() else 0)
        
        # Draw enemies
        for enemy in self.enemies:
            self.safe_addch(enemy["y"], enemy["x"], self.config.enemy_char,
                           curses.color_pair(1) if curses.has_colors() else 0)
        
        # Draw UI
        self.render_ui()

        # Draw controlls 
        self.render_control()
        
        # Draw border (optional, for better visual feedback)
        self.draw_border()
        
        # Double buffering: update virtual screen then physical screen
        self.stdscr.noutrefresh()
        curses.doupdate()

    def render_ui(self):
        """Render game UI elements"""
        # self.safe_addstr(0, 1, f"Health: {self.health}")
        # self.safe_addstr(0, 15, f"Score: {self.score}")
        # self.safe_addstr(0, 30, f"Enemies: {len(self.enemies)}")
        self.safe_addstr(0, 45, f"player: ({self.player_x},{self.player_y})")
        
        # Instructions
        instructions = "Arrow keys: Move | Space: Fire | Q: Quit"
        if len(instructions) < self.game_width - 2:
            self.safe_addstr(self.height - 1, 1, instructions)

    def draw_border(self):
        """Draw a simple border around the play area"""
        try:
            # Top and bottom borders
            for x in range(self.game_width):
                self.safe_addch(1, x, "-")
            
            # Side borders (avoid bottom line to prevent cursor wrap issues)
            for y in range(2, self.height - 1):
                self.safe_addch(y, 0, "|")
                self.safe_addch(y, self.game_width - 1, "|")
                
        except curses.error:
            pass  # Skip border if it causes issues

    def control_frame_rate(self, frame_start):
        """Control frame rate for smooth animation"""
        frame_time = time.time() - frame_start
        sleep_time = self.frame_time - frame_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    def fire_simple(self):
        """Fire a bullet"""
        if self.player_y > 2:  # Only fire if not at top
            self.shoots.append({"x": self.player_x, "y": self.player_y - 1})

    def generate_enemy(self):
        """Generate new enemies"""
        may_we_generate_now = self._random_true_false(chance=0.20)

        if not may_we_generate_now:
            return
        
        num_enemies = self.config.max_enemy_per_line
        for i in range(num_enemies):
            if self._random_true_false(chance=0.6):
                new_enemy = {
                    "x": random.randint(2, self.game_width - 3),
                    "y": 2
                }
                self.enemies.append(new_enemy)

    def update_shoot_enemy(self):
        """Handle bullet-enemy collisions"""
        enemies_to_remove = []
        shoots_to_remove = []
        bullet_hitbox = self.config.bullet_hitbox
        
        for eidx, enemy in enumerate(self.enemies):
            for sidx, shoot in enumerate(self.shoots):
                # Check collision
                if (abs(shoot["x"] - enemy["x"]) <= bullet_hitbox and 
                    abs(shoot["y"] - enemy["y"]) <= bullet_hitbox):
                    enemies_to_remove.append(eidx)
                    shoots_to_remove.append(sidx)
    
        # Remove collided objects (in reverse order to avoid index issues)
        for eidx in sorted(set(enemies_to_remove), reverse=True):
            if eidx < len(self.enemies):
                self.enemies.pop(eidx)
                self.score += 1
        
        for sidx in sorted(set(shoots_to_remove), reverse=True):
            if sidx < len(self.shoots):
                self.shoots.pop(sidx)

    def _random_true_false(self, chance: float):
        """Generate random boolean with given probability"""
        if chance >= 0.99:
            return True
        return random.random() < chance

def main(stdscr):
    try:
        game = SpaceShooter(stdscr)
        game.run()
    except KeyboardInterrupt:
        pass  # Clean exit on Ctrl+C
    finally:
        # Ensure proper cleanup
        curses.curs_set(1)  # Restore cursor
        stdscr.nodelay(False)  # Restore blocking input

if __name__ == "__main__":
    curses.wrapper(main)
