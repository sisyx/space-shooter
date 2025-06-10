import curses
from dataclasses import dataclass
import random

@dataclass
class Config:
    player_char: str = "▲"
    enemy_char: str = "O"
    player_step: int = 4
    simple_fire_char: str = "^"
    max_enemy_per_line = 8
    enemy_move_delay = 25
    bullte_hitbox = 2 # the radius a bullet has impact

class SpaceShooter:
    def __init__(self, stdscr, config=Config()):
        self.config = config
        self.stdscr = stdscr
        self.height, self.width = self.stdscr.getmaxyx()
        self.player_x = self.width // 2
        self.player_y = self.height - 1
        self.stdscr.timeout(50)
        curses.curs_set(0)
        self.shoots: list = []
        self.enemies: list = []
        self.loop_count = 0

    def run(self):
        while True:
            self.stdscr.clear()
            self.stdscr.addch(self.player_y, self.player_x, self.config.player_char)

            key = self.stdscr.getch()
            if key == ord("q"):
                break
            elif key == curses.KEY_LEFT:
                if self.player_x > 1:
                    self.player_x -= self.config.player_step
            elif key == curses.KEY_RIGHT:
                if self.player_x < self.width - 2:
                    self.player_x += self.config.player_step
            elif key == curses.KEY_UP:
                if self.player_y > 1:
                    self.player_y -= self.config.player_step
            elif key == curses.KEY_DOWN:
                if self.player_y < self.height - 1:
                    self.player_y += self.config.player_step
            elif key == ord(" "):
                self.fire_simple()

            # Manage Shoots
            for idx in range(len(self.shoots) -1 , -1, -1):
                shoot = self.shoots[idx]
                shoot["y"] = shoot["y"] - 2
                if shoot["y"] <= 1:
                    self.shoots.pop(idx)
                else:
                    self.stdscr.addch(shoot["y"], shoot["x"], self.config.simple_fire_char)
            
            # Manage Enemies
            if self.loop_count % self.config.enemy_move_delay == 0:
                self.update_enemies()
                for idx in range(len(self.enemies) - 1, -1, -1):
                    enemy = self.enemies[idx]
                    enemy["y"] = enemy["y"] + 2
                    if enemy["y"] >= self.height:
                        self.enemies.pop(idx)
                        # update user score here

            self.update_shoot_enemy()
            for enemy in self.enemies:
                self.stdscr.addch(enemy["y"], enemy["x"],
                                  self.config.enemy_char, curses.color_pair(1))
                
            self.stdscr.addch(5, 5, str(len(self.enemies))[0])
            self.stdscr.refresh()
            self.loop_count += 1

    def fire_simple(self):
        self.shoots.append({"x": self.player_x, "y": self.player_y - 2})

    def generate_enemy(self):
        may_we_generate_now = self._random_true_false(chance=0.20)

        if not may_we_generate_now:
            return
        
        num_enemies = self.config.max_enemy_per_line
        for i in range(num_enemies):
            if (self._random_true_false(chance=0.6)):
                new_enemy = {
                            "x": random.randint(5, self.width - 5),
                            "y": 5
                       }
                self.enemies.append(new_enemy)

    
    def update_enemies(self):
        self.generate_enemy()
    
    def update_shoot_enemy(self):
        enemies_to_remove = []
        shoots_to_remove = []
        bullte_hitbox = self.config.bullte_hitbox
        for eidx in range(len(self.enemies) - 1, -1, -1):
            enemy = self.enemies[eidx]
            for sidx in range(len(self.shoots) - 1, -1, -1):
                shoot = self.shoots[sidx]
                # Check if enemy is within a 5-pixel box around shoot
                if (
                    (shoot["x"] - bullte_hitbox <= enemy["x"] <= shoot["x"] + bullte_hitbox) and
                    (shoot["y"] - bullte_hitbox <= enemy["y"] <= shoot["y"] + bullte_hitbox)
                ):
                    enemies_to_remove.append(eidx)
                    shoots_to_remove.append(sidx)
    
        # Remove in reverse order to avoid index shifting issues
        for eidx in sorted(set(enemies_to_remove), reverse=True):
            if eidx < len(self.enemies):
                self.enemies.pop(eidx)
        
        for sidx in sorted(set(shoots_to_remove), reverse=True):
            if sidx < len(self.shoots):
                self.shoots.pop(sidx)
    
    # Update score here (e.g., self.score += len(enemies_to_remove))
    def _random_true_false(self, chance: int):
        if chance > 0.99: ## don't calculate if chance is already a win chance
            return True

        generated_num = random.randint(0, 100) / 100
        return generated_num >  1 - chance

def main(stdscr):
    game = SpaceShooter(stdscr)
    game.run()

if __name__=="__main__":
    curses.wrapper(main)
