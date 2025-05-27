import os
import numpy as np
import yaml
import pygame
from typing import Optional, Tuple

class GridWorldEnvDiscrete:
    """
    Discrete GridWorld: 8-connected moves on a grid with walls, traps, and goals.
    State: agent's (row, col) cell indices.
    Action: integer 0-7 corresponding to directions N, NE, E, SE, S, SW, W, NW.
    Rewards: -1 per step, +100 at goal, -100 at trap.
    """
    def __init__(
        self,
        config_path: str,
        cell_size_px: int = 80,
        headless: bool = False
    ):
        # Load map configuration
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.height = cfg['height']
        self.width = cfg['width']
        raw_map = cfg['map']
        # support both string and list formats
        if isinstance(raw_map[0], str):
            self.grid = np.array([[int(c) for c in row] for row in raw_map], dtype=int)
        else:
            self.grid = np.array(raw_map, dtype=int)

        # Determine start cell
        if cfg.get('start') is not None:
            self.start_cell = tuple(cfg['start'])
        else:
            zeros = np.argwhere(self.grid == 0)
            if zeros.size == 0:
                raise ValueError("No empty cell (0) found in map.")
            self.start_cell = tuple(zeros[0])

        # Pygame setup
        self.cell_size_px = cell_size_px
        if headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        self.screen = None
        if not headless:
            screen_w = self.width * cell_size_px
            screen_h = self.height * cell_size_px
            self.screen = pygame.display.set_mode((screen_w, screen_h))
            pygame.display.set_caption("GridWorld_Discrete")
        self.clock = pygame.time.Clock()

        # Prepare rects for collision detection
        self.wall_rects = []
        self.trap_rects = []
        self.goal_rects = []
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(
                    c*cell_size_px, r*cell_size_px,
                    cell_size_px, cell_size_px
                )
                cell = self.grid[r, c]
                if cell == 1:
                    self.wall_rects.append(rect)
                elif cell == 2:
                    self.trap_rects.append(rect)
                elif cell == 3:
                    self.goal_rects.append(rect)

        # 8-connected actions: N, NE, E, SE, S, SW, W, NW
        self.action_dirs = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]

        # Agent state
        self.agent_cell: Optional[Tuple[int,int]] = None
        self.done = False

    def reset(self, start_pos: Optional[Tuple[int,int]] = None) -> np.ndarray:
        # Place agent at start cell (center)
        if start_pos is None:
            self.agent_cell = self.start_cell
        else:
            self.agent_cell = tuple(start_pos)
        self.done = False
        return self._get_obs()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")
        # Determine candidate cell
        dr, dc = self.action_dirs[action]
        r, c = self.agent_cell
        nr, nc = r + dr, c + dc
        # Check bounds and wall
        if (0 <= nr < self.height and 0 <= nc < self.width and
            self.grid[nr, nc] != 1):
            self.agent_cell = (nr, nc)
        # Determine reward and done
        cell = self.grid[self.agent_cell]
        if cell == 2:
            self.done = True
            return self._get_obs(), -100.0, True, {}
        if cell == 3:
            self.done = True
            return self._get_obs(), 100.0, True, {}
        # Normal step
        return self._get_obs(), -1.0, False, {}

    def _get_obs(self) -> np.ndarray:
        # Return agent cell coordinates
        return np.array(self.agent_cell, dtype=int)

    def render(self, tick: int = 30) -> None:
        if self.screen is None:
            return
        # Draw grid cells
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(c*self.cell_size_px, r*self.cell_size_px,
                                   self.cell_size_px, self.cell_size_px)
                cell = self.grid[r, c]
                if cell == 1:
                    color = (50,50,50)
                elif cell == 2:
                    color = (200,0,0)
                elif cell == 3:
                    color = (0,200,0)
                else:
                    color = (220,220,220)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (150,150,150), rect, 1)
        # Draw agent as circle at cell center
        r, c = self.agent_cell
        px = c*self.cell_size_px + self.cell_size_px//2
        py = r*self.cell_size_px + self.cell_size_px//2
        pygame.draw.circle(self.screen, (0,0,255), (px, py), self.cell_size_px//4)
        pygame.display.flip()
        self.clock.tick(tick)

    def close(self) -> None:
        pygame.quit()

# Example __main__
if __name__ == '__main__':
    env = GridWorldEnvDiscrete('../configs/map1.yaml')
    state = env.reset()
    print(f"Start: {state}")
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
        key = input("Action [0-7] or q: ")
        if key.lower() == 'q':
            break
        if not key.isdigit() or int(key) not in range(8):
            print("Enter 0-7 or q")
            continue
        a = int(key)
        state, reward, done, _ = env.step(a)
        print(f"A:{a} S:{state} R:{reward} D:{done}")
        env.render()
        if done:
            print("Resetting...")
            state = env.reset()
    env.close()
