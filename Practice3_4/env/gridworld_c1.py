import os
import numpy as np
import yaml
import pygame
from typing import Optional

class GridWorldEnv_c1:
    """
    GridWorld_c1 with precise collider handling using pygame.Rect,
    physical units separation, headless mode support, and improved radius logic.
    """
    def __init__(
        self,
        config_path: str,
        cell_size_px: int = 100,
        cell_size_m: float = 1.0,
        step_size_m: float = 0.2,
        agent_radius_px: Optional[int] = 20,
        headless: bool = False
    ):
        # ---- Load map config ----
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.width = cfg['width']
        self.height = cfg['height']
        raw_map = cfg['map']
        if isinstance(raw_map[0], str):
            self.grid = np.array([[int(c) for c in row] for row in raw_map], dtype=int)
        else:
            self.grid = np.array(raw_map, dtype=int)

        # ---- Determine start cell ----
        if 'start' in cfg and cfg['start'] is not None:
            self.start_cell = tuple(cfg['start'])
        else:
            idx = np.argwhere(self.grid == 0)
            if idx.size == 0:
                raise ValueError("No normal cells (0) found in map.")
            self.start_cell = tuple(idx[0])

        # ---- Physical and movement parameters ----
        self.cell_size_m = cell_size_m  # meters per cell
        self.step_size_m = step_size_m  # meters per step
        angles = np.deg2rad(np.arange(0, 360, 45))
        deltas_m = np.stack([np.sin(angles), np.cos(angles)], axis=1) * self.step_size_m
        self.deltas = deltas_m  # in meters

        # ---- Agent radius logic ----
        if agent_radius_px is None:
            self.agent_radius = int(cell_size_px * 0.4)
        else:
            self.agent_radius = agent_radius_px

        # ---- Pygame setup (headless support) ----
        if headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        self.cell_size_px = cell_size_px
        self.screen_w = self.width * cell_size_px
        self.screen_h = self.height * cell_size_px
        if not headless:
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("GridWorld_c1 (Discrete Action)")
        else:
            self.screen = None
        self.clock = pygame.time.Clock()

        # ---- Prepare rects for collisions (pixel coords) ----
        self.wall_rects, self.trap_rects, self.goal_rects = [], [], []
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(
                    c*cell_size_px, r*cell_size_px,
                    cell_size_px, cell_size_px
                )
                cell_type = self.grid[r, c]
                if cell_type == 1:
                    self.wall_rects.append(rect)
                elif cell_type == 2:
                    self.trap_rects.append(rect)
                elif cell_type == 3:
                    self.goal_rects.append(rect)

        # Continuous agent position in meters (row, col)
        self.agent_pos = None
        self.done = False

    def reset(self, start_pos=None):
        # Place agent at center of start cell
        if start_pos is None:
            r, c = self.start_cell
        else:
            r, c = start_pos
        # convert cell coords to meters
        self.agent_pos = np.array([r + 0.5, c + 0.5], dtype=float)
        self.done = False
        return self._get_obs()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")

        # 원래 델타
        dr, dc = self.deltas[action]
        old_pos = self.agent_pos.copy()
        delta = np.array([dr, dc])

        # 새 위치 후보
        # target = old_pos + delta

        # 이분탐색으로 t ∈ [0,1] 찾아서 최대 이동
        lo, hi = 0.0, 1.0
        for _ in range(8):
            mid = (lo + hi) / 2
            test_pos = old_pos + delta * mid
            px = test_pos[1] * self.cell_size_px
            py = test_pos[0] * self.cell_size_px
            rect = pygame.Rect(
                px - self.agent_radius, py - self.agent_radius,
                self.agent_radius * 2, self.agent_radius * 2
            )

            # 1) 벽 충돌
            wall_hit = any(rect.colliderect(w) for w in self.wall_rects)
            # 2) 맵 외곽 충돌
            out_of_bounds = (
                    rect.left < 0 or
                    rect.top < 0 or
                    rect.right > self.screen_w or
                    rect.bottom > self.screen_h
            )

            if wall_hit or out_of_bounds:
                hi = mid
            else:
                lo = mid

        # 최종 허용 위치
        self.agent_pos = old_pos + delta * lo

        # 함정/목표 충돌은 새 위치에서 그대로 체크
        px = self.agent_pos[1] * self.cell_size_px
        py = self.agent_pos[0] * self.cell_size_px
        agent_rect = pygame.Rect(px - self.agent_radius, py - self.agent_radius,
                                 self.agent_radius * 2, self.agent_radius * 2)

        if any(agent_rect.colliderect(t) for t in self.trap_rects):
            self.done = True
            return self._get_obs(), -100.0, True, {}
        if any(agent_rect.colliderect(g) for g in self.goal_rects):
            self.done = True
            return self._get_obs(), 100.0, True, {}

        return self._get_obs(), -1.0, False, {}

    def _no_move(self):
        # Stay in place with -1 reward
        return self._get_obs(), -1.0, False, {}

    def _get_obs(self):
        # Return continuous position (row, col) in meters
        return self.agent_pos.copy()

    def render(self, tick=30):
        if self.screen is None:
            return

        # Draw grid and cells (pixel coords)
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(
                    c*self.cell_size_px, r*self.cell_size_px,
                    self.cell_size_px, self.cell_size_px
                )
                cell_type = self.grid[r, c]
                if cell_type == 1:
                    color = (50, 50, 50)
                elif cell_type == 2:
                    color = (200, 0, 0)
                elif cell_type == 3:
                    color = (0, 200, 0)
                else:
                    color = (220, 220, 220)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (150, 150, 150), rect, 1)

        # Draw agent as circle
        px = self.agent_pos[1] * self.cell_size_px
        py = self.agent_pos[0] * self.cell_size_px
        pygame.draw.circle(
            self.screen, (0, 0, 255), (int(px), int(py)), self.agent_radius
        )

        pygame.display.flip()
        self.clock.tick(tick)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = GridWorldEnv_c1(
        "../configs/map2.yaml",
    )
    obs = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        action = np.random.randint(0, 8)
        obs, reward, done, _ = env.step(action)
        env.render()
        if done:
            pygame.time.delay(500)
            env.reset()
    env.close()
