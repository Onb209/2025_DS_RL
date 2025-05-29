import os
import numpy as np
import yaml
import pygame
from typing import Optional, Union, Sequence
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv_c2:
    """
    GridWorld_c2 environment with ray-based distance sensors in 8 directions and ray visualization,
    treating out-of-bounds as walls.
    """
    def __init__(
        self,
        config_path: str,
        cell_size_px: int = 100,
        cell_size_m: float = 1.0,
        step_size_m: float = 1.0,
        agent_radius_px: Optional[int] = 20,
        headless: bool = False,
        ray_length_m: float = 1.0,
        ray_num: int = 8
    ):
        # ---- Load and parse map config ----
        with open(config_path, 'r', encoding='utf8') as f:
            cfg = yaml.safe_load(f)
        self.width = cfg['width']
        self.height = cfg['height']
        raw_map = cfg['map']
        if isinstance(raw_map[0], str):
            self.grid = np.array([[int(c) for c in row] for row in raw_map], dtype=int)
        else:
            self.grid = np.array(raw_map, dtype=int)

        # ---- Determine start cell ----
        if cfg.get('start') is not None:
            self.start_cell = tuple(cfg['start'])
        else:
            zeros = np.argwhere(self.grid == 0)
            if zeros.size == 0:
                raise ValueError("No empty cell (0) in map to start.")
            self.start_cell = tuple(zeros[0])

        # ---- Physical parameters ----
        self.cell_size_m = cell_size_m
        self.step_size_m = step_size_m
        self.ray_length_m = ray_length_m
        self.ray_num = ray_num
        self._init_ray_dirs()

        # ---- Pygame / rendering setup ----
        if headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        self.cell_size_px = cell_size_px
        self.agent_radius = agent_radius_px if agent_radius_px is not None else int(cell_size_px * 0.4)

        screen_w = self.width * cell_size_px
        screen_h = self.height * cell_size_px
        if not headless:
            self.screen = pygame.display.set_mode((screen_w, screen_h))
            pygame.display.set_caption("GridWorld_c2 (Continuous Action)")
        else:
            self.screen = None
        self.clock = pygame.time.Clock()

        # ---- Collision rects (pixel coords) ----
        self.wall_rects = []
        self.trap_rects = []
        self.goal_rects = []
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(
                    c * cell_size_px, r * cell_size_px,
                    cell_size_px, cell_size_px
                )
                t = self.grid[r, c]
                if t == 1:
                    self.wall_rects.append(rect)
                elif t == 2:
                    self.trap_rects.append(rect)
                elif t == 3:
                    self.goal_rects.append(rect)

        # ---- Define Gymnasium spaces ----
        obs_low = np.array([0.0, 0.0] + [0.0] * self.ray_num, dtype=np.float32)
        obs_high = np.array([
            self.height * self.cell_size_m,
            self.width * self.cell_size_m
        ] + [self.ray_length_m] * self.ray_num, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # ---- Agent state ----
        self.agent_pos = None  # [row_m, col_m]
        self.done = False

    def _init_ray_dirs(self):
        """Precompute unit vectors for evenly spaced directions"""
        angles = np.linspace(0, 2 * np.pi, self.ray_num, endpoint=False)
        self.ray_dirs = np.stack([np.sin(angles), np.cos(angles)], axis=1)

    def _to_px(self, pos_m: Sequence[float]) -> tuple[int, int]:
        c_px = pos_m[1] / self.cell_size_m * self.cell_size_px
        r_px = pos_m[0] / self.cell_size_m * self.cell_size_px
        return int(c_px), int(r_px)

    def reset(self, start_pos: Optional[Sequence[Union[int, float]]] = None, seed=None) -> np.ndarray:
        if start_pos is None:
            r, c = self.start_cell
        else:
            r, c = start_pos
        self.agent_pos = np.array([r + 0.5, c + 0.5], dtype=float) * self.cell_size_m
        self.done = False
        return self._get_obs()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")

        delta = np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * self.step_size_m
        old_pos = self.agent_pos.copy()

        lo, hi = 0.0, 1.0
        screen_w = self.width * self.cell_size_px
        screen_h = self.height * self.cell_size_px
        for _ in range(8):
            mid = (lo + hi) / 2
            test_pos = old_pos + delta * mid
            px, py = self._to_px(test_pos)
            rect = pygame.Rect(
                px - self.agent_radius, py - self.agent_radius,
                self.agent_radius * 2, self.agent_radius * 2
            )
            wall_hit = any(rect.colliderect(w) for w in self.wall_rects)
            out_of_bounds = (
                rect.left < 0 or rect.top < 0 or
                rect.right > screen_w or rect.bottom > screen_h
            )
            if wall_hit or out_of_bounds:
                hi = mid
            else:
                lo = mid
        self.agent_pos = old_pos + delta * lo

        px, py = self._to_px(self.agent_pos)
        agent_rect = pygame.Rect(
            px - self.agent_radius, py - self.agent_radius,
            self.agent_radius * 2, self.agent_radius * 2
        )
        if any(agent_rect.colliderect(t) for t in self.trap_rects):
            self.done = True
            return self._get_obs(), -100.0, True, {}
        if any(agent_rect.colliderect(g) for g in self.goal_rects):
            self.done = True
            return self._get_obs(), 100.0, True, {}

        return self._get_obs(), -1.0, False, {}

    def _cast_rays(self) -> np.ndarray:
        """Cast rays and treat out-of-bounds as walls"""
        dists = []
        res = self.cell_size_m / 10.0
        for dir_vec in self.ray_dirs:
            dist = 0.0
            while dist < self.ray_length_m:
                sample = self.agent_pos + dir_vec * dist
                # check out-of-bounds in meters
                if not (0 <= sample[0] <= self.height * self.cell_size_m and 0 <= sample[1] <= self.width * self.cell_size_m):
                    break
                px, py = self._to_px(sample)
                rect = pygame.Rect(
                    px - self.agent_radius, py - self.agent_radius,
                    self.agent_radius * 2, self.agent_radius * 2
                )
                if any(rect.colliderect(w) for w in self.wall_rects):
                    break
                dist += res
            dists.append(min(dist, self.ray_length_m))
        return np.array(dists, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        ray_obs = self._cast_rays()
        return np.concatenate((self.agent_pos.copy(), ray_obs))

    def render(self, tick: int = 30) -> None:
        if self.screen is None:
            return
        # draw cells
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(
                    c*self.cell_size_px, r*self.cell_size_px,
                    self.cell_size_px, self.cell_size_px
                )
                t = self.grid[r, c]
                if t == 1:
                    color = (50,50,50)
                elif t == 2:
                    color = (200,0,0)
                elif t == 3:
                    color = (0,200,0)
                else:
                    color = (220,220,220)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (150,150,150), rect, 1)
        # draw agent
        px, py = self._to_px(self.agent_pos)
        pygame.draw.circle(self.screen, (0,0,255), (px, py), self.agent_radius)
        # draw rays
        dists = self._cast_rays()
        for dir_vec, dist in zip(self.ray_dirs, dists):
            end_pos_m = self.agent_pos + dir_vec * dist
            end_px = self._to_px(end_pos_m)
            color = (0, 0, 255) if dist >= self.ray_length_m else (255, 0, 0)
            pygame.draw.line(self.screen, color, (px, py), end_px, 2)
        pygame.display.flip()
        self.clock.tick(tick)

    def close(self) -> None:
        pygame.quit()

if __name__ == '__main__':
    config_path = '../configs/map0.yaml'
    env = GridWorldEnv_c2(config_path)
    obs = env.reset()
    done = False
    try:
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render()
            pygame.time.delay(1000)
            print(f'Obs: {obs}, Reward: {reward}, Done: {done}')
    except KeyboardInterrupt:
        pass
    finally:
        env.close()