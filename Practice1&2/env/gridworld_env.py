# env/gridworld_env.py
import pygame
import numpy as np
import random
from enum import Enum
from PIL import Image
import json

# 타일 종류 정의
class TileType(Enum):
    NORMAL = 0
    WALL = 1
    TRAP = 2
    GOAL = 3

# 에이전트의 행동 정의
class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

# 맵 JSON 파일 로드
def load_map_from_file(file_name):
    file_path = 'env/maps/' + file_name
    print(file_path)
    with open(file_path, "r") as f:
        data = json.load(f)

    width = data["width"]
    height = data["height"]
    grid = np.full((height, width), TileType.NORMAL)

    for y, x in data.get("walls", []):
        grid[y][x] = TileType.WALL
    for y, x in data.get("traps", []):
        grid[y][x] = TileType.TRAP
    goal_y, goal_x = data["goal"]
    grid[goal_y][goal_x] = TileType.GOAL

    return width, height, grid

class GridWorldEnv:
    def __init__(self, width=6, height=6, cell_size=64, map_file=None):
        if map_file:
            width, height, self.grid = load_map_from_file(map_file)
        else:
            width = max(5, min(width, 15))
            height = max(5, min(height, 15))
            self.grid = np.full((height, width), TileType.NORMAL)

        self.width = width
        self.height = height
        self.agent_pos = [0, 0] # 에이전트 초기 위치
        self.done = False
        self.cell_size = cell_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.width * cell_size, self.height * cell_size))
        pygame.display.set_caption("GridWorld-d")

        self.tileset = pygame.image.load("assets/tileset.png").convert_alpha()
        self._load_tile_images()
        self._load_agent_gif("assets/agent.gif")

        if not map_file:
            self._place_special_tiles()

        self.frame_counter = 0
        self.current_agent_frame = 0

    def _load_tile_images(self):
        def tile(x, y):
            TILE_SIZE = 16
            surface = self.tileset.subsurface(pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
            return pygame.transform.scale(surface, (self.cell_size, self.cell_size))
        self.images = {
            TileType.NORMAL: tile(4, 8),
            TileType.WALL: tile(11, 12),
            TileType.TRAP: tile(3, 13),
            TileType.GOAL: tile(2, 12)
        }

    def _load_agent_gif(self, gif_path):
        gif = Image.open(gif_path)
        self.agent_frames = []
        i = 0
        while True:
            gif.seek(i)
            frame = gif.convert("RGBA")
            frame_path = f"assets/agent_frame_{i}.png"
            frame.save(frame_path)
            img = pygame.image.load(frame_path).convert_alpha()
            img = pygame.transform.scale(img, (self.cell_size, self.cell_size))
            self.agent_frames.append(img)
            i += 1
            if i == gif.n_frames:
                break

    def _place_special_tiles(self): 
        # random.seed(40)
        num_walls = (self.width * self.height) // 5
        num_traps = (self.width * self.height) // 10
        positions = [(i, j) for i in range(self.height) for j in range(self.width)]
        
        # 시작 위치와 goal 위치는 제외
        positions.remove((0, 0))
        goal_pos = (self.height - 1, self.width - 1)
        if goal_pos in positions:
            positions.remove(goal_pos)
        
        random.shuffle(positions)

        for _ in range(num_walls):
            y, x = positions.pop()
            self.grid[y][x] = TileType.WALL
        for _ in range(num_traps):
            y, x = positions.pop()
            self.grid[y][x] = TileType.TRAP

        # goal을 항상 오른쪽 하단에 배치
        self.grid[goal_pos[0]][goal_pos[1]] = TileType.GOAL

    # 초기화
    def reset(self):
        self.agent_pos = [0, 0]
        self.done = False
        return self.agent_pos

    # action 수행 후 pos, reward, done return
    def step(self, action):
        if self.done:
            return self.agent_pos, 0, self.done

        move = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1)
        }[Action(action)]

        new_pos = [self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]]
        if 0 <= new_pos[0] < self.height and 0 <= new_pos[1] < self.width:
            tile = self.grid[new_pos[0]][new_pos[1]]
            if tile != TileType.WALL:
                self.agent_pos = new_pos

        tile = self.grid[self.agent_pos[0]][self.agent_pos[1]]
        reward = -1
        if tile == TileType.TRAP:
            reward = -100
            self.done = True
        elif tile == TileType.GOAL:
            reward = 100
            self.done = True
        return self.agent_pos, reward, self.done
    
    # 렌더링 함수
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        for y in range(self.height):
            for x in range(self.width):
                tile = self.grid[y][x]
                self.screen.blit(self.images[TileType.NORMAL], (x * self.cell_size, y * self.cell_size))
                if tile != TileType.NORMAL:
                    self.screen.blit(self.images[tile], (x * self.cell_size, y * self.cell_size))

        self.frame_counter += 1
        if self.frame_counter % 10 == 0:
            self.current_agent_frame = (self.current_agent_frame + 1) % len(self.agent_frames)

        x, y = self.agent_pos[1], self.agent_pos[0]
        self.screen.blit(self.agent_frames[self.current_agent_frame], (x * self.cell_size, y * self.cell_size))

        pygame.display.flip()
