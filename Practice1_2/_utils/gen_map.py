import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
import json
import os
from env.gridworld_env import GridWorldEnv, TileType

def extract_map_data(env):
    data = {
        "width": env.width,
        "height": env.height,
        "walls": [],
        "traps": [],
        "goal": []
    }
    for y in range(env.height):
        for x in range(env.width):
            tile = env.grid[y][x]
            if tile == TileType.WALL:
                data["walls"].append([y, x])
            elif tile == TileType.TRAP:
                data["traps"].append([y, x])
            elif tile == TileType.GOAL:
                data["goal"] = [y, x]
    return data

def save_map_to_file(data, seed):
    os.makedirs("maps", exist_ok=True)
    filename = f"env/maps/saved_map_{seed}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Map saved to {filename}")

def create_env_with_seed(seed, width=15, height=15):
    import random
    random.seed(seed)
    env = GridWorldEnv(width=width, height=height)
    return env

def main():
    pygame.init()
    seed = 0
    env = create_env_with_seed(seed)
    clock = pygame.time.Clock()
    running = True

    while running:
        env.render()
        print(f"Seed: {seed}", end="\r")
        clock.tick(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    seed += 1
                    env = create_env_with_seed(seed)
                elif event.key == pygame.K_LEFT:
                    seed = max(0, seed - 1)
                    env = create_env_with_seed(seed)
                elif event.key == pygame.K_s:
                    map_data = extract_map_data(env)
                    save_map_to_file(map_data, seed)
                elif event.key == pygame.K_q:
                    running = False

    pygame.quit()

if __name__ == "__main__":
    main()
