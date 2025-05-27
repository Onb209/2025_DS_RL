# env/gridworld_mdp.py

from env.gridworld_env import TileType, Action

class GridWorldMDP:
    def __init__(self, env):
        self.env = env
        self.height = env.height
        self.width = env.width
        self.actions = list(Action)

        # 벽이 아닌 모든 타일을 state로 정의
        self.states = [(y, x) for y in range(self.height) for x in range(self.width)
                       if env.grid[y][x] != TileType.WALL]
        # print(self.states)


    # state transtion: (state, action) → (next_state, reward, done)
    def get_transition(self, state, action):
        y, x = state
        move = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1)
        }[action]

        new_y, new_x = y + move[0], x + move[1]
        
        # 다음 위치가 맵 밖을 벗어났을 경우
        if not (0 <= new_y < self.height and 0 <= new_x < self.width):
            return state, -1, False

        tile = self.env.grid[new_y][new_x]
        if tile == TileType.WALL:
            return state, -1, False
        elif tile == TileType.TRAP:
            return (new_y, new_x), -100, True
        elif tile == TileType.GOAL:
            return (new_y, new_x), 100, True
        else:
            return (new_y, new_x), -1, False


