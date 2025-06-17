import pygame
import numpy as np
import random
import time

# --- Configuration ---
GRID_SIZE = 5
CELL_SIZE = 100
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 10

# Colors
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 255)

# Grid layout: S=start, G=goal, X=obstacle
grid = [
    ['S', '.', '.', '.', '.'],
    ['.', 'X', 'X', '.', '.'],
    ['.', '.', '.', 'X', '.'],
    ['.', 'X', '.', '.', '.'],
    ['.', '.', '.', 'X', 'G']
]

# RL setup
actions = ['up', 'down', 'left', 'right']
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(actions)))

alpha = 0.1
gamma = 0.9
epsilon = 0.3
episodes = 500

# Reward function
def get_reward(row, col):
    if grid[row][col] == 'G':
        return 100
    elif grid[row][col] == 'X':
        return -100
    else:
        return -1

# Take action
def take_action(row, col, action):
    if action == 'up':
        row = max(0, row - 1)
    elif action == 'down':
        row = min(GRID_SIZE - 1, row + 1)
    elif action == 'left':
        col = max(0, col - 1)
    elif action == 'right':
        col = min(GRID_SIZE - 1, col + 1)
    return row, col

# Train Q-learning agent
def train_q_learning():
    for ep in range(episodes):
        row, col = 0, 0
        while grid[row][col] != 'G':
            if random.uniform(0, 1) < epsilon:
                a_idx = random.randint(0, 3)
            else:
                a_idx = np.argmax(q_table[row, col])

            new_row, new_col = take_action(row, col, actions[a_idx])
            reward = get_reward(new_row, new_col)
            best_next = np.max(q_table[new_row, new_col])

            q_table[row, col, a_idx] = (1 - alpha) * q_table[row, col, a_idx] + \
                                       alpha * (reward + gamma * best_next)

            if grid[new_row][new_col] == 'X':
                break
            row, col = new_row, new_col

# --- Pygame setup ---
def draw_grid(win, agent_pos):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x, y = col * CELL_SIZE, row * CELL_SIZE
            cell = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            if grid[row][col] == 'X':
                color = BLACK
            elif grid[row][col] == 'G':
                color = GREEN
            elif grid[row][col] == 'S':
                color = BLUE
            else:
                color = WHITE
            pygame.draw.rect(win, color, cell)
            pygame.draw.rect(win, GRAY, cell, 2)

    # Draw agent
    x = agent_pos[1] * CELL_SIZE + CELL_SIZE // 4
    y = agent_pos[0] * CELL_SIZE + CELL_SIZE // 4
    pygame.draw.circle(win, RED, (x + 25, y + 25), 20)

# --- Run Game ---
def run_game():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Q-Learning Grid World")
    clock = pygame.time.Clock()

    agent_pos = [0, 0]
    running = True

    while running:
        clock.tick(FPS)

        # Exit handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        win.fill(WHITE)
        draw_grid(win, agent_pos)
        pygame.display.update()

        row, col = agent_pos
        action_idx = np.argmax(q_table[row][col])
        next_row, next_col = take_action(row, col, actions[action_idx])
        agent_pos = [next_row, next_col]

        if grid[next_row][next_col] in ['G', 'X']:
            time.sleep(1)
            agent_pos = [0, 0]  # Reset to start

# --- Main ---
if __name__ == "__main__":
    train_q_learning()
    run_game()
    pygame.quit()
