import pygame
import numpy as np
import random
import time

# Initialize Pygame
pygame.init()

# Fonts
FONT = pygame.font.SysFont("Arial", 24)

# Configurations
CELL_SIZE = 50
MARGIN = 2
SCREEN_PADDING = 100
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Global Variables
grid_size = 0
grid = []
start = None
goal = None
obstacles = set()
q_table = None

# Pygame screen
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Q-Learning Visualizer")

# Helper to draw grid
def draw_grid(agent=None):
    screen.fill(WHITE)
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * (CELL_SIZE + MARGIN) + SCREEN_PADDING
            y = row * (CELL_SIZE + MARGIN) + SCREEN_PADDING
            color = WHITE
            if (row, col) in obstacles:
                color = BLACK
            elif (row, col) == goal:
                color = GREEN
            elif (row, col) == start:
                color = BLUE
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))

    if agent:
        x = agent[1] * (CELL_SIZE + MARGIN) + SCREEN_PADDING
        y = agent[0] * (CELL_SIZE + MARGIN) + SCREEN_PADDING
        pygame.draw.circle(screen, RED, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 3)

    pygame.display.update()

# Q-learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.3
episodes = 200

actions = ['up', 'down', 'left', 'right']
action_map = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Reward function
def get_reward(pos):
    if pos == goal:
        return 100
    elif pos in obstacles:
        return -100
    else:
        return -1

# Take action
def take_action(pos, action):
    row, col = pos
    dr, dc = action_map[action]
    new_row = max(0, min(grid_size - 1, row + dr))
    new_col = max(0, min(grid_size - 1, col + dc))
    return new_row, new_col

# Q-learning algorithm with animation
def train_q_learning():
    global q_table
    q_table = np.zeros((grid_size, grid_size, len(actions)))

    for ep in range(1, episodes + 1):
        pos = start
        draw_grid(pos)
        pygame.display.set_caption(f"Training Episode: {ep}/{episodes}")
        time.sleep(0.2)
        while pos != goal:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if random.uniform(0, 1) < epsilon:
                a_idx = random.randint(0, 3)
            else:
                a_idx = np.argmax(q_table[pos[0]][pos[1]])

            action = actions[a_idx]
            new_pos = take_action(pos, action)
            reward = get_reward(new_pos)
            best_next = np.max(q_table[new_pos[0]][new_pos[1]])

            q_table[pos[0]][pos[1]][a_idx] = (1 - alpha) * q_table[pos[0]][pos[1]][a_idx] + \
                                            alpha * (reward + gamma * best_next)

            pos = new_pos
            draw_grid(pos)

            if pos in obstacles:
                time.sleep(0.5)
                pos = start  # Reboot
                break

            time.sleep(0.05)

# Setup function to select grid, goal, obstacles
def setup_grid():
    global grid_size, start, goal, obstacles
    selecting = True
    input_mode = 'grid_size'
    user_text = ''

    while selecting:
        screen.fill(WHITE)
        txt = "Enter Grid Size (e.g., 5): " if input_mode == 'grid_size' else \
              "Click to place START (Blue), then GOAL (Green), then Obstacles (Black). Press Enter to start."
        label = FONT.render(txt + user_text, True, BLACK)
        screen.blit(label, (50, 50))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if input_mode == 'grid_size':
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        try:
                            grid_size = int(user_text)
                            input_mode = 'placement'
                            screen.fill(WHITE)
                            draw_grid()
                        except:
                            user_text = ''
                    elif event.key == pygame.K_BACKSPACE:
                        user_text = user_text[:-1]
                    else:
                        user_text += event.unicode

            elif input_mode == 'placement':
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    col = (x - SCREEN_PADDING) // (CELL_SIZE + MARGIN)
                    row = (y - SCREEN_PADDING) // (CELL_SIZE + MARGIN)
                    if 0 <= row < grid_size and 0 <= col < grid_size:
                        if start is None:
                            start = (row, col)
                        elif goal is None:
                            goal = (row, col)
                        else:
                            obstacles.add((row, col))
                    draw_grid()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN and start and goal:
                        selecting = False

if __name__ == '__main__':
    setup_grid()
    train_q_learning()
    pygame.quit()
