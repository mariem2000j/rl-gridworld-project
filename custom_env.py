# custom_env.py
import pygame
import numpy as np
import sys

class ThiefGridWorld:
    def __init__(self, size=5, render=True):
        self.size = size
        self.render_enabled = render
        self.action_space = [0, 1, 2, 3]  # Up, Down, Left, Right
        self.action_names = ['↑', '↓', '←', '→']
        self.action_to_delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.terminal_states = [(4, 4), (2, 2)]
        self.walls = {(1, 1), (3, 3)}
        self.treasure = (4, 4)
        self.trap = (2, 2)
        self.reset()

        if self.render_enabled:
            pygame.init()
            self.cell_size = 80
            self.width = self.size * self.cell_size
            self.height = self.size * self.cell_size
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Thief GridWorld - RL Project")
            self.font = pygame.font.SysFont(None, 24)
            self.clock = pygame.time.Clock()

    def reset(self):
        self.agent_pos = (0, 0)
        self.done = False
        self.steps = 0
        return self._pos_to_state(self.agent_pos)

    def _pos_to_state(self, pos):
        return pos[0] * self.size + pos[1]

    def _state_to_pos(self, state):
        return (state // self.size, state % self.size)

    def step(self, action):
        if self.done:
            return self._pos_to_state(self.agent_pos), 0, True, {}

        # 80% de chance de réussir, 20% aléatoire
        if np.random.rand() < 0.2:
            action = np.random.choice(self.action_space)

        dx, dy = self.action_to_delta[action]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        # Reste dans les limites
        new_x = np.clip(new_x, 0, self.size - 1)
        new_y = np.clip(new_y, 0, self.size - 1)

        new_pos = (new_x, new_y)

        # Mur → pas de déplacement
        if new_pos in self.walls:
            new_pos = self.agent_pos

        self.agent_pos = new_pos
        self.steps += 1

        # Récompenses
        reward = 0
        if self.agent_pos == self.treasure:
            reward = 10
            self.done = True
        elif self.agent_pos == self.trap:
            reward = -10
            self.done = True
        elif self.steps >= 100:
            self.done = True  # Timeout

        state = self._pos_to_state(self.agent_pos)
        return state, reward, self.done, {}

    def render(self):
        if not self.render_enabled:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((240, 240, 240))
        for i in range(self.size):
            for j in range(self.size):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                pos = (i, j)
                if pos == self.agent_pos:
                    color = (70, 130, 180)  # Agent: SteelBlue
                elif pos == self.treasure:
                    color = (50, 205, 50)   # Treasure: LimeGreen
                elif pos == self.trap:
                    color = (220, 20, 60)   # Trap: Crimson
                elif pos in self.walls:
                    color = (100, 100, 100) # Wall: Gray
                else:
                    color = (255, 255, 255) # Empty: White
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.render_enabled:
            pygame.quit()