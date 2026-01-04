# visualize_grid.py
import pygame
import sys

# Configuration de la grille
GRID_SIZE = 5
CELL_SIZE = 100
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE

# Définition des éléments
START = (0, 0)          # Coin supérieur gauche
TREASURE = (4, 4)       # Coin inférieur droit
TRAP = (2, 2)           # Centre
WALLS = {(1, 1), (3, 3)}

# Couleurs (R, G, B)
BACKGROUND = (240, 240, 240)
GRID_LINE = (50, 50, 50)
AGENT_COLOR = (70, 130, 180)    # SteelBlue
TREASURE_COLOR = (50, 205, 50)  # LimeGreen
TRAP_COLOR = (220, 20, 60)      # Crimson
WALL_COLOR = (100, 100, 100)    # Gray
TEXT_COLOR = (0, 0, 0)

def draw_grid(screen):
    # Fond
    screen.fill(BACKGROUND)

    # Dessiner chaque cellule
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pos = (row, col)

            # Choisir la couleur selon le contenu de la case
            if pos == START:
                color = AGENT_COLOR
                label = "Start"
            elif pos == TREASURE:
                color = TREASURE_COLOR
                label = "Treasure (+10)"
            elif pos == TRAP:
                color = TRAP_COLOR
                label = "Trap (-10)"
            elif pos in WALLS:
                color = WALL_COLOR
                label = "Wall"
            else:
                color = BACKGROUND
                label = ""

            # Dessiner la cellule
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID_LINE, rect, 2)

            # Ajouter du texte si nécessaire
            if label:
                font = pygame.font.SysFont(None, 18)
                text = font.render(label, True, TEXT_COLOR)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Thief GridWorld – Visualisation de la Grille")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_grid(screen)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()