import pygame
import random
import multiprocessing
import heapq

pygame.init()

WIDTH, HEIGHT = 800, 600
CELL_SIZE = 20
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Horloge pour gérer les FPS
clock = pygame.time.Clock()

def generate_food(snake):
    while True:
        x = random.randint(0, GRID_WIDTH - 1) * CELL_SIZE
        y = random.randint(0, GRID_HEIGHT - 1) * CELL_SIZE
        if (x, y) not in snake:
            return x, y
from collections import deque

def bfs_path(start, goal, snake_body, grid_width, grid_height):
    queue = deque([(start, [])])  # File d'attente : position actuelle et chemin parcouru
    visited = set()

    while queue:
        (x, y), path = queue.popleft()

        if (x, y) == goal:  # Si on atteint le fruit
            return path

        if (x, y) in visited:
            continue

        visited.add((x, y))

        # itérations sur les 4 directions
        for dx, dy in [(0, -CELL_SIZE), (0, CELL_SIZE), (-CELL_SIZE, 0), (CELL_SIZE, 0)]:
            nx, ny = x + dx, y + dy
            if (
                    0 <= nx < WIDTH and 0 <= ny < HEIGHT and  # Dans les limites du plateau
                    (nx, ny) not in snake_body and (nx, ny) not in visited
            ):
                queue.append(((nx, ny), path + [(dx, dy)]))

    return None

def calculate_free_space(start, snake_body, grid_width, grid_height):
    queue = deque([start])
    visited = set()
    free_space = 0

    while queue:
        x, y = queue.popleft()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        free_space += 1

        # itérations sur les 4 directions
        for dx, dy in [(0, -CELL_SIZE), (0, CELL_SIZE), (-CELL_SIZE, 0), (CELL_SIZE, 0)]:
            nx, ny = x + dx, y + dy
            if (
                    0 <= nx < WIDTH and 0 <= ny < HEIGHT and
                    (nx, ny) not in snake_body and (nx, ny) not in visited
            ):
                queue.append((nx, ny))

    return free_space

def simulate_future_state(snake, direction, food, grid_width, grid_height):
    new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])

    # vérifs collisions
    if (
            new_head[0] < 0 or new_head[1] < 0 or
            new_head[0] >= WIDTH or new_head[1] >= HEIGHT or
            new_head in snake
    ):
        return -1  #mouvement impossible

    simulated_snake = [new_head] + snake[:-1]

    free_space = calculate_free_space(new_head, simulated_snake, grid_width, grid_height)

    distance_to_food = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])
    proximity_score = 1 / (distance_to_food + 1)

    return free_space + proximity_score
def a_star_path(start, goal, snake_body, grid_width, grid_height):
    def heuristic(pos1, pos2):
        # Heuristique : distance de Manhattan
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    open_set = []
    heapq.heappush(open_set, (0, start, []))  # (coût estimé, position actuelle, chemin)
    visited = set()

    while open_set:
        _, current, path = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path

        x, y = current
        for dx, dy in [(0, -CELL_SIZE), (0, CELL_SIZE), (-CELL_SIZE, 0), (CELL_SIZE, 0)]:
            nx, ny = x + dx, y + dy
            if (
                    0 <= nx < WIDTH and 0 <= ny < HEIGHT and
                    (nx, ny) not in snake_body and (nx, ny) not in visited
            ):
                new_cost = len(path) + 1 + heuristic((nx, ny), goal)
                heapq.heappush(open_set, (new_cost, (nx, ny), path + [(dx, dy)]))

    return None
def run_simulation(queue, mode):

    # Initialisations
    snake = [(100, 100), (80, 100), (60, 100)]
    direction = (CELL_SIZE, 0)
    food = generate_food(snake)
    score = 0
    game_speed=150
    font = pygame.font.Font(None, 36)

    # Boucle de jeu
    running = True
    while running:
        screen.fill(BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:  # Gestion de la vitesse du jeu
                if event.key == pygame.K_UP or event.key == pygame.K_KP_PLUS:
                    game_speed += 10 if game_speed < 150 else 0
                elif event.key == pygame.K_DOWN or event.key == pygame.K_KP_MINUS:
                    game_speed -= 10 if game_speed > 20 else 0
        if mode == 1:
            path = bfs_path(snake[0], food, snake, GRID_WIDTH, GRID_HEIGHT)
        else:
            path = a_star_path(snake[0], food, snake, GRID_WIDTH, GRID_HEIGHT)
        if path:
            direction = path[0]
        else:
            # si pas de chemin évident :
            best_direction = None
            best_score = -float('inf')

            # verif 4 directions
            for dx, dy in [(0, -CELL_SIZE), (0, CELL_SIZE), (-CELL_SIZE, 0), (CELL_SIZE, 0)]:
                simulated_score = simulate_future_state(snake, (dx, dy), food, GRID_WIDTH, GRID_HEIGHT)
                if simulated_score > best_score:
                    best_score = simulated_score
                    best_direction = (dx, dy)

            if best_direction:
                direction = best_direction
            else:
                print("No valid moves. Game Over!")
                pygame.time.delay(2000)
                running = False

        # déplacements serpent
        head_x, head_y = snake[0]
        new_head = (head_x + direction[0], head_y + direction[1])

        # vérif collisions
        if (
                new_head[0] < 0 or new_head[1] < 0 or
                new_head[0] >= WIDTH or new_head[1] >= HEIGHT or
                new_head in snake
        ):
            score_text = font.render(f"Game Over ! Score final : {score}", True, RED)
            print(score)
            screen.blit(score_text, (WIDTH // 2 - 160, HEIGHT // 2 - 20))
            pygame.display.update()
            pygame.time.delay(2000)
            running = False

        snake.insert(0, new_head)

        # logique nourriture serpent
        if new_head == food:
            food = generate_food(snake)
            score += 1
        else:
            snake.pop()

        # serpent
        for segment in snake:
            pygame.draw.rect(screen, GREEN, pygame.Rect(segment[0], segment[1], CELL_SIZE, CELL_SIZE))

        # fruit
        pygame.draw.rect(screen, RED, pygame.Rect(food[0], food[1], CELL_SIZE, CELL_SIZE))

        # Affichage score et vitesse
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        speed_text = font.render(f"Speed: {game_speed}", True, WHITE)
        screen.blit(speed_text, (10, 40))

        pygame.display.flip()

        clock.tick(game_speed)

    pygame.quit()
    queue.put(score)

def run_multiple_simulations(num_simulations, mode):
    processes = []
    score_queue = multiprocessing.Queue()
    for sim_id in range(num_simulations):
        process = multiprocessing.Process(target=run_simulation, args=(score_queue, mode))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    scores = []
    while not score_queue.empty():
        scores.append(score_queue.get())
    return scores

def select_mode():
    """Affiche un menu pour sélectionner le mode de résolution."""
    print("=== Sélection du mode de résolution ===")
    print("1. BFS (Breadth-First Search)")
    print("2. A* (A Star Search)")
    print("=======================================")

    while True:
        try:
            choice = int(input("Choisissez un mode (1/2) : "))
            if choice == 1:
                print("Mode sélectionné : BFS")
                return 1
            elif choice == 2:
                print("Mode sélectionné : A*")
                return 2
            else:
                print("Veuillez entrer 1 ou 2.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer 1 ou 2.")

if __name__ == "__main__":
    mode = select_mode()
    num_simulations = 1
    scores = run_multiple_simulations(num_simulations, mode)
    print(f"Score moyen sur {num_simulations} simulations: {sum(scores) / len(scores)}")