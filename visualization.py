import math
import random
from collections import defaultdict

import pygame
import numpy as np

from main import run_ant_colony_optimization


WIDTH, HEIGHT = 800, 600

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


def generate_graph(n):
    graph = []
    coords = [(random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100)) for _ in range(n)]
    for start in range(1, n + 1):
        x1, y1 = coords[start - 1]
        for end in range(start, n + 1):
            if start != end:
                x2, y2 = coords[end - 1]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) // 10

                graph.append((start, end, int(distance)))

    return graph, coords


def to_matrix(graph: list[tuple[int, int, int]], n: int) -> list[list[int]]:
    matrix = [[-1] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0

    for u, v, distance in graph:
        matrix[u - 1][v - 1] = distance
        matrix[v - 1][u - 1] = distance

    return matrix


def draw_graph(screen, graph, coords, state):
    paths_count = state['paths_count']
    max_count = max(paths_count.values() or [0])
    min_count = 0
    for start, end, distance in graph:
        start_pos = coords[start - 1]
        end_pos = coords[end - 1]

        count = paths_count.get(frozenset((start, end)), 0)
        try:
            normalized = int(255 * (count - min_count) / (max_count - min_count))
            color = 255 - (normalized if normalized > 15 else 15)
        except ZeroDivisionError:
            color = 240

        pygame.draw.line(screen, (color, color, color, 255), start_pos, end_pos, 2)

        mid_x = (start_pos[0] + end_pos[0]) // 2
        mid_y = (start_pos[1] + end_pos[1]) // 2
        distance_label = pygame.font.SysFont("Arial", 14).render(f"{distance}", True, BLACK)
        screen.blit(distance_label, (mid_x, mid_y))

    for coo in coords:
        pygame.draw.circle(screen, BLACK, coo, 10)


def move_ants(screen, state, ants_state, config):
    node_num = config['nodes']
    coords = config['coords']
    for i in range(config['ants']):
        depth = int(ants_state["depth"][i])
        if depth >= config['iters']:
            continue

        progress = ants_state["progress"][i]
        speed = ants_state["speeds"][i]

        distances = config['distances']

        path = state["paths"][depth][i]

        curr_node_idx = int(ants_state["nodes_passed"][i])
        next_node_idx = curr_node_idx + 1 if curr_node_idx + 1 < node_num + 1 else 0

        curr_node = int(path[curr_node_idx])
        next_node = int(path[next_node_idx])

        if curr_node == next_node:
            ants_state['progress'][i].fill(0)
            ants_state['nodes_passed'][i] = 0
            ants_state['depth'][i] += 1

            continue

        edge_progress = progress[curr_node_idx]
        start, end = coords[curr_node - 1], coords[next_node - 1],

        edge_distance = (distances.get((curr_node, next_node))
                         or distances.get((next_node, curr_node)))

        ant_x = start[0] + (edge_progress / edge_distance) * (end[0] - start[0])
        ant_y = start[1] + (edge_progress / edge_distance) * (end[1] - start[1])

        pygame.draw.circle(screen, RED, (int(ant_x), int(ant_y)), 5)

        edge_progress += speed
        progress[curr_node_idx] = edge_progress

        if edge_progress >= edge_distance:
            state['paths_count'][frozenset((curr_node, next_node))] += 1
            ants_state["nodes_passed"][i] += 1


def init_ants_state(config):
    ants = config['ants']
    nodes = config['nodes']
    return dict(
        depth = np.zeros((ants, )),
        nodes_passed = np.zeros((ants, )),
        progress = np.zeros((ants, nodes)),
        speeds = [random.uniform(*config['speed_range']) for _ in range(ants)],
        finished = np.full(ants, False),
    )


def init_algo_state(iters):
    return dict(
        paths = [()] * iters,
        paths_count = defaultdict(int),
    )


def update_config(config, graph, coords):
    edge_distances = {(s, e): d for s, e, d in graph}
    return dict(
        iters=config['common'].getint('iters'),
        ants=config['common'].getint('ants'),
        nodes=config['common'].getint('nodes'),
        evaporation_rate=config['common'].getfloat('evaporation_rate'),
        pheromone_factor=config['common'].getfloat('pheromone_factor'),
        visibility_factor=config['common'].getfloat('visibility_factor'),
        distances=edge_distances,
        coords=coords,
        **config
    )


def run(config):
    pygame.init()
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ACO Visualization")

    graph, coords = generate_graph(config['common'].getint('nodes'))
    config = update_config(config, graph, coords)

    state = init_algo_state(config['common'].getint('iters'))
    ants_state = init_ants_state(config)

    matrix = to_matrix(graph, config['nodes'])

    iterations = run_ant_colony_optimization(
        matrix,
        config['iters'],
        config['ants'],
        config['evaporation_rate'],
        config['pheromone_factor'],
        config['visibility_factor']
    )

    iteration = 0
    path = next(iterations)
    state['paths'][iteration] = path

    running = True
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_graph(screen, graph, coords, state)
        move_ants(screen, state, ants_state, config)

        if iteration < config['iters'] - 1:
            iteration += 1
            path = next(iterations)
            state['paths'][iteration] = path

        pygame.display.flip()
        clock.tick(config['fps'])

    pygame.quit()


if __name__ == '__main__':
    import configparser
    ini = configparser.ConfigParser()
    ini.read('config.ini')

    algo_config = ini['common']
    config = dict(
        common=algo_config,
        speed_range=(ini['visual'].getfloat('ant_speed_min'), ini['visual'].getfloat('ant_speed_max')),
        fps=ini['visual'].getint('fps')
    )

    run(config)
