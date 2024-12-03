from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import numpy as np


@dataclass
class ACOParameters:
    """Class for storing parameters for ACO algorithm"""
    graph: list[list[int]]
    num_iterations: int
    num_ants: int
    evaporation_rate: float
    pheromone_factor: float
    visibility_factor: float

def read_graph_file(filename: str) -> list[list[int]]:
    """
    Reads a graph.csv from a file and converts it to an adjacency matrix.

    The input file should contain edges of the graph.csv, one per line, formatted as:
    u,v,weight
    where:
    - u and v are integers representing the connected nodes (0-indexed),
    - weight is an integer representing the weight of the edge.

    Args:
        filename (str): The path to the file containing the graph.csv data.

    Returns:
        list[list[int]]: A 2D adjacency matrix representing the graph.csv.

    Examples:
    >>> import tempfile
    >>> csv_data = "0,1,5\\n0,2,3\\n1,3,2\\n2,3,4\\n"
    >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding="utf-8") as tmpfile:
    ...     _ = tmpfile.write(csv_data)
    >>> read_graph_file(tmpfile.name)
    [[0, 5, 3, 0], [5, 0, 0, 2], [3, 0, 0, 4], [0, 2, 4, 0]]
    >>> csv_data = "0,7,5\\n0,2,3\\n1,3,2\\n2,3,4\\n"
    >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding="utf-8") as tmpfile:
    ...     _ = tmpfile.write(csv_data)
    >>> read_graph_file(tmpfile.name)
    [[0, 0, 3, 0, 5], [0, 0, 0, 2, 0], [3, 0, 0, 4, 0], [0, 2, 4, 0, 0], [5, 0, 0, 0, 0]]
    """
    edges = {}
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            u, v, weight = map(int, row)
            if u not in edges:
                edges[u] = set()
            if v not in edges:
                edges[v] = set()
            edges[u].add((v, weight))
            edges[v].add((u, weight))

    nodes = sorted(edges.keys())
    node_index = {node: i for i, node in enumerate(nodes)}

    n = len(nodes)
    graph = [[-1] * n for _ in range(n)]
    for i in range(n):
        graph[i][i] = 0

    for u, connections in edges.items():
        u_idx = node_index[u]
        for v, weight in connections:
            v_idx = node_index[v]
            graph[u_idx][v_idx] = weight
            graph[v_idx][u_idx] = weight

    return graph


def write_graph_file(filename: str, graph: list[list[int]]):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Result"])
        writer.writerow([graph])


def is_complete(graph: list[list[int]], start=0) -> bool:
    """
    Tests whether the given graph.csv is a complete unweighted graph.csv.

    A complete graph.csv is a graph.csv in which every vertex is connected to all other vertices,
    except herself. In the adjacency matrix of such a graph.csv:
    - there must be zeros on the diagonal (no loops)
    - all other elements must be positive numbers (there are edges between all vertices)
    - the matrix must be square

    Parameters:
        graph (list[list[int]]): Graph adjacency matrix
        start (int): Start vertex (default 0)

    Returns:
        bool: True if the graph.csv is complete, False otherwise
    >>> graph = [[0,10,12,11,14],[10,0,13,15,8],[12,13,0,9,14],[11,15,9,0,16],[14,8,14,16,0]]
    >>> is_complete(graph)
    True
    """
    n = len(graph)
    if not all(len(row) == n for row in graph):
        return False

    for i in range(n):
        for j in range(n):
            # Diagonal elements check
            if i == j and graph[i][j] != 0:
                return False
            # Non-diagonal elements check
            if i != j and graph[i][j] <= 0:
                return False

    return True

def initialize_parameters(params):
    graph = np.array(params.graph)
    node_num = len(graph)
    visibility = np.divide(1, graph, where=graph != 0)
    pheromone = 0.1 * np.ones((params.num_ants, node_num))
    path = np.ones((params.num_ants, node_num + 1))
    return graph, node_num, visibility, pheromone, path

def is_safe(v: int, graph: list[list[int]], path: list[int], pos: int) -> bool:
    """
    Checks if the vertex `v` can be added to the Hamiltonian path at position `pos`.

    Args:
        v (int): The vertex to be checked for inclusion in the path.
        graph (list[list[int]]): The adjacency matrix of the graph, where 
            graph[i][j] != 0 indicates an edge between vertices `i` and `j`, and 0 otherwise.
        path (list[int]): The current Hamiltonian path under construction, where 
            each element is a vertex in the order of traversal.
        pos (int): The current position in the path where the vertex `v` is to be placed.

    Returns:
        bool: 
            - True if the vertex `v` can be added to the path at position `pos`.
            - False if adding the vertex violates the conditions of the Hamiltonian path.

    Conditions checked:
        1. There must be an edge between the previously added vertex (`path[pos - 1]`) and `v`.
        2. The vertex `v` should not already exist in the current path.
    
    >>> graph = [[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]]
    >>> path = [0, -1, -1, -1]
    >>> is_safe(1, graph, path, 1)
    True
    >>> is_safe(2, graph, path, 1)
    False
    >>> path = [0, 1, -1, -1]
    >>> is_safe(1, graph, path, 2)
    False
    """
    if graph[path[pos - 1]][v] == 0:
        return False

    if v in path:
        return False

    return True


def ham_cycle_util(graph: list[list[int]], path: list[int], pos: int) -> bool:
    """
    This function attempts to add vertices to the path, one by one, while checking 
    if the conditions of a Hamiltonian cycle are satisfied (i.e., each vertex is 
    visited exactly once, and there is an edge between consecutive vertices).

    Args:
        graph (list[list[int]]): Adjacency matrix of the graph.
        path (list[int]): Current path being constructed.
        pos (int): Current position in the path to which the next vertex is to be added.

    Returns:
        bool: 
            - True if a Hamiltonian cycle is found.
            - False if no Hamiltonian cycle exists starting from the current position.

    >>> graph = [[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]]
    >>> path = [0, -1, -1, -1]
    >>> ham_cycle_util(graph, path, 1)
    True

    >>> graph = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> path = [0, -1, -1, -1]
    >>> ham_cycle_util(graph, path, 1)
    False
    """
    if pos == len(graph):
        return graph[path[pos - 1]][path[0]] != 0

    for v in range(1, len(graph)):
        if is_safe(v, graph, path, pos):
            path[pos] = v

            if ham_cycle_util(graph, path, pos + 1):
                return True

            path[pos] = -1

    return False


def is_ham_cycle(graph: list[list[int]]) -> bool:
    """
    Checks if a Hamiltonian cycle exists in the given graph.

    Args:
        graph (list[list[int]]): Adjacency matrix of the graph.

    Returns:
        bool: True if a Hamiltonian cycle exists, False otherwise.

    >>> is_ham_cycle([[0, 1, 0, 1, 0], [1, 0, 1, 1, 1], [0, 1, 0, 0, 1], [1, 1, 0, 0, 1], \
[0, 1, 1, 1, 0]])
    True

    >>> is_ham_cycle([[1, 0, 1, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 1, 1], [1, 0, 1, 1, 1], \
[1, 0, 1, 1, 1]])
    False
    """
    path = [-1] * len(graph)
    path[0] = 0

    return ham_cycle_util(graph, path, 1)


def main(graph_file, start_node, output_file):
    print(f"Processing graph.csv: {graph_file}")
    print(f"Starting node: {start_node}")
    print(f"Output file: {output_file}")

def run_iteration(graph, visibility, pheromone, params, node_num, path):
    for i in range(params.num_ants):
        local_visibility = np.array(visibility)
        for j in range(node_num - 1):
            curr_node = int(path[i, j] - 1)
            local_visibility[:, curr_node] = 0

            pheromone_characteristic = np.power(pheromone[curr_node, :], params.pheromone_factor)
            visibility_characteristic = np.power(local_visibility[curr_node, :], params.visibility_factor)
            characteristic = pheromone_characteristic * visibility_characteristic

            probabilistic_sum = np.cumsum(characteristic / np.sum(characteristic))
            r = np.random.random_sample()
            path[i, j + 1] = np.nonzero(probabilistic_sum > r)[0][0] + 1

        end_node = list(set(range(1, node_num + 1)) - set(path[i, :-2]))[0]
        path[i, -2] = end_node
    return path


def worker_run_iteration(args):
    graph, visibility, pheromone, params, node_num, path = args
    return run_iteration(graph, visibility, pheromone, params, node_num, path)


def update_pheromones(pheromone, path, graph, params, node_num):
    pheromone = (1 - params.evaporation_rate) * pheromone
    tour_total_distance = np.zeros((params.num_ants, 1))

    for i in range(params.num_ants):
        distance = 0
        for j in range(node_num - 1):
            distance += graph[int(path[i, j]) - 1, int(path[i, j + 1]) - 1]

        tour_total_distance[i] = distance
        for j in range(node_num - 1):
            dt = 1 / distance
            pheromone[int(path[i, j]) - 1, int(path[i, j + 1]) - 1] += dt

    return pheromone, tour_total_distance

def find_best_path(path, tour_total_distance):
    dist_min_idx = np.argmin(tour_total_distance)
    best_path = path[dist_min_idx, :]
    best_distance = tour_total_distance[dist_min_idx]
    return best_path, best_distance


def run_ant_colony_optimization(params):
    graph, node_num, visibility, pheromone, path = initialize_parameters(params)
    best_path = None
    best_distance = float('inf')

    with ProcessPoolExecutor(max_workers=4) as executor:
        for _ in range(params.num_iterations // 4):
            futures = [
                executor.submit(worker_run_iteration, (graph, visibility, pheromone, params, node_num, path))
                for _ in range(4)
            ]

            for future in as_completed(futures):
                local_path = future.result()
                pheromone, tour_total_distance = update_pheromones(pheromone, local_path, graph, params, node_num)
                current_best_path, current_best_distance = find_best_path(local_path, tour_total_distance)

                if current_best_distance < best_distance:
                    best_path, best_distance = current_best_path, current_best_distance

        best_distance += graph[int(best_path[-2]) - 1, 0]
    return best_path, int(best_distance)

def launch(config):
    graph = read_graph_file(config['infile'])
    if not is_complete(graph):
        return None

    gen = run_ant_colony_optimization(
        graph,
        config.getint('iters'),
        config.getint('ants'),
        config.getfloat('evaporation_rate'),
        config.getfloat('pheromone_factor'),
        config.getfloat('visibility_factor')
    )
    try:
        while True:
            next(gen)
    except StopIteration as e:
        path, distance = e.value

    return path, distance

if __name__ == "__main__":
    import configparser

    config = configparser.ConfigParser()
    config.read('config.ini')

    result = launch(config['common'])
    if result is None:
        print("Graph is not suitable for ACO")
    else:
        path, distance = result
        print(f"Path: {path}")
        print(f"Distance: {distance}")
