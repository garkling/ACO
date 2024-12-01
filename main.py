from dataclasses import dataclass
import csv
import argparse
import numpy as np
import networkx as nx  # graph.csv builder library


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
    [[0, 5, 3, -1], [5, 0, -1, 2], [3, -1, 0, 4], [-1, 2, 4, 0]]
    >>> csv_data = "0,7,5\\n0,2,3\\n1,3,2\\n2,3,4\\n"
    >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding="utf-8") as tmpfile:
    ...     _ = tmpfile.write(csv_data)
    >>> read_graph_file(tmpfile.name)
    [[0, -1, 3, -1, 5], [-1, 0, -1, 2, -1], [3, -1, 0, 4, -1], [-1, 2, 4, 0, -1], [5, -1, -1, -1, 0]]
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
    >>> graph.csv = [[0,10,12,11,14],[10,0,13,15,8],[12,13,0,9,14],[11,15,9,0,16],[14,8,14,16,0]]
    >>> is_complete(graph.csv)
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


def main(graph_file, start_node, output_file):
    print(f"Processing graph.csv: {graph_file}")
    print(f"Starting node: {start_node}")
    print(f"Output file: {output_file}")

    graph = read_graph_file(graph_file)
    write_graph_file(output_file, graph)


def run_ant_colony_optimization(params: ACOParameters):
    """
    Implements the ant colony algorithm for solving the traveling salesman problem.

    The algorithm simulates the behavior of ants to find the shortest path in the graph.csv.
    Ants leave pheromones on the paths they take. Stronger pheromone trails
    shorter paths attract more ants, leading to route optimization.

    Parameters:
        - graph.csv (list[list[int]]): Matrix of distances between cities (graph.csv nodes)
        - num_iterations (int): Number of algorithm iterations
        - num_ants (int): Number of ants in the colony
        - evaporation_rate (float): Pheromone evaporation rate (0 < rate < 1)
        - pheromone_factor (float): Weighting factor of pheromone influence
        - visibility_factor (float): Visibility weighting factor (inverse distance)

    Returns:
        tuple: (best_path, path_cost) where:
            - best_path (numpy.ndarray): The sequence of vertices of the optimal route
            - path_cost (int): Total length of the optimal route
    """

    graph = np.array(params.graph)
    node_num = len(graph)

    # visibility calculation of each node - visibility(i,j)=1/d(i,j)
    visibility = np.divide(1, graph, where=graph != 0)
    # pheromone matrix initialization
    pheromone = 0.1 * np.ones((params.num_ants, node_num))
    # path matrix initialization with shape (num_ants, node_count + 1) (+ 1 node because we want to return back to the start)
    path = np.ones((params.num_ants, node_num + 1))

    best_path = np.zeros(node_num)
    dist_min_distance = np.zeros(node_num)
    for _ in range(params.num_iterations):
        # ensure all ants start with 1-st node
        path[:, 0] = 1
        for i in range(params.num_ants):
            local_visibility = np.array(visibility)
            for j in range(node_num - 1):
                curr_node = int(path[i, j] - 1)
                # set current node visibility to 0
                local_visibility[:, curr_node] = 0

                pheromone_characteristic = np.power(pheromone[curr_node, :], params.pheromone_factor)
                visibility_characteristic = np.power(local_visibility[curr_node, :], params.evaporation_rate)

                # conversion from 1D to 2D matrix
                pheromone_characteristic = pheromone_characteristic[:, np.newaxis]
                visibility_characteristic = visibility_characteristic[:, np.newaxis]

                characteristic = np.multiply(pheromone_characteristic, visibility_characteristic)
                # calculating probabilistic intervals from 1 to 0 for given characteristic
                probabilistic_sum = np.cumsum(characteristic / np.sum(characteristic))

                r = np.random.random_sample()
                path[i, j + 1] = np.nonzero(probabilistic_sum > r)[0][0] + 1

            # search the last not visited node by exclusion
            end_node = list(set(range(1, node_num + 1)) - set(path[i, :-2]))[0]
            path[i, -2] = end_node

        optimized_path = np.array(path)
        tour_total_distance = np.zeros((params.num_ants, 1))
        for i in range(params.num_ants):
            distance = 0
            for j in range(node_num - 1):
                distance = distance + graph[int(optimized_path[i, j]) - 1, int(optimized_path[i, j + 1]) - 1]

            tour_total_distance[i] = distance

        # get iteration best path
        dist_min_idx = np.argmin(tour_total_distance)
        dist_min_distance = tour_total_distance[dist_min_idx]
        best_path = path[dist_min_idx, :]

        # adjust pheromones
        pheromone = (1 - params.evaporation_rate) * pheromone
        for i in range(params.num_ants):
            for j in range(node_num):
                dt = 1 / tour_total_distance[i]
                pheromone[int(optimized_path[i, j]) - 1, int(optimized_path[i, j + 1]) - 1] += dt

    best_distance = int(dist_min_distance[0]) + graph[int(best_path[-2]) - 1, 0]
    return best_path, best_distance


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Perform ACO pathfinding on a given graph.csv")
    # parser.add_argument("-g", "--graph.csv", required=True, help="Path to the graph.csv file.")
    # parser.add_argument("--start-node", required=True, help="The starting node in the graph.csv")
    # parser.add_argument("-o", "--output", required=True, help="Path to the output file.")

    # args = parser.parse_args()

    # main(args.graph.csv, args.start_node, args.output)

    # graph_ = [[0, 10, 12, 11, 14], [10, 0, 13, 15, 8], [12, 13, 0, 9, 14], [11, 15, 9, 0, 16], [14, 8, 14, 16, 0]]
    parameters = ACOParameters(graph=read_graph_file("graph.csv"),
        num_iterations=100,
        num_ants=100,
        evaporation_rate=.5,
        pheromone_factor=1,
        visibility_factor=2)

    if is_complete(parameters.graph):
        path, distance = run_ant_colony_optimization(parameters)

        print(f"Path: {path}")
        print(f"Distance: {distance}")
