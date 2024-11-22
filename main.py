import csv
import argparse
import numpy as np
import networkx as nx   # graph builder library


Graph = dict


def read_graph_file(filename: str) -> list[list[int]]:
    """
    Reads a graph from a file and converts it to an adjacency matrix.

    The input file should contain edges of the graph, one per line, formatted as:
    u,v,weight
    where:
    - u and v are integers representing the connected nodes (0-indexed),
    - weight is an integer representing the weight of the edge.

    Args:
        filename (str): The path to the file containing the graph data.

    Returns:
        list[list[int]]: A 2D adjacency matrix representing the graph.

    Examples:
    >>> import tempfile
    >>> csv_data = "0,1,5\\n0,2,3\\n1,3,2\\n2,3,4\\n"
    >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding="utf-8") as tmpfile:
    ...     _ = tmpfile.write(csv_data)
    >>> read_graph_file(tmpfile.name)
    [[-1, 5, 3, -1], [5, -1, -1, 2], [3, -1, -1, 4], [-1, 2, 4, -1]]
    >>> csv_data = "0,7,5\\n0,2,3\\n1,3,2\\n2,3,4\\n"
    >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding="utf-8") as tmpfile:
    ...     _ = tmpfile.write(csv_data)
    >>> read_graph_file(tmpfile.name)
    [[-1, -1, 3, -1, 5], [-1, -1, -1, 2, -1], [3, -1, -1, 4, -1], [-1, 2, 4, -1, -1], [5, -1, -1, -1, -1]]
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

    for u, connections in edges.items():
        u_idx = node_index[u]
        for v, weight in connections:
            v_idx = node_index[v]
            graph[u_idx][v_idx] = weight
            graph[v_idx][u_idx] = weight

    return graph


def write_graph_file(filename: str, graph: Graph):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Result"])
        writer.writerow([graph])


def check_graph(graph:list[list[int, int, int]], start = 0) -> bool:
    """
    Tests whether the given graph is a complete unweighted graph.

    A complete graph is a graph in which every vertex is connected to all other vertices,
    except herself. In the adjacency matrix of such a graph:
    - there must be zeros on the diagonal (no loops)
    - all other elements must be positive numbers (there are edges between all vertices)
    - the matrix must be square

    Parameters:
        graph (list[list[int]]): Graph adjacency matrix
        start (int): Start vertex (default 0)

    Returns:
        bool: True if the graph is complete, False otherwise
    >>> graph = [[0,10,12,11,14],[10,0,13,15,8],[12,13,0,9,14],[11,15,9,0,16],[14,8,14,16,0]]
    >>> check_graph(graph)
    True
    """
    n = len(graph)
    if not all(len(row) == n for row in graph):
        return False

    for i in range(n):
        for j in range(n):
            # Перевірка діагональних елементів
            if i == j and graph[i][j] != 0:
                return False
            # Перевірка недіагональних елементів
            if i != j and graph[i][j] <= 0:
                return False
    return True


    # n = len(graph)
    # visited = [False] * n

    # def dfs(vertex):
    #     visited[vertex] = True
    #     for neighbor in range(n):
    #         if graph[vertex][neighbor] > 0 and not visited[neighbor]: #Внесені зміни в логіку. 0 - це вершина наклається сама на себе
    #             dfs(neighbor)

    # dfs(start)
    # return all(visited)



def main(graph_file, start_node, output_file):

    print(f"Processing graph: {graph_file}")
    print(f"Starting node: {start_node}")
    print(f"Output file: {output_file}")

    graph = read_graph_file(graph_file)
    # process()
    write_graph_file(output_file, graph)



def run_ant_colony_optimization(graph, num_iterations, num_ants, evaporation_rate,
                                pheromone_factor, visibility_factor):

    """
    Implements the ant colony algorithm for solving the traveling salesman problem.

    The algorithm simulates the behavior of ants to find the shortest path in the graph.
    Ants leave pheromones on the paths they take. Stronger pheromone trails
    shorter paths attract more ants, leading to route optimization.

    Parameters:
        - graph (list[list[int]]): Matrix of distances between cities (graph nodes)
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

    working_graph = np.array(graph)
    number_nodes = len(graph)
    #обчислення видимості наступної ноди visibility(i,j)=1/d(i,j). Працює все за масивом з numpy
    visibility = 1 / working_graph
    #Оскільки на головній діагоналі у нас нулі, то при діленні
    # на 0 у numpy ми отримаємо inf. Повертаємо всім значенням inf - 0
    visibility[visibility == np.inf] = 0


    #ініціалізація феромнів, присутніх на стежках до нод
    #numpy.ones -> Повертає новий масив заданої форми та типу, заповнений одиницями.
    pheromne = 0.1 * np.ones((num_ants, number_nodes))

    # ініціалізація маршруту мурах із розміром path(num_ants, number_nodes + 1)
    # number_nodes + 1, тому що ми хочемо повернутися до початку

    path = np.ones((num_ants, number_nodes + 1))

    for _ in range(num_iterations):
        # Індексація за numpy. : -всі елементи, 0 - з першого рядка.
        path[:, 0] = 1
        for i in range(num_ants):
            temp_visibility = np.array(visibility)
            for j in range(number_nodes -1):
                characteristic  = np.zeros(number_nodes)
                #ініціалізація сукупного масиву ймовірностей. спершу заповнений нулями
                sum_of_probability  = np.zeros(number_nodes)

                current_node = int(path[i,j] - 1)
                #скидаємо видимість поточної ноди до нуля
                temp_visibility[:,current_node] = 0

                pheromne_characteristic = np.power(pheromne[current_node,:], pheromone_factor)
                visibility_characteristic = np.power(temp_visibility[current_node,:],
                                                     visibility_factor)

                pheromne_characteristic = pheromne_characteristic[:,np.newaxis]
                visibility_characteristic = visibility_characteristic[:,np.newaxis]

                characteristic = np.multiply(pheromne_characteristic, visibility_characteristic)
                total = np.sum(characteristic)
                sum_of_probability = np.cumsum(characteristic / total)

                r = np.random.random_sample()
                path[i,j+1] = np.nonzero(sum_of_probability>r)[0][0]+1
            #пошук останнього ноди, яку ми не відвідали.
            # така схема, бо до останньої ноди є лише один шлях
            end_node = list(set(list(range(1, number_nodes + 1))) - set(path[i, :-2]))[0]
            path[i,-2] = end_node

        optimized_path = np.array(path)
        total_distance_of_tour = np.zeros((num_ants, 1))

        for i in range (num_ants):
            distance = 0
            for j in range(number_nodes - 1):
                #розрахунок загальної відстані
                distance = distance +working_graph[int(optimized_path[i, j])-1, int(optimized_path[i, j + 1]) - 1]
            total_distance_of_tour[i] = distance

        dist_min_loc = np.argmin(total_distance_of_tour)
        dist_min_cost = total_distance_of_tour[dist_min_loc]

        best_path = path[dist_min_loc,:]
        pheromne = (1 - evaporation_rate) * pheromne

        for i in range(num_ants):
            for j in range(number_nodes):
                dt = 1 / total_distance_of_tour[i]
                pheromne[int(optimized_path[i, j]) - 1,int(optimized_path[i, j + 1]) - 1] = pheromne[int(optimized_path[i, j] ) - 1,int(optimized_path[i, j + 1]) - 1] + dt

    # print(f'оптимізований шлях : {best_path}')
    cost_of_best = int(dist_min_cost[0]) + working_graph[int(best_path[-2])-1,0]
    # print('вартість оптимізованого шляху',cost_of_best)
    return (best_path, cost_of_best)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # parser = argparse.ArgumentParser(description="Perform ACO pathfinding on a given graph")
    # parser.add_argument("-g", "--graph", required=True, help="Path to the graph file.")
    # parser.add_argument("--start-node", required=True, help="The starting node in the graph")
    # parser.add_argument("-o", "--output", required=True, help="Path to the output file.")

    # args = parser.parse_args()

    # main(args.graph, args.start_node, args.output)

    graph_ = [[0,10,12,11,14],[10,0,13,15,8],[12,13,0,9,14],[11,15,9,0,16],[14,8,14,16,0]]

    if check_graph(graph_):
        run_ant_colony_optimization(graph_, 100, 10, 0.5 , 1 ,2 )
