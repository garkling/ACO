import csv
import argparse

import networkx as nx   # graph builder library


Graph = dict


def read_graph_file(filename: str) -> list[list[int, int, int]]:
    """
    Reads a graph from a file and converts it to an adjacency matrix.

    The input file should contain edges of the graph, one per line, formatted as:
    u,v,weight
    where:
    - u and v are integers representing the connected nodes (0-indexed),
    - weight is an integer representing the weight of the edge.

    The graph is assumed to be undirected. The adjacency matrix is symmetrical, 
    and all non-connected node pairs have a weight of 0.

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
    [[0, 5, 3, 0], [5, 0, 0, 2], [3, 0, 0, 4], [0, 2, 4, 0]]
    """
    edges = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            u, v, weight = map(int, row)
            edges.append((u, v, weight))

    n = max(max(u, v) for u, v, _ in edges) + 1

    graph = [[0] * n for _ in range(n)]

    for u, v, weight in edges:
        graph[u][v] = weight
        graph[v][u] = weight
    
    return graph


def write_graph_file(filename: str, graph: Graph):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Result"])
        writer.writerow([graph])


def visualize_graph(graph: Graph):
    pass


def main(graph_file, start_node, output_file):

    print(f"Processing graph: {graph_file}")
    print(f"Starting node: {start_node}")
    print(f"Output file: {output_file}")

    graph = read_graph_file(graph_file)
    # process()
    write_graph_file(output_file, graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform ACO pathfinding on a given graph")
    parser.add_argument("-g", "--graph", required=True, help="Path to the graph file.")
    parser.add_argument("--start-node", required=True, help="The starting node in the graph")
    parser.add_argument("-o", "--output", required=True, help="Path to the output file.")

    args = parser.parse_args()

    main(args.graph, args.start_node, args.output)
