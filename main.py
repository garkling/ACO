import csv
import argparse

import networkx as nx   # graph builder library


Graph = dict


def read_graph_file(filename: str):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(reader)     # todo: add graph deserialization from the file


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
