import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain


def random_graph(vertices_count: int, edges_count: int) -> nx.Graph:
    """
    Function that generate random graph with input count of vertices and edges

    :param vertices_count: vertices count
    :param edges_count: edges count
    :type vertices_count: int
    :type edges_count: int
    :return: generated graph
    :rtype: nx.Graph
    """

    if edges_count > vertices_count ** 2 / 2:
        raise ValueError("Edges count must be less then vertices count ^ 2 / 2")

    if edges_count < vertices_count:
        raise ValueError("Vertices count must be bigger then edges count")

    result = nx.Graph()

    for i in range(vertices_count):
        result.add_node(i)

    possible_vertices_in_matrix = [elem for elem in range(vertices_count ** 2) if elem // vertices_count < elem % vertices_count]

    for row in range(vertices_count - 1):
        column = random.randint(row + 1, vertices_count - 1)

        result.add_edge(row, column)

        possible_vertices_in_matrix.remove(row * vertices_count + column)

    edges_left = edges_count - vertices_count

    random_vertices = random.sample(possible_vertices_in_matrix, edges_left)

    for elem in random_vertices:
        result.add_edge(elem // vertices_count, elem % vertices_count)

    return result


def from_graph_to_matrix(graph: nx.Graph) -> list:
    """
    Function that translate nx.Graph into adjacency matrix

    :param graph: input graph
    :type graph: nx.Graph
    :return: adjacency matrix of input graph
    :rtype: list
    """

    edges = graph.edges()
    result = np.zeros([len(graph.nodes()), len(graph.nodes())]).tolist()

    for edge in edges:
        result[edge[0]][edge[1]] = result[edge[1]][edge[0]] = 1

    return result


def from_graph_to_adjacency_list(graph: nx.Graph) -> dict:
    """
    Function that translate nx.Graph into adjacency list

    :param graph: input graph
    :type graph: nx.Graph
    :return: adjacency list of input graph
    :rtype: dict
    """

    result = {}
    matrix = from_graph_to_matrix(graph)

    for ind, m_row in enumerate(matrix):
        result[ind] = set([c_index for c_index, m_column in enumerate(m_row) if m_column == 1])

    return result


def depth_first(graph: dict, start: int, visited: list = None) -> set:
    """
    Function that find connected components of the input graph

    :param graph: adjacency list of input graph
    :param start: start vertice
    :param visited: list of visited vertices
    :type graph: dict
    :type start: int
    :type visited: list
    :return: set of connected components of the input graph
    :rtype: set
    :raise: Value error("Start vertice not in vertices of graph")
    """

    all_vertices = set(chain.from_iterable([list(value) for key, value in graph.items()]))

    if start not in all_vertices:
        raise ValueError("Start vertice not in vertices of graph")

    if visited is None:
        visited = set()

    visited.add(start)

    for next_vertice in graph[start] - visited:
        depth_first(graph, next_vertice, visited)

    return visited


def breadth_first(graph: dict, start_vertice: int, end_vertice: int) -> list:
    """
    Function that find the shortest path between two input vertices

    :param graph: adjacency list of input graph
    :param start_vertice: start vertice
    :param end_vertice: end vertice
    :type graph: dict
    :type start_vertice: int
    :type end_vertice: int
    :return: the shortest path between start_vertice and ens_vertice
    :rtype: list
    :raise: ValueError("Start or end vertice not in vertices of graph")
    """

    all_vertices = set(chain.from_iterable([list(value) for key, value in graph.items()]))

    if start_vertice not in all_vertices or end_vertice not in all_vertices:
        raise ValueError("Start or end vertice not in vertices of graph")

    queue = [[start_vertice]]

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node == end_vertice:
            return path

        for communicate in graph.get(node, []):
            new_path = list(path)
            new_path.append(communicate)
            queue.append(new_path)


# Generating graph
my_graph = random_graph(100, 200)

# Translate graph into adjacency matrix and list
graph_matrix = from_graph_to_matrix(my_graph)
graph_list = from_graph_to_adjacency_list(my_graph)

# Print adjacency matrix
print("ADJACENCY MATRIX")
print('\t   ', end='')

for col in range(len(graph_matrix)):
    if col < 10:
        end = '  '
    else:
        end = ' '

    print(col, end=end)

print()
print('\t  ', end='')

for _ in range(len(graph_matrix)):
    print(' _ ', end='')

print()

for index, row in enumerate(graph_matrix):
    print(index, '\t| ', end=' ')
    for column in row:
        print(int(column), ' ', end='')
    print()

print()

# Print adjacency list
print("ADJACENCY LIST")
for row, comm in graph_list.items():
    print(str(row) + ': ' + '[' + ', '.join(str(elem) for elem in comm) + ']')

# Plot graph
nx.draw(my_graph, with_labels=True, font_color='white', font_size=8)
plt.show()

# Using depth-first search to find connected components of the graph
print()
print("DEPTH-FIRST")
print(depth_first(from_graph_to_adjacency_list(my_graph), 1))

# Using breadth-first search to find the shortest path between two vertices
print()
print("BREADTH-FIRST")
print(breadth_first(from_graph_to_adjacency_list(my_graph), 0, 9))
