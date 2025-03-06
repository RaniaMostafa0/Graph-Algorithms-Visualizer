import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import timeit
from tkinter import messagebox
from collections import deque

def switch_to_bfs_gui():
    main_frame.pack_forget()
    bfs_gui_frame.pack()
# Function to switch to the DFS GUI screen
def switch_to_dfs_gui():
    main_frame.pack_forget()
    dfs_gui_frame.pack()

def switch_to_main_screen():
    dfs_gui_frame.pack_forget()
    dijkstra_gui_frame.pack_forget()
    floyd_warshall_frame.pack_forget()
    bfs_gui_frame.pack_forget()
    bellman_gui_frame.pack_forget()
    main_frame.pack()

def switch_to_dijkstra_gui():
    main_frame.pack_forget()
    dijkstra_gui_frame.pack()

def switch_to_bellman_gui():
    main_frame.pack_forget()
    bellman_gui_frame.pack()

def switch_to_floyd_warshall_gui():
    main_frame.pack_forget()
    floyd_warshall_frame.pack()

def clear_graph_canvas_dfs():
    for widget in graph_frame.winfo_children():
        widget.destroy()

# Create the GUI window
window = tk.Tk()
window.title('Graph Algorithms GUI')
window.geometry("800x600")  # Set the window size

# Create the main frame
main_frame = tk.Frame(window)
main_frame.pack(pady=10)

# Create the buttons in the main frame
bfs_button = tk.Button(main_frame, text='BFS Algorithm', command=switch_to_bfs_gui)
bfs_button.grid(row=0, column=0, padx=5, pady=5)

dfs_button = tk.Button(main_frame, text='DFS Algorithm', command=switch_to_dfs_gui)
dfs_button.grid(row=0, column=1, padx=5, pady=5)

dijkstra_button = tk.Button(main_frame, text="Dijkstra's Algorithm", command=switch_to_dijkstra_gui)
dijkstra_button.grid(row=0, column=2, padx=5, pady=5)

bellman_button = tk.Button(main_frame, text='Bellman Ford Algorithm', command=switch_to_bellman_gui)
bellman_button.grid(row=0, column=3, padx=5, pady=5)

floyd_warshall_button = tk.Button(main_frame, text='Floyd Warshall Algorithm', command=switch_to_floyd_warshall_gui)
floyd_warshall_button.grid(row=0, column=4, padx=5, pady=5)

dfs_gui_frame = tk.Frame(window)

dfs_button_frame = tk.Frame(dfs_gui_frame)
dfs_button_frame.pack(pady=10)


# DFS algorithm
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    result_dfs.append(start)

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Function to display DFS result
def show_dfs():
    clear_graph_canvas_dfs()  # Clear the graph canvas
    result_dfs.clear()
    dfs(graph, 'A')
    result_label_dfs.config(text="DFS Result: " + ', '.join(result_dfs))


# Function to display the graph
def show_graph():
    clear_graph_canvas_dfs()  # Clear the graph canvas
    nx_graph = nx.Graph(graph)
    fig, ax = plt.subplots(figsize=(6, 4))
    nx.draw(nx_graph, with_labels=True, ax=ax)
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Create the graph
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# Create a list to store the result of the DFS traversal
result_dfs = []

# Create the frame for buttons in the DFS GUI frame
dfs_button_frame = tk.Frame(dfs_gui_frame)
dfs_button_frame.pack(pady=10)

# Create the buttons in the DFS GUI frame
dfs_button = tk.Button(dfs_button_frame, text='Perform DFS', command=show_dfs, padx=10, pady=5)
dfs_button.grid(row=0, column=0, padx=5)

graph_button = tk.Button(dfs_button_frame, text='Show Graph', command=show_graph, padx=10, pady=5)
graph_button.grid(row=0, column=2, padx=5)

back_button = tk.Button(dfs_gui_frame, text='Back', command=switch_to_main_screen, padx=10, pady=5)
back_button.pack()

# Create a label to display the DFS result in the DFS GUI frame
result_label_dfs = tk.Label(dfs_gui_frame, text="DFS Result: ", font=('Arial', 12), pady=10)
result_label_dfs.pack()

# Create a frame to hold the graph canvas in the DFS GUI framegraph_frame = tk.Frame(dfs_gui_frame)
graph_frame = tk.Frame(dfs_gui_frame)
graph_frame.pack(pady=10)

bfs_gui_frame = tk.Frame(window)

def bfs(graphbfs, start):
    visited = set()
    result_bfs = []
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        result_bfs.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result_bfs

def show_bfs():
    clear_graph_canvas_dfs()  # Clear the graph canvas
    result_bfs = bfs(graphbfs, 'A')
    result_label_bfs.config(text="BFS Result: " + ', '.join(result_bfs))
# Create the DFS GUI frame


def show_bfs_graph():
    clear_graph_canvas_dfs()  # Clear the graph canvas
    nx_graphbfs = nx.Graph(graphbfs)
    fig, ax = plt.subplots(figsize=(6, 4))
    nx.draw(nx_graphbfs, with_labels=True, ax=ax)
    canvas = FigureCanvasTkAgg(fig, master=graph_frame_bfs)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
graphbfs = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
    
bfs_button_frame = tk.Frame(bfs_gui_frame)
bfs_button_frame.pack(pady=10)

bfs_button = tk.Button(bfs_button_frame, text='Perform BFS', command=show_bfs, padx=10, pady=5)
bfs_button.grid(row=0, column=0, padx=5)

graph_button_bfs = tk.Button(bfs_button_frame, text='Show Graph', command=show_bfs_graph, padx=10, pady=5)
graph_button_bfs.grid(row=0, column=1, padx=5)

back_button = tk.Button(bfs_gui_frame, text='Back', command=switch_to_main_screen, padx=10, pady=5)
back_button.pack()

# Create a label to display the DFS result in the DFS GUI frame
result_label_bfs = tk.Label(bfs_gui_frame, text="BFS Result: ", font=('Arial', 12), pady=10)
result_label_bfs.pack()

graph_frame_bfs = tk.Frame(bfs_gui_frame)
graph_frame_bfs.pack(pady=10)
    
# Create the Dijkstra GUI frame
dijkstra_gui_frame = tk.Frame(window)


# Dijkstra's algorithm
def dijkstra(graph, start):
    # Initialize distances dictionary with infinite values for all nodes except the start node
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]

    # Main loop of the algorithm
    while len(pq) > 0:
        # Sort the priority queue and take the node with the minimum distance
        pq.sort(key=lambda x: x[0])
        current_distance, current_node = pq.pop(0)

        # Skip the iteration if the current distance is already greater than the distance in the dictionary
        if current_distance > distances[current_node]:
            continue

        # Update the distances to the neighboring nodes if a shorter path is found
        for neighbor, edge_weight in graph[current_node].items():
            distance = current_distance + edge_weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                pq.append((distance, neighbor))

    return distances


# Function to display Dijkstra's result
def show_dijkstra():
    clear_graph_canvas_dijkstra()  # Clear the graph canvas
    result_dijkstra = dijkstra(weighted_graph, 'A')
    result_label_dijkstra.config(text="Dijkstra's Result: " + ', '.join(f"{k}: {v}" for k, v in result_dijkstra.items()))

def clear_graph_canvas_dijkstra():
    for widget in graph_frame_dijkstra.winfo_children():
        widget.destroy()

# Function to display the graph
def show_graph_dijkstra():
    clear_graph_canvas_dijkstra()  # Clear the graph canvas
    nx_graph = nx.DiGraph(weighted_graph)
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos, with_labels=True, ax=ax)
    labels = nx.get_edge_attributes(nx_graph, 'weight')
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=labels, ax=ax)
    canvas = FigureCanvasTkAgg(fig, master=graph_frame_dijkstra)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Create the weighted graph
weighted_graph = {
    'A': {'B': 5, 'C': 3},
    'B': {'A': 5, 'C': 1, 'D': 2},
    'C': {'A': 3, 'B': 1, 'D': 4},
    'D': {'B': 2, 'C': 4}
}

bellman_gui_frame = tk.Frame(window)

def bellman_ford(graphbellman, source):
    # Find the number of vertices in the graph by finding the length of the dictionary keys.
    V = len(graphbellman)
    inf = float("inf")
    # Initialize an array of distances to all nodes as infinite, and set the source node to zero.
    distance = [inf] * V
    distance[source] = 0

    # Relax each edge V-1 times
    for i in range(V - 1):
        for u in graphbellman:
            for v, w in graphbellman[u]:
                if distance[u] != inf and distance[u] + w < distance[v]:
                    distance[v] = distance[u] + w

    # Check for negative-weight cycles
    for u in graphbellman:
        for v, w in graphbellman[u]:
            if distance[u] != inf and distance[u] + w < distance[v]:
                return "Graph contains negative weight cycle"

    # Return the shortest distances from the source node to all other nodes
    return distance

def show_bellman_graph():
    G = nx.DiGraph()
    for u in graphbellman:
        for v, w in graphbellman[u]:
            G.add_edge(u, v, weight=w)

    # Specify the positions of nodes
    pos = {0: (0, 0), 1: (5, 0), 2: (0, 2), 3: (2.5, 1), 4: (5, 2)}

    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', edge_color='black', ax=ax)
    # gets the weight and set it on the edge in the graph
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)

    # Embed the matplotlib figure in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=graph_frame_bellman)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
def display_shortest_distances(entry):
    source_node = int(entry.get())
    result_bellman = bellman_ford(graphbellman, source_node)

    if result_bellman == "Graph contains negative weight cycle":
        result_label_bellman.config(text=result_bellman)
    else:
        result_label_bellman.config(text="")
        result_message = ""
        for i in range(len(result_bellman)):
            result_message += f"Distance from source {source_node} to node {i} is: {result_bellman[i]}\n"
        messagebox.showinfo("Shortest Distances", result_message)

entry_frame = tk.Frame(bellman_gui_frame)
entry_frame.pack()

label_entry = tk.Label(entry_frame, text="Source Node:")
label_entry.pack(side="left")

entry = tk.Entry(entry_frame)
entry.pack(side="left")

bellman_button_frame = tk.Frame(bellman_gui_frame)
bellman_button_frame.pack(pady=10)

bellman_button = tk.Button(bellman_button_frame, text='Perform Bellman Ford', command=lambda: display_shortest_distances(entry), padx=10, pady=5)
bellman_button.pack(side='left', padx=5)

graph_button_bellman = tk.Button(bellman_button_frame, text='Show Graph', command=show_bellman_graph, padx=10, pady=5)
graph_button_bellman.pack(side='left', padx=5)

back_button = tk.Button(bellman_gui_frame, text='Back', command=switch_to_main_screen, padx=10, pady=5)
back_button.pack()

# Create a label to display the DFS result in the DFS GUI frame
result_label_bellman = tk.Label(bellman_gui_frame, text="Bellman Ford Result: ", font=('Arial', 12), pady=10)
result_label_bellman.pack()

graph_frame_bellman = tk.Frame(bellman_gui_frame)
graph_frame_bellman.pack(pady=10)


graphbellman = {
    0: [(1, -1), (2, 4)],
    1: [(2, 3), (3, 5), (4, 5)],
    2: [],
    3: [(2, 9)],
    4: [(3, -3)]
}

# Create the Floyd Warshall frame
floyd_warshall_frame = tk.Frame(window)

# Floyd Warshall Algorithm
INF = float("inf")

graph_fw = [
    [0, 8, INF, 1],
    [INF, 0, 1, INF],
    [4, INF, 0, INF],
    [INF, 2, 9, 0]
]

def floyd(graph):
    distances = graph
    len_matrix = len(graph)

    # Applying the Floyd algorithm
    for k in range(len_matrix):
        for i in range(len_matrix):
            for j in range(len_matrix):
                distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])

    return distances

def show_floyd_warshall():
    distances = floyd(graph_fw)
    create_output_window(distances, graph_fw)

def create_output_window(distances, graph):
    output_window = tk.Toplevel(window)
    output_window.title("Floyd Warshall Algorithm")

    # Draw the directed graph diagram
    fig, ax = plt.subplots(figsize=(5, 5))
    G = nx.Graph()
    for i in range(len(graph)):
        for j in range(len(graph)):
            if i != j and graph[i][j] != INF:
                G.add_edge(i, j, weight=graph[i][j])
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
    ax.set_title("Directed Graph Diagram")

    # Display the distances matrix
    table = tk.Frame(output_window)
    table.grid(row=0, column=0, padx=10, pady=10)

    for i, row in enumerate(distances):
        for j, distance in enumerate(row):
            label = tk.Label(table, text=str(distance), padx=10, pady=5)
            label.grid(row=i, column=j, padx=5, pady=5)

    canvas = FigureCanvasTkAgg(fig, master=output_window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

    output_window.mainloop()

# Create the frame for buttons in the Dijkstra GUI frame
dijkstra_button_frame = tk.Frame(dijkstra_gui_frame)
dijkstra_button_frame.pack(pady=10)

floyd_warshall_button_frame = tk.Frame(floyd_warshall_frame)
floyd_warshall_button_frame.pack(pady=10)

# Create the buttons in the Dijkstra GUI frame
dijkstra_button = tk.Button(dijkstra_button_frame, text="Perform Dijkstra's", command=show_dijkstra, padx=10, pady=5)
dijkstra_button.grid(row=0, column=0, padx=5)

graph_button_dijkstra = tk.Button(dijkstra_button_frame, text='Show Graph', command=show_graph_dijkstra, padx=10, pady=5)
graph_button_dijkstra.grid(row=0, column=1, padx=5)

floyd_warshall_show_button = tk.Button(floyd_warshall_button_frame, text="Perform FloydWarshall Algorithm", command=show_floyd_warshall)
floyd_warshall_show_button.grid(row=0, column=0, padx=5, pady=5)

floyd_warshall_back_button = tk.Button(floyd_warshall_button_frame, text="Back", command=switch_to_main_screen)
floyd_warshall_back_button.grid(row=0, column=1, padx=5, pady=5)

back_button_dijkstra = tk.Button(dijkstra_gui_frame, text='Back', command=switch_to_main_screen, padx=10, pady=5)
back_button_dijkstra.pack()

# Create a label to display Dijkstra's result in the Dijkstra GUI frame
result_label_dijkstra = tk.Label(dijkstra_gui_frame, text="Dijkstra's Result: ", font=('Arial', 12), pady=10)
result_label_dijkstra.pack()

# Create a frame to hold the graph canvas in the Dijkstra GUI frame
graph_frame_dijkstra = tk.Frame(dijkstra_gui_frame)
graph_frame_dijkstra.pack(pady=10)

# Run the GUI
window.mainloop()