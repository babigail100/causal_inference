import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
dag = nx.DiGraph()

# Add nodes and edges
dag.add_edges_from([
    ("Exercise", "Depression/Anxiety"),
    ("Age", "Depression/Anxiety"),
    ("Gender", "Depression/Anxiety"),
    ("SES", "Exercise"),
    ("SES", "Depression/Anxiety"),
    ("Conditions", "Exercise"),
    ("Conditions", "Depression/Anxiety"),
    ("Therapy", "Depression/Anxiety"),
    ("Medication", "Depression/Anxiety"),
    ("Age", "Conditions"),
    ("Genetic Predisposition","Depression/Anxiety"),
    ("Gender","Genetic Predisposition"),
    ("Conditions","Medication"),
    ("SES","Medication"),
    ("SES","Therapy"),
    ("Conditions","Therapy"),
    ("Gender","Conditions"),
    ("Age","Exercise"),
    ("Therapy","Medication")
])
plt.clf()
# Draw the graph
pos = nx.shell_layout(dag)
nx.draw(dag, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")
plt.show()


# plt.savefig("MHdag.png", format="png", dpi=300)


undirected_dag = dag.to_undirected()

paths = list(nx.all_simple_paths(undirected_dag, source="Exercise", target="Depression/Anxiety"))

for path in paths:
    formatted_path = []
    for i in range(len(path) - 1):
        if dag.has_edge(path[i], path[i + 1]): 
            formatted_path.append(f"{path[i]} →")
        else:  # Backward direction
            formatted_path.append(f"{path[i]} ←")
    formatted_path.append(path[-1]) 
    print(" ".join(formatted_path))
