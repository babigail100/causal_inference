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




plt.savefig("MHdag.png", format="png", dpi=300)
