# Inputs
import MDAnalysis as mda
import numpy as np
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import re
from collections import Counter
from itertools import combinations
from MDAnalysis.lib import distances
from sklearn.cluster import HDBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# INPUT PDB FILES
SYSTEM_FILES = {
    "DMBA":[
        "/mnt/d/work/dmba.hso4/1dmba-h2o/dmba_hso4_1h2o_tip4e_l100ns.pdb",
        "/mnt/d/work/dmba.hso4/2dmba-h2o/dmba_hso4_2h2o_tip4e_l100ns.pdb",
        "/mnt/d/work/dmba.hso4/3dmba-h2o/dmba_hso4_3h2o_tip4e_l100ns.pdb",
        "/mnt/d/work/dmba.hso4/4dmba-h2o/dmba_hso4_4h2o_tip4e_l100ns.pdb",
        "/mnt/d/work/dmba.hso4/5dmba-h2o/dmba_hso4_5h2o_tip4e_l100ns.pdb",
        "/mnt/d/work/dmba.hso4/6dmba-h2o/dmba_hso4_6h2o_tip4e_l100ns.pdb"
    ],

    "HAM":[
        "/mnt/d/work/ha.hso4/1h2o_ha/ham_hso4_1h2o_tip4ew_l100ns.pdb",
        "/mnt/d/work/ha.hso4/2h2o_ha/ham_hso4_2h2o_tip4ew_l100ns.pdb",
        "/mnt/d/work/ha.hso4/3h2o_ha/ham_hso4_3h2o_tip4ew_l100ns.pdb",
        "/mnt/d/work/ha.hso4/4h2o_ha/ham_hso4_4h2o_tip4ew_l100ns.pdb",
        "/mnt/d/work/ha.hso4/5h2o_ha/ham_hso4_5h2o_tip4ew_l100ns.pdb",
        "/mnt/d/work/ha.hso4/6h2o_ha/ham_hso4_6h2o_tip4ew_l100ns.pdb"
    ]
}


# RDF CUT-OFFS (ONLY GRAPH CONSTRUCTION)
CUTOFFS = {
    "DMBA":{
        ("WAT","WAT"):3.4,
        ("WAT","DMBA"):5.5,
        ("WAT","HSO"):5.6,
        ("DMBA","HSO"):6.7,
        ("DMBA","DMBA"):5.5,
        ("HSO","HSO"):5.6
    },

    "HAM":{
        ("WAT","WAT"):3.3,
        ("WAT","HAM"):4.0,
        ("WAT","HSO"):5.0,
        ("HAM","HSO"):5.2,
        ("HAM","HAM"):5.5,
        ("HSO","HSO"):5.6
    }
}


FRAME_STRIDE = 1
HOPS = 1
WATER_SAMPLE_RATE = 0.05


# NODE FEATURES
NODE_MAP = {
    "WAT":[1,0,0],
    "DMBA":[0,1,0],
    "HAM":[0,1,0],
    "HSO":[0,0,1]
}


# MOLECULE GROUPS
def get_groups(u, system):

    if system=="DMBA":
        return {
            "WAT":u.select_atoms("resname WAT"),
            "DMBA":u.select_atoms("resname BMM"),
            "HSO":u.select_atoms("resname HSO")
        }

    if system=="HAM":
        return {
            "WAT":u.select_atoms("resname WAT"),
            "HAM":u.select_atoms("resname HAM"),
            "HSO":u.select_atoms("resname HSO")
        }

# BUILD MOLECULAR NODES
def build_nodes(groups):
    labels, positions, types = [], [], []

    for moltype, atoms in groups.items():
        for res in atoms.residues:
            labels.append(f"{moltype}_{res.resid}")
            positions.append(res.atoms.center_of_mass())
            types.append(moltype)

    return labels, np.array(positions), np.array(types)



# BUILD NETWORKX GRAPH
def build_graph(labels, positions, types, box, system):

    G = nx.Graph()

    for i,t in enumerate(types):

        G.add_node(i, label=labels[i], moltype=t, feature=NODE_MAP[t])

    dist = distances.distance_array(positions, positions, box=box)

    cutoffs = CUTOFFS[system]

    for i,j in combinations(range(len(positions)),2):
        t1,t2 = types[i],types[j]

        cutoff = cutoffs.get((t1,t2), cutoffs.get((t2,t1)))

        if cutoff is None:
            continue


        if dist[i,j] <= cutoff:

            G.add_edge(i, j, distance=float(dist[i,j]), weight=float(1/(dist[i,j]+1e-6)))

    clustering = nx.clustering(G)

    for n in G.nodes():

        G.nodes[n]["degree"] = G.degree(n)
        G.nodes[n]["clustering"] = clustering[n]

    return G

# Water Classification
def classify_water_environment(G,node):

    cation = 0
    anion = 0
    water = 0

    for n in G.neighbors(node):

        t = G.nodes[n]["moltype"]

        if t=="WAT":
            water += 1

        elif t in ["DMBA","HAM"]:
            cation += 1

        elif t=="HSO":
            anion += 1

    if cation==0 and anion==0:
        return 0

    elif cation>0 and anion>0:
        return 2

    else:
        return 1

# Extract Water Graph
def extract_water_subgraphs(G, system, hydration, frame):

    samples=[]

    for node in G.nodes():

        if G.nodes[node]["moltype"]!="WAT":
            continue

        if np.random.random()>WATER_SAMPLE_RATE:
            continue

        ego = nx.ego_graph(G, radius=HOPS)
        label = classify_water_environment(G,node)

        metadata = {
            "system":system,
            "hydration":hydration,
            "frame":frame,
            "water_id":G.nodes[node]["label"]
        }

        samples.append((ego,label,metadata))

    return samples


# Graph Conversion

def nx_to_pyg(G):

    mapping = {
        node:i
        for i,node in enumerate(G.nodes())}

    x = []
    edge_index = []
    edge_attr = []

    # Node features
    for n in G.nodes():

        identity = G.nodes[n]["feature"]
        degree = G.nodes[n]["degree"]
        clustering = G.nodes[n]["clustering"]

        x.append(identity + [degree, clustering])

    # Edge features
    for u,v,data in G.edges(data=True):

        edge_index.append([mapping[u], mapping[v]])
        edge_index.append([mapping[v],mapping[u]])

        features = [data["distance"],data["weight"]]

        edge_attr.append(features)
        edge_attr.append(features)

    # Convert tensors
    x = torch.tensor(x,dtype=torch.float)

    if len(edge_index)==0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,2), dtype=torch.float)

    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor( edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Generate Dataset
def build_dataset(pdb_file, system):
    print("\nLoading:", system)

    u=mda.Universe(pdb_file)
    groups=get_groups(u, system)

    match=re.search(r'(\d+)', pdb_file)
    hydration=int(match.group(1)) if match else None

    dataset=[]
    contact_data=[]

    for ts in u.trajectory[::FRAME_STRIDE]:
        print("Frame",ts.frame)

        labels,pos,types=build_nodes(groups)

        G=build_graph(labels, pos, types, u.dimensions, system)

        for a,b in G.edges():
            if types[a]=="WAT" and types[b]=="WAT":
                contact_data.append((labels[a], labels[b], ts.frame))

        for subgraph,label,meta in extract_water_subgraphs(G, system, hydration, ts.frame):

            pyg=nx_to_pyg(subgraph)
            pyg.y=torch.tensor(label, dtype=torch.long)

            pyg.metadata=meta
            pyg.system=system
            pyg.hydration=hydration
            pyg.frame=ts.frame


            dataset.append(pyg)

        del G

    print("Graphs:", len(dataset))
    return (dataset, contact_data)

# Graph Attention Network

class WaterGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.gat1 = GATConv(in_channels=5, out_channels=32, heads=4, edge_dim=2)
        self.gat2 = GATConv(in_channels=128, out_channels=64, heads=1, edge_dim=2)
        self.classifier = torch.nn.Linear(64, 3)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)

        x = self.gat2(x, edge_index, edge_attr)

        # graph embedding
        z = global_mean_pool(x, batch)
        output = self.classifier(z)

        return output,z

# Training Parameters
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3

def split_dataset(dataset):
    frames = np.array([g.frame for g in dataset])

    unique_frames = np.unique(frames)
    train_frames, test_frames = train_test_split(unique_frames, test_size=0.2, random_state=42)

    train = [
        g for g in dataset
        if g.frame in train_frames]

    test = [
        g for g in dataset
        if g.frame in test_frames]


    print("Training graphs:", len(train))
    print("Testing graphs:", len(test))
    return train,test

# TRAIN GNN
def train_model(train_dataset):
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = WaterGNN().to( DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss=0

        for batch in loader:
            batch=batch.to(DEVICE)
            optimizer.zero_grad()

            output,_ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.cross_entropy(output, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:

            print(
                f"Epoch {epoch+1}/{EPOCHS} "
                f"Loss={total_loss:.4f}"
            )

    return model

# PREDICTION + WATER ID STORAGE
def predict(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    predictions = []
    probabilities = []
    water_ids = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            output, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            prob = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)

            predictions.extend(pred.cpu().numpy())
            probabilities.extend(prob.cpu().numpy())

            # Store water identity
            for graph in batch.to_data_list():
                water_ids.append(graph.metadata["water_id"])

    return (np.array(predictions), np.array(probabilities), water_ids)

# ASSIGN GNN PREDICTIONS TO FULL NETWORK
def assign_predictions_to_network(visualisation_data, test_dataset, predictions):

    prediction_map = {
        g.metadata["water_id"]: int(p)
        for g, p in zip(test_dataset, predictions)}

    for frame_data in visualisation_data:
        G = frame_data["graph"]

        for node in G.nodes():
            label = G.nodes[node]["label"]

            G.nodes[node]["prediction"] = prediction_map.get(label, None)

    return visualisation_data

# Evaluation
def evaluate(predictions, test_dataset):
    true = np.array([
            g.y.item()
            for g in test_dataset])


    print("\nAccuracy:", accuracy_score(true,predictions))
    print("\nClassification Report")
    print(classification_report(true,predictions,target_names=["Quasi-bulk",
                                                               "Single-ion shell","Shared solvent"]))
    print("\nConfusion Matrix")
    print(confusion_matrix(true, predictions))

# Extract Graph Data
def extract_embeddings(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    embeddings = []
    water_ids = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)

            _, z = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            embeddings.append(z.cpu().numpy())

            for graph in batch.to_data_list():
                water_ids.append(graph.metadata["water_id"])

    return np.vstack(embeddings), water_ids

# UMAP Embedding
def cluster_embeddings(embeddings, water_ids):
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    umap_data = {
        wid: reduced[i]
        for i,wid in enumerate(water_ids)}

    return reduced, umap_data

# Network Generation
def build_water_network_from_csv(pdb_file, prediction_csv, contacts, threshold=0.5):
    G = nx.Graph()

    # Read predictions
    df = pd.read_csv(prediction_csv)

    # Last frame coordinates
    u = mda.Universe(pdb_file)
    u.trajectory[-1]

    coords = {
        f"WAT_{r.resid}": r.atoms.center_of_mass()
        for r in u.select_atoms("resname WAT").residues}

    # Create one node for EVERY prediction
    for i, row in df.iterrows():
        wid = row["Water_ID"]

        if wid not in coords:
            continue

        node = f"{wid}_{i}"

        G.add_node(node, water_id=wid, pos=coords[wid], prediction=int(row["Prediction"]))

    # Persistent edges
    counts = {}
    frames = set()

    for w1, w2, f in contacts:
        key = tuple(sorted((w1, w2)))
        counts[key] = counts.get(key, 0) + 1
        frames.add(f)

    total = max(len(frames), 1)

    for (w1, w2), count in counts.items():
        if count / total < threshold:
            continue

        # Connect every copy of water1 to every copy of water2
        nodes1 = [n for n, d in G.nodes(data=True) if d["water_id"] == w1]
        nodes2 = [n for n, d in G.nodes(data=True) if d["water_id"] == w2]

        for n1 in nodes1:
            for n2 in nodes2:
                G.add_edge(n1, n2, persistence=count / total)

    print(
        f"Network: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges")

    return G

# Save Predictions
def save_predictions(dataset,predictions,probabilities,tag):

    names={
        0:"Quasi-bulk",
        1:"Single-ion shell",
        2:"Shared solvent"}

    rows=[]

    for i,g in enumerate(dataset):

        rows.append({
            "System": g.system,
            "Hydration": g.hydration,
            "Frame": g.frame,
            "Water_ID": g.metadata["water_id"],
            "True_Class": int(g.y),
            "True_Label": names[int(g.y)],
            "Prediction": int(predictions[i]),
            "Prediction_Label": names[int(predictions[i])],
            "Confidence": float(probabilities[i].max())})

    df=pd.DataFrame(rows)
    df.to_csv(f"{tag}_GNN_predictions.csv", index=False)
    print("Saved water_GNN_predictions.csv")

# UMAP Plot
def plot_umap(reduced, predictions, tag):
    colours={
        0:"blue",
        1:"yellow",
        2:"orange"}

    plt.figure(figsize=(7,7))
    plt.scatter(reduced[:,0], reduced[:,1],
        c=[
            colours[x]
            for x in predictions],
        s=15)


    legend=[mpatches.Patch(color="blue", label="Quasi-bulk"),
            mpatches.Patch(color="yellow", label="Single-ion shell"),
            mpatches.Patch(color="orange", label="Shared solvent")]

    plt.legend(handles=legend)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(f"{tag}_UMAP.png", dpi=300)
    plt.show()

# 2D WATER NETWORK
def visualize_network_2D(G,tag):
    colours={0:"royalblue",1:"yellow",2:"orange"}
    pos={n:(G.nodes[n]["pos"][0],G.nodes[n]["pos"][1]) for n in G}

    plt.figure(figsize=(8,8))
    nx.draw_networkx_edges(G,pos,alpha=0.4,width=0.5)
    nx.draw_networkx_nodes(G,pos,node_size=35, node_color=[colours.get(G.nodes[n]["prediction"],"black") for n in G])

    plt.legend(handles=[
        mpatches.Patch(color="royalblue",label="Quasi-bulk"),
        mpatches.Patch(color="yellow",label="Single-ion shell"),
        mpatches.Patch(color="orange",label="Shared solvent")],loc="upper right")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{tag}_network_2D.png",dpi=300)
    plt.close()

# 3D WATER NETWORK
def visualize_network_3D(G,tag):
    colours={0:"royalblue",1:"gold",2:"orangered"}
    names={0:"Quasi-bulk",1:"Single-ion shell",2:"Shared solvent"}

    pos={n:G.nodes[n]["pos"] for n in G}

    ex,ey,ez=[],[],[]
    for u,v in G.edges():
        ex+=[pos[u][0],pos[v][0],None]
        ey+=[pos[u][1],pos[v][1],None]
        ez+=[pos[u][2],pos[v][2],None]

    fig=go.Figure([
        go.Scatter3d(
            x=ex,y=ey,z=ez,
            mode="lines",
            line=dict(color="lightgrey",width=2),
            hoverinfo="none",
            showlegend=False
        ),

        go.Scatter3d(
            x=[pos[n][0] for n in G],
            y=[pos[n][1] for n in G],
            z=[pos[n][2] for n in G],
            mode="markers",
            marker=dict(size=5, color=[colours.get(G.nodes[n]["prediction"],"black") for n in G]),
            text=[
                f"{G.nodes[n]['water_id']}<br>{names.get(G.nodes[n]['prediction'],'Unknown')}"
                for n in G],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False),

        go.Scatter3d(x=[None],y=[None],z=[None],mode="markers", marker=dict(size=8,color="royalblue"),name="Quasi-bulk"),
        go.Scatter3d(x=[None],y=[None],z=[None],mode="markers", marker=dict(size=8,color="gold"),name="Single-ion shell"),
        go.Scatter3d(x=[None],y=[None],z=[None],mode="markers", marker=dict(size=8,color="orangered"),name="Shared solvent")])

    fig.update_layout(
        title=f"{tag} Water Network",
        showlegend=True,
        margin=dict(l=0,r=0,b=0,t=40),
        scene=dict(aspectmode="data", bgcolor="white", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))

    fig.write_html(f"{tag}_network_3D.html")

# MAIN
if __name__ == "__main__":

    for system in SYSTEM_FILES:
        for pdb_file in SYSTEM_FILES[system]:
            hydration = re.search(r'(\d+)h2o', pdb_file.lower() ).group(1)

            tag = f"{system}_{hydration}H2O"

            print("\nRunning:", tag)

            dataset, contacts = build_dataset(pdb_file, system)

            if not dataset:
                continue

            train, test = split_dataset(dataset)
            model = train_model(train)
            pred, prob, _ = predict(model, dataset)
            test_pred, _, _ = predict(model, test)

            evaluate(test_pred,test)

            emb, ids = extract_embeddings(model, dataset)
            reduced, _ = cluster_embeddings(emb, ids)

            plot_umap(reduced, pred, tag)
            save_predictions(dataset, pred, prob, tag)
            prediction_csv = f"{tag}_GNN_predictions.csv"
            network = build_water_network_from_csv(pdb_file, prediction_csv, contacts)

            visualize_network_2D(network, tag)
            visualize_network_3D(network, tag)

            del dataset, train, test, model
            del pred, prob, emb, reduced, network

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("Finished:", tag)
    print("\nALL SYSTEMS COMPLETE")
