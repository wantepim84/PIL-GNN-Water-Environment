<h1>Graph Neural Network Analysis of Hydrated Protic Ionic Liquids</h1>

<h2>Overview</h2>

<p>
This repository contains the data, analysis scripts, graph representations, predictions and visualisations associated with the study of <strong>hydrated protic ionic liquids (PILs)</strong> using <strong>graph neural networks (GNNs)</strong>.
</p>

<p>
The study investigates how the local environments experienced by individual water molecules change as hydration increases. Molecular configurations are represented as molecular interaction networks, allowing the local connectivity surrounding each water molecule to be analysed using graph-based methods.
</p>

<p>
A <strong>Graph Attention Network (GAT)</strong> is trained to classify individual water molecules into three chemically motivated reference environments:
</p>

<ul>
<li><strong>Quasi-bulk</strong></li>
<li><strong>Single-ion shell</strong></li>
<li><strong>Shared solvent</strong></li>
</ul>

<p>
The learned graph representations are subsequently analysed using <strong>Uniform Manifold Approximation and Projection (UMAP)</strong> to examine how local water environments evolve with hydration.
</p>

<hr>

<h2>Molecular systems</h2>

<p>
Two hydrogen sulfate-based protic ionic liquids are investigated:
</p>

<table>
<thead>
<tr>
<th>System</th>
<th>Cation</th>
<th>Anion</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>DMBA·HSO₄</strong></td>
<td>DMBA</td>
<td>HSO₄⁻</td>
</tr>
<tr>
<td><strong>HA·HSO₄</strong></td>
<td>HA</td>
<td>HSO₄⁻</td>
</tr>
</tbody>
</table>

<p>
Six hydration states are considered for each system:
</p>

<p>
<strong>1H₂O, 2H₂O, 3H₂O, 4H₂O, 5H₂O and 6H₂O per ion pair.</strong>
</p>

<p>
The analysis focuses on the evolution of the molecular interaction networks as water content increases.
</p>

<hr>

<h2>Graph representation</h2>

<p>
Each molecular configuration is represented as an undirected molecular interaction graph.
</p>

<ul>
<li><strong>Nodes</strong> represent individual molecules.</li>
<li><strong>Edges</strong> represent intermolecular interactions.</li>
<li>Molecular positions are determined from molecular centres of mass.</li>
<li>Interactions are identified using RDF-derived distance cutoffs.</li>
<li>Edge attributes contain intermolecular distance and an inverse-distance interaction weight.</li>
<li>Node features contain molecular identity, degree and clustering coefficient.</li>
</ul>

<p>
This representation preserves the local connectivity surrounding each water molecule and provides the graph input used by the GNN.
</p>

<hr>

<h2>Water-environment classification</h2>

<p>
Each water molecule is classified according to the composition of its immediate graph neighbourhood.
</p>

<h3>Quasi-bulk</h3>

<p>
Water molecules with no neighbouring cations or anions within the defined interaction range.
</p>

<h3>Single-ion shell</h3>

<p>
Water molecules interacting with either a cation or anion, but not both.
</p>

<h3>Shared solvent</h3>

<p>
Water molecules simultaneously interacting with both a cation and anion.
</p>

<p>
These classifications are <strong>chemically motivated reference classes</strong> generated from local molecular neighbourhoods. They are not treated as experimentally established absolute states.
</p>

<hr>

<h2>Graph Neural Network</h2>

<p>
The molecular interaction graphs are converted into PyTorch Geometric graph objects and analysed using a <strong>Graph Attention Network (GAT)</strong>.
</p>

<pre>
Molecular interaction graph
            │
            ▼
      Node features
            │
            ▼
       GAT layer 1
       32 channels
        4 heads
            │
            ▼
       GAT layer 2
       64 channels
        1 head
            │
            ▼
     Global mean pooling
            │
            ▼
       Graph embedding
            │
            ▼
       Classification
            │
     ┌──────┼──────┐
     ▼      ▼      ▼
 Quasi-  Single-  Shared
  bulk     ion    solvent
          shell
</pre>

<p>
The attention mechanism allows the network to learn the relative importance of neighbouring molecular nodes when constructing graph representations.
</p>

<p>
The resulting model is used to predict the reference environment of individual water molecules and to extract graph-level representations for subsequent analysis.
</p>

<hr>

<h2>UMAP analysis</h2>

<p>
The learned graph representations are projected into two dimensions using <strong>Uniform Manifold Approximation and Projection (UMAP)</strong>.
</p>

<p>
UMAP is used to investigate whether water molecules with similar local interaction environments occupy similar regions of the learned representation space.
</p>

<p>
Each point represents a water molecule. Clustering and separation within the projections provide a visual representation of how local water environments evolve across hydration levels.
</p>

<hr>

<h2>Network visualisation</h2>

<p>
Two-dimensional and interactive three-dimensional molecular networks are generated to provide complementary visual representations of the predicted water environments.
</p>

<h3>2D networks</h3>

<p>
Two-dimensional NetworkX representations show the connectivity of the molecular interaction network and the predicted water environments.
</p>

<h3>3D networks</h3>

<p>
Interactive three-dimensional networks retain the spatial organisation of the molecular system.
</p>

<p>
Water molecules are positioned using their molecular coordinates. Persistent water-water interactions are represented as edges, while nodes are labelled according to their predicted solvent environment.
</p>

<p>
The interactive HTML files allow the molecular networks to be rotated, zoomed and inspected directly.
</p>

<h2>Data structure</h2>

<p>
Each hydration directory contains the data and outputs associated with that hydration state.
</p>

<p>A typical directory contains:</p>

<pre>
1H2O/
├── predictions.csv
├── UMAP.png
├── network_2D.png
└── network_3D.html
</pre>

<p>
The exact files provided depend on the analysis performed for each hydration state.
</p>

<h3>Prediction files</h3>

<p>
Prediction CSV files contain information including:
</p>

<ul>
<li>System</li>
<li>Hydration</li>
<li>Frame</li>
<li>Water ID</li>
<li>Reference class</li>
<li>Predicted class</li>
<li>Prediction confidence</li>
</ul>

<h3>Embeddings</h3>

<p>
The learned graph representations are provided as NumPy arrays where available.
</p>

<h3>UMAP projections</h3>

<p>
UMAP projections are provided as image files for the analysed systems and hydration states.
</p>

<h3>Network visualisations</h3>

<p>
Two-dimensional network images and interactive three-dimensional HTML networks are provided for visual inspection of the predicted molecular organisation.
</p>

<hr>

<h2>Software</h2>

<p>
The analysis uses the following Python packages:
</p>

<ul>
<li>Python</li>
<li>MDAnalysis</li>
<li>NetworkX</li>
<li>NumPy</li>
<li>pandas</li>
<li>SciPy</li>
<li>PyTorch</li>
<li>PyTorch Geometric</li>
<li>scikit-learn</li>
<li>UMAP</li>
<li>Matplotlib</li>
<li>Plotly</li>
</ul>

<p>
See <a href="requirements.txt"><code>requirements.txt</code></a> for the package requirements used for the analysis.
</p>

<hr>

<h2>Reproducibility</h2>

<p>
The computational workflow follows the sequence:
</p>

<pre>
Molecular dynamics configurations
              │
              ▼
   RDF-derived interaction cutoffs
              │
              ▼
    Molecular interaction graphs
              │
              ▼
 Local water-environment classification
              │
              ▼
     PyTorch Geometric graphs
              │
              ▼
      Graph Attention Network
              │
              ▼
    Predictions + graph embeddings
              │
        ┌─────┴─────┐
        ▼           ▼
      UMAP      Network visualisation
        │           │
        ▼           ▼
  2D embeddings   2D / 3D networks
</pre>

<p>
The repository provides the computational scripts, graph-based data and analysis outputs required to examine the hydration-dependent evolution of local water environments.
</p>

<p>
The original molecular dynamics simulations and trajectories are not included unless explicitly provided in the corresponding data directories.
</p>

<hr>

<h2>Results</h2>

<p>
The analysis reveals hydration-dependent reorganisation of local water environments in both PIL systems.
</p>

<p>
At low hydration, water molecules are predominantly associated with the ionic network. As hydration increases, the molecular networks become increasingly water-rich and the distribution of local water environments changes.
</p>

<p>
The two PIL systems exhibit different hydration-dependent behaviour. <strong>DMBA·HSO₄</strong> retains a strong population of shared solvent environments, whereas <strong>HA·HSO₄</strong> shows a greater transition towards single-ion shell and quasi-bulk environments at higher hydration.
</p>

<p>
The UMAP projections and three-dimensional network visualisations provide complementary representations of these structural changes.
</p>

<hr>

<h2>Supplementary material</h2>

<p>
Additional material associated with the study is provided within the <code>supplementary/</code> directory.
</p>

<p>
This includes supporting numerical data, network representations, GNN outputs, UMAP projections and interactive three-dimensional visualisations used to examine the hydration-dependent evolution of local water environments.
</p>

<hr>

<h2>Citation</h2>

<p>
If you use the data, scripts or analysis presented in this repository, please cite the associated publication:
</p>

<pre>
[Publication details to be added]
</pre>

<p>
A <a href="CITATION.cff"><code>CITATION.cff</code></a> file is provided for citation information.
</p>

<hr>

<h2>Licence</h2>

<p>
This repository is distributed under the licence specified in <a href="LICENSE"><code>LICENSE</code></a>.
</p>

<hr>

<h2>Contact</h2>

<p>
For questions regarding the data, analysis or computational workflow, please contact the corresponding author listed in the associated publication.
</p>
