# Graph Neural Networks & Network Science Explorations

Welcome to my repository of projects and tutorials on Network Science and Graph Neural Networks (GNNs). This collection serves as a practical journey through the foundational concepts of graph theory to the implementation of modern GNN architectures and knowledge representation techniques.

The notebooks are organized into two main sections:
1.  **Network Science (University Coursework):** A series of notebooks from my university course at the Higher School of Computer Science (ESI-SBA), covering graph theory, classical node embedding techniques, and the fundamentals of GNNs.
2.  **Knowledge Graphs (YouTube Series):** Implementations and explorations based on a YouTube tutorial series, focusing on practical knowledge representation and building a recommendation system.

---

## Network Science (University Coursework)

This section contains the work completed as part of my Network Science course. It builds concepts from the ground up, starting with basic graph manipulation and moving toward sophisticated deep learning models for graphs.

### 1. Introduction to Graph Theory
**File:** `01_Introduction_to_Graph_theory_for_GNNs_with_NetworkX.ipynb`

This notebook serves as a foundational introduction to graph theory and its implementation using the `networkx` library. It covers the essential building blocks for any graph-based analysis.

**Key Concepts Covered:**
*   **Graph Types:** Creating and visualizing Undirected, Directed, and Weighted graphs.
*   **Graph Properties:** Checking for connectivity and understanding different graph layouts.
*   **Centrality Measures:** Computing Degree, Closeness, and Betweenness centrality to identify important nodes.
*   **Graph Traversal:** Implementing Breadth-First Search (BFS) and Depth-First Search (DFS).

### 2 p1. Node Embeddings via Matrix Factorization (Graph Factorization & HOPE)
**File:** `Matrix_Factorization.ipynb`

This notebook explores classical node embedding techniques that represent nodes as low-dimensional vectors. It provides from-scratch implementations of two matrix factorization methods.

**Key Concepts Covered:**
*   **Graph Factorization (GF):** Learns embeddings by factorizing the adjacency matrix (`A ≈ U * U^T`) using gradient descent.
*   **HOPE (High-Order Proximity Preserved Embedding):** Preserves higher-order proximity by factorizing a similarity matrix (Katz Index) using Singular Value Decomposition (SVD).
*   **Comparison:** A direct visual and quantitative comparison between GF and HOPE on a Barbell Graph.

### 2 p2. Node Embeddings via Random Walks (DeepWalk & Node2Vec)
**File:** `skip-gram.ipynb`

This notebook bridges Natural Language Processing (NLP) and Graph Theory, implementing powerful embedding techniques inspired by Word2Vec.

**Key Concepts Covered:**
*   **DeepWalk:** Uses unbiased random walks to generate "sentences" from a graph, which are then fed into a Word2Vec model to learn node embeddings.
*   **Node2Vec:** Improves upon DeepWalk by using **biased random walks** (controlled by parameters `p` and `q`) to better capture both local and global network structures.

### 3. Vanilla GNN vs. MLP Baseline
**File:** `03_Nodes_Classification_with_Vanilla_Graph_Neural_Networks.ipynb`

This notebook demonstrates the power of GNNs by comparing a from-scratch Vanilla GNN against a standard Multi-Layer Perceptron (MLP) baseline on node classification tasks.

**Key Concepts Covered:**
*   **Comparative Analysis:** Training both MLP and GNN models on the **Cora** and the larger **Facebook Page-Page** datasets.
*   **Performance Gain:** Quantifies the significant accuracy improvement achieved by the GNN's ability to leverage graph structure (message passing).

### 4. Introduction to Graph Convolutional Networks (GCN)
**File:** `04_Introduction_to_Graph_Convolution_Networks_(GCN).ipynb`

This notebook dives into the Graph Convolutional Network, covering its theory and providing multiple levels of implementation.

**Key Concepts Covered:**
*   **Theoretical Foundation:** Explains the symmetrically normalized adjacency matrix, the core of the GCN propagation rule.
*   **Implementation Levels:** Builds a GCN from scratch using PyTorch, then compares it with the optimized `GCNConv` layer from PyTorch Geometric.
*   **Architectural Depth:** Explores the performance of a deeper GCN model with dropout.

### 5. Introduction to Graph Attention Networks (GAT)
**File:** `05_Introduction_to_Graph_Attention_Networks_(GATs).ipynb`

This notebook introduces the Graph Attention Network, an advanced architecture that allows nodes to dynamically learn the importance of their neighbors' messages.

**Key Concepts Covered:**
*   **Self-Attention:** A simplified, from-scratch implementation to build intuition.
*   **GAT Model:** Implements a full GAT using PyTorch Geometric's `GATv2Conv` layer, leveraging multi-head attention to capture diverse relationships.

---

## Knowledge Graphs (YouTube Series)

This section contains projects based on a YouTube tutorial series (https://youtube.com/playlist?list=PLZsOBAyNTZwacEFVI8yo5o-1rZasRqm18&si=CEgPaFP-i5ydm8RL) from DigitalSreeni focused on building and querying knowledge graphs.

### 1. Building a Knowledge Graph with NetworkX
**File:** `01_knowledge_graphs_using_NetworkX.ipynb`

This notebook focuses on the basics of creating a knowledge graph from the ground up using `networkx`. It demonstrates how to encode domain knowledge (a learning path for Python) into a graph structure with rich metadata.

**Key Concepts Covered:**
*   **Graph Construction:** Adding nodes and edges with detailed attributes like difficulty, weights, and descriptions.
*   **Visualization:** Creating a visually informative graph with custom colors, labels, and edge thickness.
*   **Graph Algorithms:** Applying algorithms like shortest path and betweenness centrality to answer practical questions (e.g., "What is the optimal learning path?").
*   **Community Detection:** Using modularity-based algorithms to find clusters of related topics.

### 2. Knowledge Representation with RDFLib and SPARQL
**File:** `02_Defining_Knowledge_Graphs_with_NetworkX_and_RDF.ipynb`

This notebook introduces a more formal and standardized approach to knowledge representation using the Resource Description Framework (RDF). It contrasts the flexibility of `networkx` with the semantic rigor of `rdflib`.

**Key Concepts Covered:**
*   **RDF Fundamentals:** Understanding the core concepts of Triples (Subject-Predicate-Object), URIs, and Literals.
*   **Predefined Vocabularies:** Using standard vocabularies like RDF, RDFS, and XSD to add semantic meaning to data.
*   **SPARQL Query Language:** An in-depth guide to querying RDF graphs with SPARQL, including `FILTER` clauses and path expressions.
*   **NetworkX vs. RDFLib:** A direct comparison of the two approaches, showing how to model the same learning path knowledge in both frameworks and highlighting their respective strengths.

### 3. Advanced Knowledge Graph and Recommendation System
**File:** `03_Building_a_Learning_Path_Recommender_-_Manual_Construction_of_Knowledge_Graphs.ipynb`

This notebook combines all previous concepts into a comprehensive, multi-domain knowledge graph (Programming, Finance, and Bioimage Analysis) and uses it to power a recommendation system.

**Key Concepts Covered:**
*   **Multi-Domain Modeling:** Structuring and merging topic data from three distinct fields into a single, cohesive graph.
*   **Interactive Visualization:** Using `pyvis` to create a dynamic, interactive HTML visualization of the complex knowledge graph.
*   **Advanced Analysis:** Implementing functions for prerequisite discovery, learning path recommendation, and topic search by keyword.
*   **Data Persistence:** Demonstrates how to export the graph to JSON and store it in a SQLite database for use in external applications.

---

## Setup & Installation

To run these notebooks, it is recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a conda environment:**
    ```bash
    conda create -n graph_env python=3.12
    conda activate graph_env
    ```

3.  **Install the required packages.**

    The key libraries used in this project include:
    *   `torch` & `torch_geometric`
    *   `networkx` & `pyvis`
    *   `numpy` & `pandas`
    *   `scikit-learn` & `rdflib`
    *   `matplotlib`
    *   `gensim` (for DeepWalk)
    *   `node2vec` (for Node2Vec library)

5.  **Launch Jupyter Notebook or Jupyter Lab:**
    ```bash
    jupyter notebook
    ```

---

## Acknowledgements

*   The **"Network Science"** coursework was developed under the guidance of **Dr. Belkacem KHALDI (b.khaldi@esi-sba.dz)** at the Higher School of Computer Science (ESI-SBA).
*   The **"Knowledge Graphs"** series of notebooks is based on the excellent tutorials from the **DigitalSreeni** YouTube channel. You can find the original playlist here: **https://youtube.com/playlist?list=PLZsOBAyNTZwacEFVI8yo5o-1rZasRqm18&si=CEgPaFP-i5ydm8RL**.
