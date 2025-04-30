import torch
from torch.serialization import add_safe_globals
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os
import argparse


# Add safe globals for serialization
add_safe_globals([BaseStorage, NodeStorage, EdgeStorage])


def visualize_graph(data, save_path):

    # Convert to NetworkX (project investor-fund to homogeneous)
    edge_index = data['investor', 'invests', 'fund'].edge_index

    G = nx.Graph()
    num_investors = data['investor'].x.size(0)
    num_funds = data['fund'].x.size(0)

    # Add nodes
    G.add_nodes_from(range(num_investors), node_type='investor')
    G.add_nodes_from(range(num_investors, num_investors + num_funds), node_type='fund')

    # Add edges
    investor_nodes = edge_index[0].tolist()
    fund_nodes = (edge_index[1] + num_investors).tolist()  # offset fund IDs
    edges = list(zip(investor_nodes, fund_nodes))
    G.add_edges_from(edges)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Simple visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # force-directed layout
    investor_nodes = [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'investor']
    fund_nodes = [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'fund']

    nx.draw_networkx_nodes(G, pos, nodelist=investor_nodes, node_color='blue', node_size=20, label='Investors')
    nx.draw_networkx_nodes(G, pos, nodelist=fund_nodes, node_color='green', node_size=20, label='Funds')
    nx.draw_networkx_edges(G, pos, alpha=0.2)

    plt.legend()
    plt.title('Investor-Fund Heterogeneous Graph (Projected)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_path}/graph_visualization.png')
    plt.show()


def create_color_map(subgraph, investors_df, funds_df, num_investors):
    color_map = []
    for node in subgraph.nodes():
        node_type = subgraph.nodes[node]['node_type']
        if node_type == 'investor':
            elig_count = investors_df.iloc[node]['fund_eligibility_count']
            if elig_count == 1:
                color_map.append('blue')
            elif elig_count == 2:
                color_map.append('skyblue')
            else:
                color_map.append('lightgreen')
        else:
            fund_cat = funds_df.iloc[node - num_investors]['category']
            color_map.append('red' if fund_cat == 'ETF' else 'orange')
    return color_map


def visualize_batches(data, investors_df, funds_df, batch_size, save_path='output/graph/graph_batch', mode='edge'):  # 'edge' or 'node'
    edge_index = data['investor', 'invests', 'fund'].edge_index
    G = nx.Graph()
    num_investors = data['investor'].x.size(0)
    num_funds = data['fund'].x.size(0)
    total_nodes = num_investors + num_funds

    # Add nodes
    G.add_nodes_from(range(num_investors), node_type='investor')
    G.add_nodes_from(range(num_investors, total_nodes), node_type='fund')

    # Add edges
    investor_nodes = edge_index[0].tolist()
    fund_nodes = (edge_index[1] + num_investors).tolist()
    edges = list(zip(investor_nodes, fund_nodes))
    G.add_edges_from(edges)

    save_path = os.path.join(save_path, f'batch_{batch_size}')
    os.makedirs(save_path, exist_ok=True)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Add legend manually
    legend_elements = [
        Patch(facecolor='blue', label='Investor (1 cat)'),
        Patch(facecolor='skyblue', label='Investor (2 cats)'),
        Patch(facecolor='lightgreen', label='Investor (3 cats)'),
        Patch(facecolor='red', label='Fund (ETF)'),
        Patch(facecolor='orange', label='Fund (Other)')
    ]

    if mode == 'edge':
        random.shuffle(edges)
        num_batches = len(edges) // batch_size + 1
        for i in tqdm(range(num_batches), desc='Edge-based Visualization'):
            sampled_edges = edges[i * batch_size : (i + 1) * batch_size]
            batch_nodes = set()
            for u, v in sampled_edges:
                batch_nodes.add(u)
                batch_nodes.add(v)
            sub_G = G.subgraph(batch_nodes).copy()
            color_map = create_color_map(sub_G, investors_df, funds_df, num_investors)

            plt.figure(figsize=(10, 7))
            pos = nx.spring_layout(sub_G, seed=42)
            nx.draw_networkx_nodes(sub_G, pos, nodelist=sub_G.nodes(), node_color=color_map, node_size=30)
            nx.draw_networkx_edges(sub_G, pos, alpha=0.3)

            plt.legend(handles=legend_elements, loc='best')
            plt.title(f'Subgraph Batch {i+1}/{num_batches} (Edge-Based)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_path}/edge_batch_{i+1:02d}.png')
            plt.close()
    else:
        node_ids = list(G.nodes)
        num_batches = len(node_ids) // batch_size + 1
        for i in tqdm(range(num_batches), desc='Node-based Visualization'):
            # Ensure we don't exceed the number of nodes
            if i * batch_size >= len(node_ids):
                num_batches = i
                break

            batch_nodes = node_ids[i * batch_size : (i + 1) * batch_size]
            sub_G = G.subgraph(batch_nodes).copy()
            color_map = create_color_map(sub_G, investors_df, funds_df, num_investors)

            plt.figure(figsize=(10, 7))
            pos = nx.spring_layout(sub_G, seed=42)
            nx.draw_networkx_nodes(sub_G, pos, nodelist=sub_G.nodes(), node_color=color_map, node_size=30)
            nx.draw_networkx_edges(sub_G, pos, alpha=0.3)
            plt.legend(handles=legend_elements, loc='best')
            plt.title(f'Subgraph Batch {i+1}/{num_batches} (Node-Based)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_path}/node_batch_{i+1:02d}.png')
            plt.close()

    print(f'Saved {num_batches} {mode}-based subgraph visualizations to {save_path}.png')


def graph_diagnostics(data, save_path):
  
    edge_index = data['investor', 'invests', 'fund'].edge_index

    num_investors = data['investor'].x.size(0)
    num_funds = data['fund'].x.size(0)

    G = nx.Graph()
    G.add_nodes_from(range(num_investors), node_type='investor')
    G.add_nodes_from(range(num_investors, num_investors + num_funds), node_type='fund')

    investor_nodes = edge_index[0].tolist()
    fund_nodes = (edge_index[1] + num_investors).tolist()
    edges = list(zip(investor_nodes, fund_nodes))
    G.add_edges_from(edges)

    degrees = [deg for _, deg in G.degree()]
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    num_components = nx.number_connected_components(G)

    isolated = list(nx.isolates(G))
    num_isolated = len(isolated)

    print("\nGraph Diagnostics:")
    print(f"Average Degree: {avg_degree:.2f}")
    print(f"Max Degree: {max_degree}")
    print(f"Number of Connected Components: {num_components}")
    print(f"Number of Isolated Nodes: {num_isolated}")

    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=30, alpha=0.7)
    plt.title("Node Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f'{save_path}/degree_distribution.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Graph')
    parser.add_argument('--batch_size', type=int, default=350, help='Batch size for visualization')
    parser.add_argument('--save_path', type=str, default='../output/graph', help='Path to save batch visualizations')
    parser.add_argument('--data_path', type=str, default='../output/graph/hetero_graph.pt', help='Path to the graph data file')
    parser.add_argument('--mode', type=str, default='edge', choices=['edge', 'node'], help='Visualization mode: edge or node')
    parser.add_argument('--all_data', action='store_true', help='Visualize all data')
    parser.add_argument('--no_show', action='store_true', help='Run graph diagnostics')
    args = parser.parse_args()

    data = torch.load(args.data_path)
    investors_df = pd.read_csv('../data/investors.csv')
    funds_df = pd.read_csv('../data/funds.csv')

    if args.all_data:
        visualize_graph(data, save_path=args.save_path)
    else:
        visualize_batches(data, investors_df, funds_df, batch_size=args.batch_size, save_path=args.save_path, mode=args.mode)

    if not args.no_show:
        graph_diagnostics(data, save_path=args.save_path)