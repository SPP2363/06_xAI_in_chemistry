"""
Deep diagnostic tests for PGExplainer to identify why explanations may be meaningless.

These tests investigate several potential failure modes:
1. Degenerate uniform masks (MLP outputs same value for all edges)
2. Model insensitivity to edge structure
3. Embedding quality issues
4. Faithfulness failures (masks don't correlate with actual importance)
5. Loss landscape issues

Run with: pytest tests/test_pgexplainer_diagnostics.py -v -s
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import PGExplainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.utils import get_embeddings


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def save_artifact(name: str, data: dict) -> Path:
    """Save diagnostic artifact as JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ARTIFACTS_DIR / f"{name}_{timestamp}.json"

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(convert(data), f, indent=2)
    return path


def save_plot(name: str) -> Path:
    """Save current matplotlib figure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ARTIFACTS_DIR / f"{name}_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# =============================================================================
# Test Models
# =============================================================================

class SimpleGCN(nn.Module):
    """Simple GCN for testing."""
    def __init__(self, input_dim=8, hidden_dim=32, output_dim=1):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool = SumAggregation()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        if batch is not None:
            x = self.pool(x, batch)
        else:
            x = x.sum(dim=0, keepdim=True)
        return self.fc(x)


# =============================================================================
# Test Data Generation
# =============================================================================

def create_motif_dataset(num_graphs=100, motif_determines_label=True):
    """
    Create a dataset where a specific motif (triangle) determines the label.

    If motif_determines_label=True:
      - Graphs WITH triangle motif → label 1
      - Graphs WITHOUT triangle motif → label 0

    The triangle is always at nodes [2, 3, 4] when present.
    """
    graphs = []

    for i in range(num_graphs):
        has_motif = i < num_graphs // 2

        # Base chain: 0-1-2-3-4-5-6-7
        num_nodes = 8
        edges = []
        for j in range(num_nodes - 1):
            edges.append([j, j + 1])
            edges.append([j + 1, j])

        motif_edges = []
        if has_motif:
            # Add triangle at nodes 2, 3, 4 (edge 2-4 completes it)
            edges.append([2, 4])
            edges.append([4, 2])
            motif_edges = [(2, 4), (4, 2)]

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        x = torch.randn(num_nodes, 8)

        if motif_determines_label:
            y = torch.tensor([1.0 if has_motif else 0.0])
        else:
            # Random labels - motif doesn't matter
            y = torch.tensor([float(np.random.randint(0, 2))])

        data = Data(x=x, edge_index=edge_index, y=y)
        data.has_motif = has_motif
        data.motif_edges = motif_edges
        graphs.append(data)

    return graphs


def create_random_graphs(num_graphs=50, min_nodes=5, max_nodes=15):
    """Create random graphs for general testing."""
    graphs = []
    for i in range(num_graphs):
        num_nodes = np.random.randint(min_nodes, max_nodes + 1)
        num_edges = np.random.randint(num_nodes, num_nodes * 2)

        edge_index = torch.randint(0, num_nodes, (2, num_edges * 2))
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        x = torch.randn(num_nodes, 8)
        y = torch.tensor([float(i % 2)])

        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return graphs


# =============================================================================
# DIAGNOSTIC TEST 1: Mask Distribution Analysis
# =============================================================================

class TestMaskDistribution:
    """Test whether PGExplainer produces meaningful mask distributions."""

    def test_mask_variance_within_graphs(self):
        """
        Check if edge masks vary within individual graphs.

        FAILURE MODE: If std ≈ 0 within graphs, the MLP is outputting
        nearly identical values for all edges → degenerate solution.
        """
        graphs = create_random_graphs(50)
        model = SimpleGCN()

        # Train model briefly
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(30):
            for data in graphs[:20]:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.binary_cross_entropy_with_logits(out, data.y.view(1, 1))
                loss.backward()
                optimizer.step()
        model.eval()

        # Train PGExplainer
        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=20, lr=0.003, edge_size=0.0, edge_ent=0.0),
            explanation_type='phenomenon',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=ModelConfig(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
        )

        for epoch in range(20):
            for data in graphs[:30]:
                explainer.algorithm.train(
                    epoch=epoch, model=model,
                    x=data.x, edge_index=data.edge_index, target=data.y,
                )

        # Analyze mask distributions
        within_graph_stats = []
        all_masks = []

        for data in graphs[30:]:
            explanation = explainer(data.x, data.edge_index, target=data.y)
            masks = explanation.edge_mask.detach().cpu().numpy()
            all_masks.extend(masks.tolist())

            within_graph_stats.append({
                "num_edges": len(masks),
                "mean": float(masks.mean()),
                "std": float(masks.std()),
                "min": float(masks.min()),
                "max": float(masks.max()),
                "range": float(masks.max() - masks.min()),
            })

        # Aggregate statistics
        avg_within_std = np.mean([s["std"] for s in within_graph_stats])
        avg_range = np.mean([s["range"] for s in within_graph_stats])
        global_std = np.std(all_masks)

        artifact = {
            "within_graph_stats": within_graph_stats,
            "avg_within_graph_std": avg_within_std,
            "avg_within_graph_range": avg_range,
            "global_std": global_std,
            "global_mean": float(np.mean(all_masks)),
            "diagnosis": {
                "degenerate_uniform": avg_within_std < 0.05,
                "all_low": float(np.mean(all_masks)) < 0.1,
                "all_high": float(np.mean(all_masks)) > 0.9,
            }
        }

        path = save_artifact("mask_variance_analysis", artifact)

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Histogram of all masks
        axes[0].hist(all_masks, bins=50, edgecolor='black')
        axes[0].set_xlabel('Edge Mask Value')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'All Masks (global std={global_std:.4f})')
        axes[0].axvline(x=0.5, color='r', linestyle='--')

        # Within-graph std distribution
        stds = [s["std"] for s in within_graph_stats]
        axes[1].hist(stds, bins=20, edgecolor='black')
        axes[1].set_xlabel('Within-Graph Std')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Within-Graph Variance (avg={avg_within_std:.4f})')

        # Within-graph range distribution
        ranges = [s["range"] for s in within_graph_stats]
        axes[2].hist(ranges, bins=20, edgecolor='black')
        axes[2].set_xlabel('Within-Graph Range (max-min)')
        axes[2].set_ylabel('Count')
        axes[2].set_title(f'Within-Graph Range (avg={avg_range:.4f})')

        plt.tight_layout()
        plot_path = save_plot("mask_variance_analysis")

        print(f"\n{'='*60}")
        print("MASK DISTRIBUTION ANALYSIS")
        print(f"{'='*60}")
        print(f"Artifact saved to: {path}")
        print(f"Plot saved to: {plot_path}")
        print(f"\nResults:")
        print(f"  Global mean mask value: {artifact['global_mean']:.4f}")
        print(f"  Global std: {global_std:.4f}")
        print(f"  Avg within-graph std: {avg_within_std:.4f}")
        print(f"  Avg within-graph range: {avg_range:.4f}")
        print(f"\nDiagnosis:")
        print(f"  Degenerate uniform solution: {artifact['diagnosis']['degenerate_uniform']}")
        print(f"  All masks low (<0.1): {artifact['diagnosis']['all_low']}")
        print(f"  All masks high (>0.9): {artifact['diagnosis']['all_high']}")

        if artifact['diagnosis']['degenerate_uniform']:
            print("\n⚠️  WARNING: Masks show very low variance within graphs!")
            print("    The MLP may be outputting nearly identical values for all edges.")


# =============================================================================
# DIAGNOSTIC TEST 2: Model Sensitivity to Edges
# =============================================================================

class TestModelSensitivity:
    """Test whether the GNN model actually depends on edge structure."""

    def test_edge_removal_sensitivity(self):
        """
        Check if the model's predictions change when edges are removed.

        FAILURE MODE: If predictions don't change much when edges are removed,
        the model isn't using edge structure → no meaningful edge explanation possible.
        """
        graphs = create_random_graphs(30)
        model = SimpleGCN()

        # Train model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(50):
            for data in graphs[:20]:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.binary_cross_entropy_with_logits(out, data.y.view(1, 1))
                loss.backward()
                optimizer.step()
        model.eval()

        sensitivity_results = []

        for data in graphs[20:]:
            with torch.no_grad():
                # Original prediction
                orig_pred = torch.sigmoid(model(data.x, data.edge_index)).item()

                num_edges = data.edge_index.shape[1]

                # Prediction with 50% edges removed randomly
                if num_edges > 4:
                    keep_mask = torch.rand(num_edges) > 0.5
                    reduced_edges = data.edge_index[:, keep_mask]
                    reduced_pred = torch.sigmoid(model(data.x, reduced_edges)).item()
                else:
                    reduced_pred = orig_pred

                # Prediction with no edges (just node features)
                no_edge_pred = torch.sigmoid(
                    model(data.x, torch.zeros(2, 0, dtype=torch.long))
                ).item()

                sensitivity_results.append({
                    "orig_pred": orig_pred,
                    "reduced_pred": reduced_pred,
                    "no_edge_pred": no_edge_pred,
                    "change_50pct": abs(orig_pred - reduced_pred),
                    "change_no_edges": abs(orig_pred - no_edge_pred),
                })

        avg_change_50pct = np.mean([r["change_50pct"] for r in sensitivity_results])
        avg_change_no_edges = np.mean([r["change_no_edges"] for r in sensitivity_results])

        artifact = {
            "per_graph_results": sensitivity_results,
            "avg_change_50pct_removal": avg_change_50pct,
            "avg_change_no_edges": avg_change_no_edges,
            "model_uses_edges": avg_change_no_edges > 0.1,
        }

        path = save_artifact("model_edge_sensitivity", artifact)

        print(f"\n{'='*60}")
        print("MODEL EDGE SENSITIVITY ANALYSIS")
        print(f"{'='*60}")
        print(f"Artifact saved to: {path}")
        print(f"\nResults:")
        print(f"  Avg prediction change with 50% edges removed: {avg_change_50pct:.4f}")
        print(f"  Avg prediction change with ALL edges removed: {avg_change_no_edges:.4f}")
        print(f"\nDiagnosis:")
        print(f"  Model uses edge structure: {artifact['model_uses_edges']}")

        if not artifact['model_uses_edges']:
            print("\n⚠️  WARNING: Model predictions barely change when edges are removed!")
            print("    The model may be relying primarily on node features.")
            print("    Edge-based explanations may be meaningless for this model.")


# =============================================================================
# DIAGNOSTIC TEST 3: Faithfulness Test
# =============================================================================

class TestFaithfulness:
    """Test whether PGExplainer masks actually identify prediction-relevant edges."""

    def test_erasure_curve(self):
        """
        Progressively remove edges by importance and measure prediction change.

        If PGExplainer is faithful:
        - Removing HIGH importance edges first should degrade prediction FASTER
        - Removing LOW importance edges first should degrade prediction SLOWER
        - Random removal should be in between

        FAILURE MODE: If all curves are similar, masks don't correlate with
        actual prediction importance.
        """
        graphs = create_random_graphs(50)
        model = SimpleGCN()

        # Train model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(50):
            for data in graphs[:30]:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.binary_cross_entropy_with_logits(out, data.y.view(1, 1))
                loss.backward()
                optimizer.step()
        model.eval()

        # Train PGExplainer
        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=30, lr=0.003, edge_size=0.0, edge_ent=0.0),
            explanation_type='phenomenon',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=ModelConfig(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
        )

        for epoch in range(30):
            for data in graphs[:30]:
                explainer.algorithm.train(
                    epoch=epoch, model=model,
                    x=data.x, edge_index=data.edge_index, target=data.y,
                )

        # Test faithfulness via erasure
        removal_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        curves = {
            "high_first": {f: [] for f in removal_fractions},
            "low_first": {f: [] for f in removal_fractions},
            "random": {f: [] for f in removal_fractions},
        }

        for data in graphs[30:45]:
            explanation = explainer(data.x, data.edge_index, target=data.y)
            masks = explanation.edge_mask.detach().cpu().numpy()

            with torch.no_grad():
                orig_pred = torch.sigmoid(model(data.x, data.edge_index)).item()

            num_edges = data.edge_index.shape[1]

            # Sort edges by importance
            sorted_indices_high = np.argsort(-masks)  # Descending (high first)
            sorted_indices_low = np.argsort(masks)    # Ascending (low first)
            random_indices = np.random.permutation(num_edges)

            for frac in removal_fractions:
                num_remove = int(frac * num_edges)

                for order_name, sorted_idx in [
                    ("high_first", sorted_indices_high),
                    ("low_first", sorted_indices_low),
                    ("random", random_indices),
                ]:
                    if num_remove == 0:
                        pred = orig_pred
                    elif num_remove >= num_edges:
                        with torch.no_grad():
                            pred = torch.sigmoid(
                                model(data.x, torch.zeros(2, 0, dtype=torch.long))
                            ).item()
                    else:
                        keep_indices = sorted_idx[num_remove:]
                        kept_edges = data.edge_index[:, keep_indices]
                        with torch.no_grad():
                            pred = torch.sigmoid(model(data.x, kept_edges)).item()

                    pred_change = abs(pred - orig_pred)
                    curves[order_name][frac].append(pred_change)

        # Average curves
        avg_curves = {}
        for order_name in curves:
            avg_curves[order_name] = [np.mean(curves[order_name][f]) for f in removal_fractions]

        # Compute faithfulness metrics
        # Area between high_first and low_first curves (higher = more faithful)
        high_area = np.trapz(avg_curves["high_first"], removal_fractions)
        low_area = np.trapz(avg_curves["low_first"], removal_fractions)
        faithfulness_gap = high_area - low_area

        artifact = {
            "removal_fractions": removal_fractions,
            "avg_curves": avg_curves,
            "high_first_area": high_area,
            "low_first_area": low_area,
            "faithfulness_gap": faithfulness_gap,
            "is_faithful": faithfulness_gap > 0.05,
        }

        path = save_artifact("faithfulness_erasure_curves", artifact)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(removal_fractions, avg_curves["high_first"], 'r-o', label='Remove HIGH importance first')
        plt.plot(removal_fractions, avg_curves["low_first"], 'b-o', label='Remove LOW importance first')
        plt.plot(removal_fractions, avg_curves["random"], 'g--o', label='Remove RANDOM')
        plt.xlabel('Fraction of Edges Removed')
        plt.ylabel('Prediction Change (|new - original|)')
        plt.title(f'Faithfulness Test: Erasure Curves\nFaithfulness Gap = {faithfulness_gap:.4f}')
        plt.legend()
        plt.grid(True)
        plot_path = save_plot("faithfulness_erasure_curves")

        print(f"\n{'='*60}")
        print("FAITHFULNESS (ERASURE) TEST")
        print(f"{'='*60}")
        print(f"Artifact saved to: {path}")
        print(f"Plot saved to: {plot_path}")
        print(f"\nResults:")
        print(f"  Area under 'high first' curve: {high_area:.4f}")
        print(f"  Area under 'low first' curve: {low_area:.4f}")
        print(f"  Faithfulness gap (high - low): {faithfulness_gap:.4f}")
        print(f"\nDiagnosis:")
        print(f"  Explanations are faithful: {artifact['is_faithful']}")

        if not artifact['is_faithful']:
            print("\n⚠️  WARNING: Removing high-importance edges doesn't affect predictions")
            print("    more than removing low-importance edges!")
            print("    The edge masks are not capturing prediction-relevant structure.")


# =============================================================================
# DIAGNOSTIC TEST 4: Known Motif Detection
# =============================================================================

class TestMotifDetection:
    """Test whether PGExplainer can identify known important substructures."""

    def test_triangle_motif_detection(self):
        """
        Train on a dataset where a triangle motif determines the label.
        Check if PGExplainer identifies the triangle edges as important.

        FAILURE MODE: If motif edges don't have higher importance than
        non-motif edges, PGExplainer is not learning meaningful patterns.
        """
        # Create dataset where triangle motif = positive class
        train_graphs = create_motif_dataset(80, motif_determines_label=True)
        test_graphs = create_motif_dataset(20, motif_determines_label=True)

        model = SimpleGCN()

        # Train model until it learns the motif pattern
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()

        for epoch in range(100):
            total_loss = 0
            correct = 0
            for data in train_graphs:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.binary_cross_entropy_with_logits(out, data.y.view(1, 1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pred = (torch.sigmoid(out) > 0.5).float()
                correct += (pred == data.y).sum().item()

            if epoch % 20 == 0:
                acc = correct / len(train_graphs)
                print(f"  Model training epoch {epoch}: loss={total_loss/len(train_graphs):.4f}, acc={acc:.2f}")

        model.eval()

        # Verify model learned the pattern
        test_correct = 0
        for data in test_graphs:
            with torch.no_grad():
                pred = (torch.sigmoid(model(data.x, data.edge_index)) > 0.5).float()
                test_correct += (pred == data.y).sum().item()
        test_acc = test_correct / len(test_graphs)
        print(f"  Model test accuracy: {test_acc:.2f}")

        if test_acc < 0.7:
            print("  WARNING: Model didn't learn the motif pattern well. Results may be unreliable.")

        # Train PGExplainer
        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=30, lr=0.003, edge_size=0.0, edge_ent=0.0),
            explanation_type='phenomenon',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=ModelConfig(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
        )

        print("  Training PGExplainer...")
        for epoch in range(30):
            for data in train_graphs:
                explainer.algorithm.train(
                    epoch=epoch, model=model,
                    x=data.x, edge_index=data.edge_index, target=data.y,
                )

        # Analyze explanations on graphs WITH motif
        motif_importance = []
        non_motif_importance = []

        test_with_motif = [g for g in test_graphs if g.has_motif]

        for data in test_with_motif:
            explanation = explainer(data.x, data.edge_index, target=data.y)
            masks = explanation.edge_mask.detach().cpu().numpy()
            edge_list = data.edge_index.t().numpy()

            for i, (u, v) in enumerate(edge_list):
                is_motif_edge = (u, v) in data.motif_edges or (v, u) in data.motif_edges
                if is_motif_edge:
                    motif_importance.append(masks[i])
                else:
                    non_motif_importance.append(masks[i])

        avg_motif = np.mean(motif_importance) if motif_importance else 0
        avg_non_motif = np.mean(non_motif_importance) if non_motif_importance else 0

        artifact = {
            "model_test_accuracy": test_acc,
            "num_motif_edges_analyzed": len(motif_importance),
            "num_non_motif_edges_analyzed": len(non_motif_importance),
            "avg_motif_importance": float(avg_motif),
            "avg_non_motif_importance": float(avg_non_motif),
            "motif_importance_values": [float(x) for x in motif_importance],
            "non_motif_importance_sample": [float(x) for x in non_motif_importance[:50]],
            "motif_more_important": avg_motif > avg_non_motif,
            "importance_gap": float(avg_motif - avg_non_motif),
        }

        path = save_artifact("motif_detection_analysis", artifact)

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(
            [motif_importance, non_motif_importance],
            labels=['Motif Edges\n(should be HIGH)', 'Non-Motif Edges\n(should be LOW)']
        )
        ax.set_ylabel('Edge Importance')
        ax.set_title(f'Motif Detection Test\nGap = {artifact["importance_gap"]:.4f}')
        plot_path = save_plot("motif_detection_boxplot")

        print(f"\n{'='*60}")
        print("KNOWN MOTIF DETECTION TEST")
        print(f"{'='*60}")
        print(f"Artifact saved to: {path}")
        print(f"Plot saved to: {plot_path}")
        print(f"\nResults:")
        print(f"  Model test accuracy: {test_acc:.2f}")
        print(f"  Avg importance of MOTIF edges: {avg_motif:.4f}")
        print(f"  Avg importance of NON-MOTIF edges: {avg_non_motif:.4f}")
        print(f"  Importance gap (motif - non_motif): {artifact['importance_gap']:.4f}")
        print(f"\nDiagnosis:")
        print(f"  Motif edges correctly identified as more important: {artifact['motif_more_important']}")

        if not artifact['motif_more_important']:
            print("\n⚠️  WARNING: PGExplainer failed to identify the known motif!")
            print("    The motif edges do not have higher importance than other edges.")


# =============================================================================
# DIAGNOSTIC TEST 5: Embedding Quality
# =============================================================================

class TestEmbeddingQuality:
    """Test whether node embeddings contain discriminative information."""

    def test_embedding_variance(self):
        """
        Check if node embeddings vary meaningfully across nodes and graphs.

        FAILURE MODE: If embeddings are nearly constant, the MLP has no
        discriminative information to work with.
        """
        graphs = create_random_graphs(30)
        model = SimpleGCN()

        # Train model briefly
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(30):
            for data in graphs[:20]:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.binary_cross_entropy_with_logits(out, data.y.view(1, 1))
                loss.backward()
                optimizer.step()
        model.eval()

        all_embeddings = []
        per_graph_stats = []

        for data in graphs[20:]:
            with torch.no_grad():
                _ = model(data.x, data.edge_index)
                embeddings = get_embeddings(model, data.x, data.edge_index)

                if embeddings:
                    emb = embeddings[-1].numpy()  # Use last layer embeddings
                    all_embeddings.append(emb)

                    per_graph_stats.append({
                        "num_nodes": emb.shape[0],
                        "emb_dim": emb.shape[1],
                        "mean_norm": float(np.linalg.norm(emb, axis=1).mean()),
                        "std_across_nodes": float(emb.std(axis=0).mean()),
                        "mean_value": float(emb.mean()),
                    })

        # Concatenate all embeddings
        if all_embeddings:
            all_emb = np.vstack(all_embeddings)
            global_stats = {
                "total_nodes": all_emb.shape[0],
                "emb_dim": all_emb.shape[1],
                "global_mean": float(all_emb.mean()),
                "global_std": float(all_emb.std()),
                "per_dim_std": [float(s) for s in all_emb.std(axis=0)],
                "avg_per_dim_std": float(all_emb.std(axis=0).mean()),
            }
        else:
            global_stats = {"error": "No embeddings captured"}

        artifact = {
            "per_graph_stats": per_graph_stats,
            "global_stats": global_stats,
            "embeddings_are_varied": global_stats.get("avg_per_dim_std", 0) > 0.1,
        }

        path = save_artifact("embedding_quality_analysis", artifact)

        print(f"\n{'='*60}")
        print("EMBEDDING QUALITY ANALYSIS")
        print(f"{'='*60}")
        print(f"Artifact saved to: {path}")
        print(f"\nResults:")
        print(f"  Embedding dimension: {global_stats.get('emb_dim', 'N/A')}")
        print(f"  Global embedding mean: {global_stats.get('global_mean', 'N/A'):.4f}")
        print(f"  Global embedding std: {global_stats.get('global_std', 'N/A'):.4f}")
        print(f"  Avg per-dimension std: {global_stats.get('avg_per_dim_std', 'N/A'):.4f}")
        print(f"\nDiagnosis:")
        print(f"  Embeddings have meaningful variance: {artifact['embeddings_are_varied']}")

        if not artifact['embeddings_are_varied']:
            print("\n⚠️  WARNING: Node embeddings have very low variance!")
            print("    The MLP may not have discriminative information to work with.")


# =============================================================================
# DIAGNOSTIC TEST 6: Loss Landscape Analysis
# =============================================================================

class TestLossLandscape:
    """Analyze the PGExplainer training loss behavior."""

    def test_loss_convergence_pattern(self):
        """
        Track detailed loss behavior during training.

        Look for:
        - Does loss decrease meaningfully?
        - Does it plateau at a reasonable value?
        - Are gradients flowing properly?
        """
        graphs = create_random_graphs(50)
        model = SimpleGCN()

        # Train model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(50):
            for data in graphs[:30]:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.binary_cross_entropy_with_logits(out, data.y.view(1, 1))
                loss.backward()
                optimizer.step()
        model.eval()

        # Train PGExplainer with detailed tracking
        pg_algo = PGExplainer(epochs=50, lr=0.003, edge_size=0.0, edge_ent=0.0)

        explainer = Explainer(
            model=model,
            algorithm=pg_algo,
            explanation_type='phenomenon',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=ModelConfig(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
        )

        training_log = []

        for epoch in range(50):
            epoch_losses = []
            epoch_grad_norms = []

            for data in graphs[:30]:
                loss = explainer.algorithm.train(
                    epoch=epoch, model=model,
                    x=data.x, edge_index=data.edge_index, target=data.y,
                )
                epoch_losses.append(loss)

                # Track gradient norms
                grad_norms = []
                for param in pg_algo.mlp.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                epoch_grad_norms.append(np.mean(grad_norms) if grad_norms else 0)

            # Track MLP parameter norms
            param_norms = [p.norm().item() for p in pg_algo.mlp.parameters()]

            training_log.append({
                "epoch": epoch,
                "mean_loss": float(np.mean(epoch_losses)),
                "std_loss": float(np.std(epoch_losses)),
                "min_loss": float(np.min(epoch_losses)),
                "max_loss": float(np.max(epoch_losses)),
                "mean_grad_norm": float(np.mean(epoch_grad_norms)),
                "param_norms": param_norms,
            })

        # Analysis
        losses = [log["mean_loss"] for log in training_log]
        initial_loss = losses[0]
        final_loss = losses[-1]
        min_loss = min(losses)
        loss_decrease = initial_loss - final_loss
        loss_decrease_pct = (loss_decrease / initial_loss) * 100 if initial_loss > 0 else 0

        artifact = {
            "training_log": training_log,
            "summary": {
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "min_loss": min_loss,
                "loss_decrease": loss_decrease,
                "loss_decrease_pct": loss_decrease_pct,
                "converged_reasonably": final_loss < 0.5 and loss_decrease_pct > 20,
            }
        }

        path = save_artifact("loss_landscape_analysis", artifact)

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        epochs = range(len(losses))

        # Loss curve
        axes[0].plot(epochs, losses, 'b-')
        axes[0].fill_between(
            epochs,
            [log["mean_loss"] - log["std_loss"] for log in training_log],
            [log["mean_loss"] + log["std_loss"] for log in training_log],
            alpha=0.3
        )
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Training Loss\n(decrease: {loss_decrease_pct:.1f}%)')
        axes[0].axhline(y=0.693, color='r', linestyle='--', label='BCE(0.5, y) = 0.693')
        axes[0].legend()

        # Gradient norms
        grad_norms = [log["mean_grad_norm"] for log in training_log]
        axes[1].plot(epochs, grad_norms, 'g-')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Gradient Norm')
        axes[1].set_title('Gradient Magnitude')

        # Loss distribution over epochs
        axes[2].boxplot([
            [training_log[i]["min_loss"], training_log[i]["mean_loss"], training_log[i]["max_loss"]]
            for i in [0, len(training_log)//4, len(training_log)//2, 3*len(training_log)//4, -1]
        ], labels=['Epoch 0', '25%', '50%', '75%', 'Final'])
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Loss Distribution Over Training')

        plt.tight_layout()
        plot_path = save_plot("loss_landscape_analysis")

        print(f"\n{'='*60}")
        print("LOSS LANDSCAPE ANALYSIS")
        print(f"{'='*60}")
        print(f"Artifact saved to: {path}")
        print(f"Plot saved to: {plot_path}")
        print(f"\nResults:")
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Minimum loss achieved: {min_loss:.4f}")
        print(f"  Loss decrease: {loss_decrease:.4f} ({loss_decrease_pct:.1f}%)")
        print(f"\nDiagnosis:")
        print(f"  Converged reasonably: {artifact['summary']['converged_reasonably']}")

        # Interpretation
        print(f"\nInterpretation:")
        if final_loss > 0.693:
            print("  ⚠️  Final loss > 0.693 (random guessing level)")
            print("     The masked model is performing worse than chance!")
        elif final_loss > 0.5:
            print("  ⚠️  Final loss is high (>0.5)")
            print("     The masked model struggles to preserve predictions.")
        elif loss_decrease_pct < 10:
            print("  ⚠️  Loss barely decreased during training")
            print("     The MLP may not be learning anything useful.")
        else:
            print("  ✓  Loss decreased meaningfully during training")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
