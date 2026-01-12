"""
Test suite for PGExplainer functionality.

This module contains tests to verify the correct operation of PGExplainer
from PyTorch Geometric, with a focus on diagnosing common issues:
- Loss not decreasing during training
- Explanations producing all zeros or non-sensical values
- Edge mask distribution problems

Artifacts are saved to tests/artifacts/ for later analysis.
"""
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv, GINEConv
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import PGExplainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.utils import get_embeddings


# --- Path Configuration ---
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def get_artifact_path(name: str, ext: str = "json") -> Path:
    """Generate a timestamped artifact path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ARTIFACTS_DIR / f"{name}_{timestamp}.{ext}"


# --- Test Fixtures ---

@pytest.fixture
def simple_graph():
    """Create a simple test graph with known structure."""
    # A simple graph: 0 -- 1 -- 2 -- 3
    #                     |
    #                     4
    x = torch.randn(5, 8)  # 5 nodes, 8 features
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 1, 4],
        [1, 0, 2, 1, 3, 2, 4, 1]
    ], dtype=torch.long)
    y = torch.tensor([1.0])
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def simple_graph_with_edge_attr():
    """Create a simple test graph with edge attributes."""
    x = torch.randn(5, 8)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 1, 4],
        [1, 0, 2, 1, 3, 2, 4, 1]
    ], dtype=torch.long)
    edge_attr = torch.randn(8, 4)  # 8 edges, 4 features
    y = torch.tensor([1.0])
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


@pytest.fixture
def batch_of_graphs():
    """Create a batch of simple graphs for training."""
    graphs = []
    for i in range(20):
        num_nodes = np.random.randint(5, 15)
        # Create random edges
        num_edges = np.random.randint(num_nodes, num_nodes * 2)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn(num_nodes, 8)
        y = torch.tensor([float(i % 2)])  # Alternating labels
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs


# --- Simple GNN Models for Testing ---

class SimpleGCN(nn.Module):
    """A simple GCN model without edge features."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, output_dim: int = 1):
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


class SimpleGATWithEdgeAttr(nn.Module):
    """A GAT model that uses edge attributes (similar to notebook's SimpleClassifier)."""

    def __init__(self, input_dim: int = 8, edge_dim: int = 4,
                 hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.edge_dim = edge_dim
        self.lay_embedd = nn.Linear(input_dim, hidden_dim)
        self.lay_embedd_edge = nn.Linear(edge_dim, 64)

        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=64, heads=2, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=64, heads=2, concat=False)

        self.pool = SumAggregation()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Embed node features
        x = self.lay_embedd(x)

        # Embed edge features if provided
        if edge_attr is not None:
            edge_attr = self.lay_embedd_edge(edge_attr)
        else:
            # Create zero edge attributes if not provided
            edge_attr = torch.zeros(edge_index.shape[1], 64, device=x.device)

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        if batch is not None:
            x = self.pool(x, batch)
        else:
            x = x.sum(dim=0, keepdim=True)
        return self.fc(x)


# ==============================================================================
# BASIC FUNCTIONALITY TESTS
# ==============================================================================

class TestPGExplainerBasicFunctionality:
    """Test basic PGExplainer setup and execution."""

    def test_explainer_creation(self):
        """Test that PGExplainer can be instantiated."""
        pg_explainer = PGExplainer(epochs=10, lr=0.003)
        assert pg_explainer is not None
        assert pg_explainer.epochs == 10
        assert pg_explainer.lr == 0.003

    def test_explainer_with_simple_gcn(self, simple_graph):
        """Test PGExplainer works with a simple GCN model."""
        model = SimpleGCN()
        model.eval()

        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=5, lr=0.01),
            explanation_type='phenomenon',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=ModelConfig(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
        )

        # Train the explainer
        for epoch in range(5):
            loss = explainer.algorithm.train(
                epoch=epoch,
                model=model,
                x=simple_graph.x,
                edge_index=simple_graph.edge_index,
                target=simple_graph.y,
            )
            assert isinstance(loss, float)
            assert not np.isnan(loss)

    def test_get_embeddings_captures_layers(self, simple_graph):
        """Test that get_embeddings actually captures intermediate representations."""
        model = SimpleGCN()
        model.eval()

        with torch.no_grad():
            # Run forward pass
            _ = model(simple_graph.x, simple_graph.edge_index)

            # Get embeddings
            embeddings = get_embeddings(
                model,
                simple_graph.x,
                simple_graph.edge_index
            )

        # Save diagnostic info
        artifact = {
            "num_embeddings_captured": len(embeddings),
            "embedding_shapes": [list(e.shape) for e in embeddings],
            "embedding_norms": [float(e.norm().item()) for e in embeddings],
        }

        artifact_path = get_artifact_path("get_embeddings_test")
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)

        print(f"\nArtifact saved to: {artifact_path}")
        print(f"Number of embeddings captured: {len(embeddings)}")
        for i, emb in enumerate(embeddings):
            print(f"  Layer {i}: shape={emb.shape}, norm={emb.norm().item():.4f}")

        assert len(embeddings) > 0, "No embeddings were captured!"


# ==============================================================================
# DIAGNOSTIC TESTS FOR LOSS BEHAVIOR
# ==============================================================================

class TestPGExplainerLossBehavior:
    """Tests to diagnose why loss might not decrease during training."""

    def test_loss_decreases_simple_gcn(self, batch_of_graphs):
        """Test that loss decreases during training with a simple GCN."""
        model = SimpleGCN()
        model.eval()  # Model should be in eval mode for explanations

        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=30, lr=0.01),
            explanation_type='phenomenon',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=ModelConfig(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
        )

        losses_per_epoch = []

        for epoch in range(30):
            epoch_losses = []
            for data in batch_of_graphs[:10]:  # Use subset
                loss = explainer.algorithm.train(
                    epoch=epoch,
                    model=model,
                    x=data.x,
                    edge_index=data.edge_index,
                    target=data.y,
                )
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            losses_per_epoch.append(avg_loss)

        # Save loss curve artifact
        artifact = {
            "losses_per_epoch": losses_per_epoch,
            "initial_loss": losses_per_epoch[0],
            "final_loss": losses_per_epoch[-1],
            "loss_decreased": bool(losses_per_epoch[-1] < losses_per_epoch[0]),
            "loss_decrease_ratio": losses_per_epoch[-1] / losses_per_epoch[0] if losses_per_epoch[0] > 0 else None,
        }

        artifact_path = get_artifact_path("loss_curve_simple_gcn")
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)

        # Plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses_per_epoch, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('PGExplainer Training Loss (Simple GCN)')
        plt.grid(True)
        plot_path = get_artifact_path("loss_curve_simple_gcn", "png")
        plt.savefig(plot_path)
        plt.close()

        print(f"\nLoss curve saved to: {plot_path}")
        print(f"Initial loss: {losses_per_epoch[0]:.4f}")
        print(f"Final loss: {losses_per_epoch[-1]:.4f}")
        print(f"Loss decreased: {artifact['loss_decreased']}")

        # This is a diagnostic test - we want to see the behavior
        # even if it doesn't decrease
        if not artifact['loss_decreased']:
            print("WARNING: Loss did not decrease during training!")

    def test_loss_components_breakdown(self, simple_graph):
        """Analyze the individual components of the PGExplainer loss."""
        model = SimpleGCN()
        model.eval()

        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=10, lr=0.01),
            explanation_type='phenomenon',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=ModelConfig(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
        )

        # Get the coefficients used
        coeffs = explainer.algorithm.coeffs.copy()

        # Train for a few epochs and examine the internal state
        loss_components = []

        for epoch in range(10):
            # Get embeddings manually to inspect
            embeddings = get_embeddings(
                model,
                simple_graph.x,
                simple_graph.edge_index
            )

            # Train step
            loss = explainer.algorithm.train(
                epoch=epoch,
                model=model,
                x=simple_graph.x,
                edge_index=simple_graph.edge_index,
                target=simple_graph.y,
            )

            # Check MLP parameters
            mlp_params = list(explainer.algorithm.mlp.parameters())
            param_norms = [p.norm().item() for p in mlp_params]
            param_grads = [p.grad.norm().item() if p.grad is not None else 0.0 for p in mlp_params]

            loss_components.append({
                "epoch": epoch,
                "total_loss": loss,
                "embedding_norm": float(embeddings[-1].norm().item()) if embeddings else None,
                "mlp_param_norms": param_norms,
                "mlp_grad_norms": param_grads,
                "temperature": explainer.algorithm._get_temperature(epoch),
            })

        artifact = {
            "coeffs": coeffs,
            "loss_components": loss_components,
        }

        artifact_path = get_artifact_path("loss_components_breakdown")
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)

        print(f"\nLoss components artifact saved to: {artifact_path}")
        print(f"Coefficients: {coeffs}")
        for lc in loss_components:
            print(f"Epoch {lc['epoch']}: loss={lc['total_loss']:.4f}, "
                  f"temp={lc['temperature']:.4f}, "
                  f"grad_norms={[f'{g:.4f}' for g in lc['mlp_grad_norms']]}")


# ==============================================================================
# DIAGNOSTIC TESTS FOR EXPLANATION QUALITY
# ==============================================================================

class TestPGExplainerExplanationQuality:
    """Tests to diagnose why explanations might be all zeros or non-sensical."""

    def test_edge_mask_distribution(self, batch_of_graphs):
        """Analyze the distribution of edge mask values after training."""
        model = SimpleGCN()
        model.eval()

        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=20, lr=0.01),
            explanation_type='phenomenon',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=ModelConfig(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
        )

        # Train the explainer
        for epoch in range(20):
            for data in batch_of_graphs[:10]:
                explainer.algorithm.train(
                    epoch=epoch,
                    model=model,
                    x=data.x,
                    edge_index=data.edge_index,
                    target=data.y,
                )

        # Generate explanations for test graphs
        edge_mask_stats = []
        all_edge_masks = []

        for i, data in enumerate(batch_of_graphs[10:15]):
            explanation = explainer(
                data.x,
                data.edge_index,
                target=data.y,
            )

            edge_mask = explanation.edge_mask.detach().cpu().numpy()
            all_edge_masks.extend(edge_mask.tolist())

            stats = {
                "graph_idx": i,
                "num_edges": len(edge_mask),
                "min": float(edge_mask.min()),
                "max": float(edge_mask.max()),
                "mean": float(edge_mask.mean()),
                "std": float(edge_mask.std()),
                "num_zeros": int((edge_mask == 0).sum()),
                "num_ones": int((edge_mask >= 0.99).sum()),
                "all_same": bool(edge_mask.std() < 1e-6),
            }
            edge_mask_stats.append(stats)

        # Aggregate statistics
        aggregate_stats = {
            "total_edges": len(all_edge_masks),
            "global_min": float(np.min(all_edge_masks)),
            "global_max": float(np.max(all_edge_masks)),
            "global_mean": float(np.mean(all_edge_masks)),
            "global_std": float(np.std(all_edge_masks)),
            "pct_below_0.1": float((np.array(all_edge_masks) < 0.1).mean() * 100),
            "pct_above_0.9": float((np.array(all_edge_masks) > 0.9).mean() * 100),
        }

        artifact = {
            "per_graph_stats": edge_mask_stats,
            "aggregate_stats": aggregate_stats,
            "all_edge_masks": all_edge_masks,
        }

        artifact_path = get_artifact_path("edge_mask_distribution")
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)

        # Plot histogram of edge masks
        plt.figure(figsize=(10, 6))
        plt.hist(all_edge_masks, bins=50, edgecolor='black')
        plt.xlabel('Edge Mask Value')
        plt.ylabel('Count')
        plt.title('Distribution of Edge Mask Values')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold=0.5')
        plt.legend()
        plot_path = get_artifact_path("edge_mask_histogram", "png")
        plt.savefig(plot_path)
        plt.close()

        print(f"\nEdge mask distribution artifact saved to: {artifact_path}")
        print(f"Histogram saved to: {plot_path}")
        print(f"Aggregate stats: {aggregate_stats}")

        # Check for problematic patterns
        if aggregate_stats['global_std'] < 0.01:
            print("WARNING: Edge masks have very low variance - all values are nearly identical!")
        if aggregate_stats['pct_below_0.1'] > 90:
            print("WARNING: >90% of edge masks are below 0.1 - explanations may be too sparse!")
        if aggregate_stats['pct_above_0.9'] > 90:
            print("WARNING: >90% of edge masks are above 0.9 - explanations are not discriminative!")

    def test_mlp_output_before_sigmoid(self, batch_of_graphs):
        """Check the raw MLP outputs (logits) before sigmoid is applied."""
        model = SimpleGCN()
        model.eval()

        pg_algo = PGExplainer(epochs=20, lr=0.01)

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

        # Train
        for epoch in range(20):
            for data in batch_of_graphs[:10]:
                explainer.algorithm.train(
                    epoch=epoch,
                    model=model,
                    x=data.x,
                    edge_index=data.edge_index,
                    target=data.y,
                )

        # Manually get raw logits for inspection
        logits_before_sigmoid = []

        for data in batch_of_graphs[10:15]:
            embeddings = get_embeddings(model, data.x, data.edge_index)
            if not embeddings:
                continue

            emb = embeddings[-1]
            edge_index = data.edge_index

            # Replicate what PGExplainer does internally
            zs = [emb[edge_index[0]], emb[edge_index[1]]]
            inputs = torch.cat(zs, dim=-1)

            with torch.no_grad():
                logits = pg_algo.mlp(inputs).view(-1)
                logits_before_sigmoid.extend(logits.cpu().numpy().tolist())

        logits_array = np.array(logits_before_sigmoid)

        artifact = {
            "num_logits": len(logits_array),
            "min": float(logits_array.min()),
            "max": float(logits_array.max()),
            "mean": float(logits_array.mean()),
            "std": float(logits_array.std()),
            "pct_negative": float((logits_array < 0).mean() * 100),
            "pct_large_negative": float((logits_array < -5).mean() * 100),
            "pct_large_positive": float((logits_array > 5).mean() * 100),
            "logits_sample": logits_array[:100].tolist(),
        }

        artifact_path = get_artifact_path("mlp_logits_analysis")
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)

        # Plot logits distribution
        plt.figure(figsize=(10, 6))
        plt.hist(logits_array, bins=50, edgecolor='black')
        plt.xlabel('MLP Logit Value (before sigmoid)')
        plt.ylabel('Count')
        plt.title('Distribution of Raw MLP Outputs')
        plt.axvline(x=0, color='r', linestyle='--', label='Logit=0 (sigmoid=0.5)')
        plt.legend()
        plot_path = get_artifact_path("mlp_logits_histogram", "png")
        plt.savefig(plot_path)
        plt.close()

        print(f"\nMLP logits artifact saved to: {artifact_path}")
        print(f"Logits stats: min={artifact['min']:.4f}, max={artifact['max']:.4f}, "
              f"mean={artifact['mean']:.4f}, std={artifact['std']:.4f}")

        # Check for issues
        if artifact['std'] < 0.1:
            print("WARNING: MLP outputs have very low variance!")
        if artifact['pct_large_negative'] > 50:
            print("WARNING: Many logits are large negative values - sigmoid will output ~0!")


# ==============================================================================
# EDGE ATTRIBUTE COMPATIBILITY TESTS
# ==============================================================================

class TestPGExplainerEdgeAttributeCompatibility:
    """Test PGExplainer with models that use edge attributes."""

    def test_get_embeddings_with_edge_attr_model(self, simple_graph_with_edge_attr):
        """Test if get_embeddings works correctly with edge-attribute models."""
        model = SimpleGATWithEdgeAttr()
        model.eval()

        data = simple_graph_with_edge_attr

        # First, verify the model works with edge_attr
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            print(f"Model output with edge_attr: {out}")

        # Now try get_embeddings - this is where issues might arise
        # because get_embeddings might not pass edge_attr correctly
        try:
            embeddings = get_embeddings(
                model,
                data.x,
                data.edge_index,
                edge_attr=data.edge_attr  # Pass edge_attr as kwarg
            )

            artifact = {
                "success": True,
                "num_embeddings": len(embeddings),
                "embedding_shapes": [list(e.shape) for e in embeddings],
            }
            print(f"get_embeddings succeeded with edge_attr!")
            print(f"Embeddings captured: {len(embeddings)}")

        except Exception as e:
            artifact = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            print(f"get_embeddings failed with edge_attr: {e}")

        artifact_path = get_artifact_path("edge_attr_compatibility")
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)

    def test_pgexplainer_with_edge_attr_model(self, simple_graph_with_edge_attr):
        """Test full PGExplainer workflow with edge-attribute model."""
        model = SimpleGATWithEdgeAttr()
        model.eval()

        data = simple_graph_with_edge_attr

        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=10, lr=0.01),
            explanation_type='phenomenon',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=ModelConfig(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
        )

        results = {
            "training_losses": [],
            "training_errors": [],
        }

        for epoch in range(10):
            try:
                loss = explainer.algorithm.train(
                    epoch=epoch,
                    model=model,
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,  # Pass edge_attr
                    target=data.y,
                )
                results["training_losses"].append({"epoch": epoch, "loss": loss})
            except Exception as e:
                results["training_errors"].append({
                    "epoch": epoch,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                print(f"Training error at epoch {epoch}: {e}")
                break

        # Try to generate explanation
        try:
            explanation = explainer(
                data.x,
                data.edge_index,
                edge_attr=data.edge_attr,
                target=data.y,
            )
            results["explanation_success"] = True
            results["edge_mask_shape"] = list(explanation.edge_mask.shape)
            results["edge_mask_stats"] = {
                "min": float(explanation.edge_mask.min()),
                "max": float(explanation.edge_mask.max()),
                "mean": float(explanation.edge_mask.mean()),
            }
        except Exception as e:
            results["explanation_success"] = False
            results["explanation_error"] = str(e)

        artifact_path = get_artifact_path("pgexplainer_edge_attr_model")
        with open(artifact_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nEdge attr model test artifact saved to: {artifact_path}")
        print(f"Training completed epochs: {len(results['training_losses'])}")
        if results.get('explanation_success'):
            print(f"Explanation generated successfully!")
            print(f"Edge mask stats: {results['edge_mask_stats']}")
        else:
            print(f"Explanation failed: {results.get('explanation_error')}")


# ==============================================================================
# COMPARISON TESTS
# ==============================================================================

class TestPGExplainerComparisons:
    """Compare PGExplainer behavior under different configurations."""

    def test_different_learning_rates(self, batch_of_graphs):
        """Test how different learning rates affect training."""
        model = SimpleGCN()
        model.eval()

        learning_rates = [0.0001, 0.001, 0.01, 0.1]
        results = {}

        for lr in learning_rates:
            # Reset model for fair comparison
            model = SimpleGCN()
            model.eval()

            explainer = Explainer(
                model=model,
                algorithm=PGExplainer(epochs=20, lr=lr),
                explanation_type='phenomenon',
                node_mask_type=None,
                edge_mask_type='object',
                model_config=ModelConfig(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='raw',
                ),
            )

            losses = []
            for epoch in range(20):
                epoch_losses = []
                for data in batch_of_graphs[:5]:
                    loss = explainer.algorithm.train(
                        epoch=epoch,
                        model=model,
                        x=data.x,
                        edge_index=data.edge_index,
                        target=data.y,
                    )
                    epoch_losses.append(loss)
                losses.append(np.mean(epoch_losses))

            results[str(lr)] = {
                "losses": losses,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "loss_decreased": bool(losses[-1] < losses[0]),
            }

        artifact_path = get_artifact_path("learning_rate_comparison")
        with open(artifact_path, "w") as f:
            json.dump(results, f, indent=2)

        # Plot comparison
        plt.figure(figsize=(12, 6))
        for lr_str, data in results.items():
            plt.plot(data["losses"], label=f'lr={lr_str}', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('PGExplainer Training Loss vs Learning Rate')
        plt.legend()
        plt.grid(True)
        plot_path = get_artifact_path("learning_rate_comparison", "png")
        plt.savefig(plot_path)
        plt.close()

        print(f"\nLearning rate comparison saved to: {artifact_path}")
        for lr_str, data in results.items():
            print(f"  lr={lr_str}: initial={data['initial_loss']:.4f}, "
                  f"final={data['final_loss']:.4f}, decreased={data['loss_decreased']}")

    def test_different_regularization_coeffs(self, batch_of_graphs):
        """Test how different regularization coefficients affect explanations."""
        model = SimpleGCN()
        model.eval()

        # Different configurations
        configs = [
            {"edge_size": 0.0, "edge_ent": 0.0, "name": "no_reg"},
            {"edge_size": 0.05, "edge_ent": 1.0, "name": "default"},
            {"edge_size": 0.1, "edge_ent": 0.0, "name": "size_only"},
            {"edge_size": 0.0, "edge_ent": 2.0, "name": "entropy_only"},
            {"edge_size": 0.2, "edge_ent": 2.0, "name": "high_reg"},
        ]

        results = {}

        for config in configs:
            model = SimpleGCN()
            model.eval()

            explainer = Explainer(
                model=model,
                algorithm=PGExplainer(
                    epochs=20,
                    lr=0.01,
                    edge_size=config["edge_size"],
                    edge_ent=config["edge_ent"],
                ),
                explanation_type='phenomenon',
                node_mask_type=None,
                edge_mask_type='object',
                model_config=ModelConfig(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='raw',
                ),
            )

            # Train
            losses = []
            for epoch in range(20):
                epoch_losses = []
                for data in batch_of_graphs[:5]:
                    loss = explainer.algorithm.train(
                        epoch=epoch,
                        model=model,
                        x=data.x,
                        edge_index=data.edge_index,
                        target=data.y,
                    )
                    epoch_losses.append(loss)
                losses.append(np.mean(epoch_losses))

            # Generate explanations
            edge_masks = []
            for data in batch_of_graphs[10:15]:
                explanation = explainer(data.x, data.edge_index, target=data.y)
                edge_masks.extend(explanation.edge_mask.detach().cpu().numpy().tolist())

            edge_masks = np.array(edge_masks)

            results[config["name"]] = {
                "config": config,
                "losses": losses,
                "edge_mask_stats": {
                    "min": float(edge_masks.min()),
                    "max": float(edge_masks.max()),
                    "mean": float(edge_masks.mean()),
                    "std": float(edge_masks.std()),
                },
            }

        artifact_path = get_artifact_path("regularization_comparison")
        with open(artifact_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nRegularization comparison saved to: {artifact_path}")
        for name, data in results.items():
            stats = data["edge_mask_stats"]
            print(f"  {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")


# ==============================================================================
# KNOWN STRUCTURE TESTS
# ==============================================================================

class TestPGExplainerKnownStructure:
    """Test PGExplainer on graphs with known important substructures."""

    def test_motif_detection(self):
        """
        Create graphs where a specific motif determines the label.
        Test if PGExplainer can identify this motif.
        """
        # Create graphs where a triangle motif determines positive class
        def create_graph_with_motif(has_motif: bool, base_size: int = 8):
            """Create a graph, optionally with a triangle motif."""
            # Base chain graph
            edge_list = []
            for i in range(base_size - 1):
                edge_list.append([i, i + 1])
                edge_list.append([i + 1, i])

            motif_edges = []
            if has_motif:
                # Add triangle motif at nodes 2, 3, 4
                motif_edges = [(2, 4), (4, 2)]
                edge_list.extend([[2, 4], [4, 2]])

            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            x = torch.randn(base_size, 8)
            y = torch.tensor([1.0 if has_motif else 0.0])

            return Data(x=x, edge_index=edge_index, y=y), motif_edges

        # Create dataset
        train_graphs = []
        for _ in range(15):
            g, _ = create_graph_with_motif(has_motif=True)
            train_graphs.append(g)
            g, _ = create_graph_with_motif(has_motif=False)
            train_graphs.append(g)

        # Train a model that can distinguish
        model = SimpleGCN()

        # Quick training of the base model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(50):
            total_loss = 0
            for data in train_graphs:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.binary_cross_entropy_with_logits(out, data.y.view(1, 1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model.eval()

        # Now train PGExplainer
        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=30, lr=0.01),
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
            for data in train_graphs:
                explainer.algorithm.train(
                    epoch=epoch,
                    model=model,
                    x=data.x,
                    edge_index=data.edge_index,
                    target=data.y,
                )

        # Test on graphs with motif - check if motif edges are highlighted
        test_graph, motif_edges = create_graph_with_motif(has_motif=True)
        explanation = explainer(
            test_graph.x,
            test_graph.edge_index,
            target=test_graph.y,
        )

        edge_mask = explanation.edge_mask.detach().cpu().numpy()
        edge_index = test_graph.edge_index.t().numpy()

        # Find indices of motif edges
        motif_edge_indices = []
        for me in motif_edges:
            for i, (u, v) in enumerate(edge_index):
                if (u == me[0] and v == me[1]):
                    motif_edge_indices.append(i)

        # Calculate statistics
        motif_mask_values = [float(edge_mask[i]) for i in motif_edge_indices] if motif_edge_indices else []
        non_motif_mask_values = [float(edge_mask[i]) for i in range(len(edge_mask))
                                  if i not in motif_edge_indices]

        artifact = {
            "motif_edge_indices": motif_edge_indices,
            "motif_mask_values": motif_mask_values,
            "motif_mean": float(np.mean(motif_mask_values)) if motif_mask_values else None,
            "non_motif_mean": float(np.mean(non_motif_mask_values)) if non_motif_mask_values else None,
            "all_edge_masks": [float(x) for x in edge_mask.tolist()],
            "edge_index": edge_index.tolist(),
            "motif_highlighted": bool(
                np.mean(motif_mask_values) > np.mean(non_motif_mask_values)
                if motif_mask_values and non_motif_mask_values else False
            ),
        }

        artifact_path = get_artifact_path("motif_detection_test")
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)

        print(f"\nMotif detection test saved to: {artifact_path}")
        print(f"Motif edges mean importance: {artifact['motif_mean']}")
        print(f"Non-motif edges mean importance: {artifact['non_motif_mean']}")
        print(f"Motif correctly highlighted: {artifact['motif_highlighted']}")


if __name__ == "__main__":
    # Run with: pytest tests/test_pgexplainer.py -v -s
    pytest.main([__file__, "-v", "-s"])
