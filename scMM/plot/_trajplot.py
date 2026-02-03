import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging

from ._traj import PseudoTrajectory

def plot_pseudotime(traj:PseudoTrajectory,
                    figpath:str,
                    show_points=True,
                    show_curves=False,
                    point_size=10,
                    cmap='viridis',
                    branch_color=True,
                    figsize=(7, 6)):
    """
    Visualize embedding with trajectory curves and pseudotime
    """
    logging.info(f"Create pseudotime plot at {figpath}")
    X = traj.embedding_vis
    pt = traj.pseudotime
    curves = getattr(traj, 'trajectory_curves', [])
    branches = getattr(traj, 'branches', None)

    plt.figure(figsize=figsize)

    if show_points:
        if pt is not None:
            plt.scatter(X[:, 0], X[:, 1],
                        c=pt, s=point_size,
                        cmap=cmap, edgecolors='k', lw=0.2)
        else:
            plt.scatter(X[:, 0], X[:, 1],
                        c='lightgray', s=point_size)

    if show_curves and curves:
        n_curves = len(curves)
        for i, curve in enumerate(curves):
            if branch_color:
                color = sns.color_palette("tab10")[i % 10]
            else:
                color = 'red'
            plt.plot(curve[:, 0], curve[:, 1],
                     color=color, lw=2, zorder=5)

    plt.xlabel("Embedding 1")
    plt.ylabel("Embedding 2")
    plt.title("Pseudotime Trajectory")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()

def plot_branches(traj:PseudoTrajectory, figpath:str, figsize=(7, 6), point_size=10):
    """
    Optional: show branch assignment of each cell
    """
    logging.info(f"Create trajectory branch plot at {figpath}")
    X = traj.embedding_vis
    branch_ids = traj.branch_id
    if branch_ids is None:
        raise ValueError("No branch info in trajectory object")

    plt.figure(figsize=figsize)
    n_branches = branch_ids.max() + 1
    palette = sns.color_palette("tab10", n_branches)

    for b in range(n_branches):
        idx = np.where(branch_ids == b)[0]
        plt.scatter(X[idx, 0], X[idx, 1],
                    color=palette[b], s=point_size,
                    label=f'Branch {b}', alpha=0.8)

    curves = getattr(traj, 'trajectory_curves', [])
    for i, curve in enumerate(curves):
        plt.plot(curve[:, 0], curve[:, 1], color='k', lw=1.5, zorder=5)

    plt.xlabel("Embedding 1")
    plt.ylabel("Embedding 2")
    plt.title("Branch assignment")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()