import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
import pandas as pd

# Set Nature Communications style
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
    }
)

# Nature Communications color palette
colors = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#d62728",
    "light_blue": "#aec7e8",
    "light_orange": "#ffbb78",
    "background": "#f7f7f7",
    "grid": "#e0e0e0",
    "text": "#333333",
}

# Panel A 工厂函数


def create_panel_a():
    fig, ax = plt.subplots(figsize=(3.5, 3))
    np.random.seed(42)
    time_points = np.linspace(220, 250, 30)
    cell_ids = [f"DC{i:02d}" for i in range(1, 21)]
    co_cluster_prob = np.zeros((20, 30))
    peak_time = 15
    for i in range(20):
        cell_peak = peak_time + np.random.normal(0, 2)
        cell_width = np.random.uniform(4, 6)
        cell_intensity = np.random.uniform(0.7, 1.0)
        cell_profile = cell_intensity * np.exp(
            -0.5 * ((np.arange(30) - cell_peak) / cell_width) ** 2
        )
        co_cluster_prob[i, :] = cell_profile
        co_cluster_prob[i, :] += np.random.normal(0, 0.1, 30)
        co_cluster_prob[i, :] = np.clip(co_cluster_prob[i, :], 0, 1)
    im = ax.imshow(
        co_cluster_prob,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )
    ax.set_xlabel("Time (min post-fertilization)", fontsize=10)
    ax.set_ylabel("Dorsal epidermal cells", fontsize=10)
    ax.set_title("A. Co-cluster probability", fontsize=12, fontweight="bold", pad=20)
    time_ticks = [0, 10, 20, 29]
    time_labels = [220, 230, 240, 250]
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels)
    cell_ticks = [0, 4, 9, 14, 19]
    cell_labels = ["DC01", "DC05", "DC10", "DC15", "DC20"]
    ax.set_yticks(cell_ticks)
    ax.set_yticklabels(cell_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Co-cluster\nprobability", fontsize=9)
    fig.tight_layout()
    return fig, ax


# Panel B 工厂函数


def create_panel_b():
    fig, ax = plt.subplots(figsize=(3.5, 3))
    n_cells = 20
    n_timepoints = 30
    left_row_y = np.linspace(-15, 15, 10)
    right_row_y = np.linspace(-15, 15, 10)
    initial_positions = np.zeros((20, 2))
    initial_positions[:10, 0] = -5
    initial_positions[:10, 1] = left_row_y
    initial_positions[10:, 0] = 5
    initial_positions[10:, 1] = right_row_y
    trajectories = np.zeros((20, 30, 2))
    for i in range(20):
        start_pos = initial_positions[i]
        if i < 10:
            target_x = np.random.uniform(2, 8)
        else:
            target_x = np.random.uniform(-8, -2)
        target_y = start_pos[1] + np.random.normal(0, 2)
        x_traj = start_pos[0] + (target_x - start_pos[0]) * (
            1 - np.exp(-np.linspace(0, 3, 30))
        )
        y_traj = start_pos[1] + (target_y - start_pos[1]) * np.linspace(0, 1, 30)
        x_traj += np.random.normal(0, 0.5, 30)
        y_traj += np.random.normal(0, 0.3, 30)
        trajectories[i, :, 0] = x_traj
        trajectories[i, :, 1] = y_traj
    for i in range(20):
        traj = trajectories[i]
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="viridis", alpha=0.7, linewidth=1.5)
        lc.set_array(np.linspace(0, 1, len(segments)))
        ax.add_collection(lc)
        ax.scatter(traj[0, 0], traj[0, 1], c="green", s=30, marker="o", alpha=0.8)
        ax.scatter(traj[-1, 0], traj[-1, 1], c="red", s=30, marker="s", alpha=0.8)
    theta = np.linspace(0, 2 * np.pi, 100)
    embryo_x = 12 * np.cos(theta)
    embryo_y = 20 * np.sin(theta)
    ax.plot(embryo_x, embryo_y, "k-", alpha=0.3, linewidth=2)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax.text(1, 18, "Dorsal midline", rotation=90, fontsize=8, alpha=0.7)
    ax.set_xlabel("Anterior-Posterior (μm)", fontsize=10)
    ax.set_ylabel("Left-Right (μm)", fontsize=10)
    ax.set_title(
        "B. Cell trajectory visualization", fontsize=12, fontweight="bold", pad=20
    )
    ax.set_xlim(-15, 15)
    ax.set_ylim(-25, 25)
    ax.set_aspect("equal")
    ax.scatter([], [], c="green", s=30, marker="o", label="Start (220 min)")
    ax.scatter([], [], c="red", s=30, marker="s", label="End (250 min)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig, ax


# Panel C 工厂函数


def create_panel_c():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    time_minutes = np.linspace(220, 250, 30)
    shape_irregularity = np.zeros((20, 30))
    for i in range(20):
        baseline = np.random.uniform(0.3, 0.5)
        peak_increase = np.random.uniform(0.8, 1.5)
        profile = baseline + peak_increase * np.exp(
            -0.5 * ((time_minutes - 235) / 8) ** 2
        )
        profile += np.random.normal(0, 0.05, 30)
        shape_irregularity[i, :] = profile
    for i in range(20):
        ax.plot(
            time_minutes,
            shape_irregularity[i, :],
            color=colors["light_blue"],
            alpha=0.3,
            linewidth=0.8,
        )
    mean_irregularity = np.mean(shape_irregularity, axis=0)
    sem_irregularity = np.std(shape_irregularity, axis=0) / np.sqrt(20)
    ax.plot(
        time_minutes,
        mean_irregularity,
        color=colors["primary"],
        linewidth=2.5,
        label="Mean ± SEM",
    )
    ax.fill_between(
        time_minutes,
        mean_irregularity - sem_irregularity,
        mean_irregularity + sem_irregularity,
        color=colors["primary"],
        alpha=0.3,
    )
    ax.axvspan(230, 240, alpha=0.2, color="yellow", label="Co-cluster window")
    ax.set_xlabel("Time (min post-fertilization)", fontsize=10)
    ax.set_ylabel("Shape irregularity index", fontsize=10)
    ax.set_title(
        "C. Morphological feature dynamics", fontsize=12, fontweight="bold", pad=20
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# Panel D 工厂函数


def create_panel_d():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    time_points_vel = np.linspace(220, 250, 15)
    n_cocluster = 20
    n_control = 30
    velocity_data = []
    time_labels = []
    group_labels = []
    for t_idx, t in enumerate(time_points_vel):
        if 230 <= t <= 240:
            cocluster_vel = np.random.normal(2.5, 1.2, n_cocluster)
        else:
            cocluster_vel = np.random.normal(0.2, 0.8, n_cocluster)
        control_vel = np.random.normal(0.1, 0.5, n_control)
        velocity_data.extend(cocluster_vel)
        velocity_data.extend(control_vel)
        time_labels.extend([t] * (n_cocluster + n_control))
        group_labels.extend(["Co-cluster"] * n_cocluster + ["Control"] * n_control)
    df_vel = pd.DataFrame(
        {"Time": time_labels, "Velocity": velocity_data, "Group": group_labels}
    )
    sns.boxplot(
        data=df_vel,
        x="Time",
        y="Velocity",
        hue="Group",
        ax=ax,
        palette=[colors["primary"], colors["warning"]],
    )
    ax.set_xlabel("Time (min post-fertilization)", fontsize=10)
    ax.set_ylabel("Cross-midline velocity (μm/min)", fontsize=10)
    ax.set_title("D. Velocity field analysis", fontsize=12, fontweight="bold", pad=20)
    ax.tick_params(axis="x", rotation=45)
    sig_times = [230, 235, 240]
    for t in sig_times:
        if t in time_points_vel:
            t_idx = list(time_points_vel).index(t)
            ax.text(
                t_idx,
                4.5,
                "***",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def create_figure2_dorsal_intercalation():
    """
    Figure 2: Dorsal Intercalation Discovery and Validation
    180mm × 180mm (双栏)
    """

    # Create figure with subplots
    fig = plt.figure(figsize=(7.09, 7.09))  # 180mm = 7.09 inches

    # Define grid layout
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[2, 2, 1.5],
        width_ratios=[1, 1, 0.1],
        hspace=0.3,
        wspace=0.3,
    )

    # ===== Panel A: Co-cluster Heat Map =====
    ax_a = fig.add_subplot(gs[0, 0])

    # Simulate co-cluster data (20 cells × 30 time points)
    np.random.seed(42)
    time_points = np.linspace(220, 250, 30)
    cell_ids = [f"DC{i:02d}" for i in range(1, 21)]  # Dorsal cells

    # Create synthetic co-cluster probability matrix
    co_cluster_prob = np.zeros((20, 30))

    # Define intercalation event (Gaussian-like temporal profile)
    peak_time = 15  # Peak at ~235 minutes
    temporal_profile = np.exp(-0.5 * ((np.arange(30) - peak_time) / 5) ** 2)

    # Different cells have different participation patterns
    for i in range(20):
        # Add some cell-specific variation
        cell_peak = peak_time + np.random.normal(0, 2)
        cell_width = np.random.uniform(4, 6)
        cell_intensity = np.random.uniform(0.7, 1.0)

        cell_profile = cell_intensity * np.exp(
            -0.5 * ((np.arange(30) - cell_peak) / cell_width) ** 2
        )
        co_cluster_prob[i, :] = cell_profile

        # Add some noise
        co_cluster_prob[i, :] += np.random.normal(0, 0.1, 30)
        co_cluster_prob[i, :] = np.clip(co_cluster_prob[i, :], 0, 1)

    # Create heatmap
    im_a = ax_a.imshow(
        co_cluster_prob,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )

    # Customize axes
    ax_a.set_xlabel("Time (min post-fertilization)", fontsize=10)
    ax_a.set_ylabel("Dorsal epidermal cells", fontsize=10)
    ax_a.set_title("A. Co-cluster probability", fontsize=12, fontweight="bold", pad=20)

    # Set ticks
    time_ticks = [0, 10, 20, 29]
    time_labels = [220, 230, 240, 250]
    ax_a.set_xticks(time_ticks)
    ax_a.set_xticklabels(time_labels)

    cell_ticks = [0, 4, 9, 14, 19]
    cell_labels = ["DC01", "DC05", "DC10", "DC15", "DC20"]
    ax_a.set_yticks(cell_ticks)
    ax_a.set_yticklabels(cell_labels)

    # Add colorbar
    cbar_ax_a = fig.add_subplot(gs[0, 2])
    cbar_a = plt.colorbar(im_a, cax=cbar_ax_a)
    cbar_a.set_label("Co-cluster\nprobability", fontsize=9)

    # ===== Panel B: Cell Trajectory Visualization =====
    ax_b = fig.add_subplot(gs[0, 1])

    # Simulate cell trajectories during intercalation
    n_cells = 20
    n_timepoints = 30

    # Initial positions: two rows of cells
    left_row_y = np.linspace(-15, 15, 10)  # Left side cells
    right_row_y = np.linspace(-15, 15, 10)  # Right side cells

    initial_positions = np.zeros((20, 2))
    initial_positions[:10, 0] = -5  # Left row x-position
    initial_positions[:10, 1] = left_row_y
    initial_positions[10:, 0] = 5  # Right row x-position
    initial_positions[10:, 1] = right_row_y

    # Simulate intercalation movement
    trajectories = np.zeros((20, 30, 2))

    for i in range(20):
        start_pos = initial_positions[i]

        if i < 10:  # Left side cells move right
            target_x = np.random.uniform(2, 8)
        else:  # Right side cells move left
            target_x = np.random.uniform(-8, -2)

        target_y = start_pos[1] + np.random.normal(0, 2)

        # Create smooth trajectory
        x_traj = start_pos[0] + (target_x - start_pos[0]) * (
            1 - np.exp(-np.linspace(0, 3, 30))
        )
        y_traj = start_pos[1] + (target_y - start_pos[1]) * np.linspace(0, 1, 30)

        # Add some noise
        x_traj += np.random.normal(0, 0.5, 30)
        y_traj += np.random.normal(0, 0.3, 30)

        trajectories[i, :, 0] = x_traj
        trajectories[i, :, 1] = y_traj

    # Plot trajectories with color gradient
    for i in range(20):
        traj = trajectories[i]

        # Create line collection with color gradient
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap="viridis", alpha=0.7, linewidth=1.5)
        lc.set_array(np.linspace(0, 1, len(segments)))
        ax_b.add_collection(lc)

        # Mark start and end positions
        ax_b.scatter(traj[0, 0], traj[0, 1], c="green", s=30, marker="o", alpha=0.8)
        ax_b.scatter(traj[-1, 0], traj[-1, 1], c="red", s=30, marker="s", alpha=0.8)

    # Add embryo outline
    theta = np.linspace(0, 2 * np.pi, 100)
    embryo_x = 12 * np.cos(theta)
    embryo_y = 20 * np.sin(theta)
    ax_b.plot(embryo_x, embryo_y, "k-", alpha=0.3, linewidth=2)

    # Add midline
    ax_b.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax_b.text(1, 18, "Dorsal midline", rotation=90, fontsize=8, alpha=0.7)

    ax_b.set_xlabel("Anterior-Posterior (μm)", fontsize=10)
    ax_b.set_ylabel("Left-Right (μm)", fontsize=10)
    ax_b.set_title(
        "B. Cell trajectory visualization", fontsize=12, fontweight="bold", pad=20
    )
    ax_b.set_xlim(-15, 15)
    ax_b.set_ylim(-25, 25)
    ax_b.set_aspect("equal")

    # Add legend
    ax_b.scatter([], [], c="green", s=30, marker="o", label="Start (220 min)")
    ax_b.scatter([], [], c="red", s=30, marker="s", label="End (250 min)")
    ax_b.legend(loc="upper right", fontsize=8)

    # ===== Panel C: Shape Irregularity Dynamics =====
    ax_c = fig.add_subplot(gs[1, :2])

    # Simulate shape irregularity data
    time_minutes = np.linspace(220, 250, 30)

    # Individual cell traces
    shape_irregularity = np.zeros((20, 30))

    for i in range(20):
        # Baseline irregularity
        baseline = np.random.uniform(0.3, 0.5)

        # Increase during intercalation (around 235 min)
        peak_increase = np.random.uniform(0.8, 1.5)

        # Create temporal profile
        profile = baseline + peak_increase * np.exp(
            -0.5 * ((time_minutes - 235) / 8) ** 2
        )

        # Add noise
        profile += np.random.normal(0, 0.05, 30)

        shape_irregularity[i, :] = profile

    # Plot individual traces (light)
    for i in range(20):
        ax_c.plot(
            time_minutes,
            shape_irregularity[i, :],
            color=colors["light_blue"],
            alpha=0.3,
            linewidth=0.8,
        )

    # Plot mean ± SEM
    mean_irregularity = np.mean(shape_irregularity, axis=0)
    sem_irregularity = np.std(shape_irregularity, axis=0) / np.sqrt(20)

    ax_c.plot(
        time_minutes,
        mean_irregularity,
        color=colors["primary"],
        linewidth=2.5,
        label="Mean ± SEM",
    )
    ax_c.fill_between(
        time_minutes,
        mean_irregularity - sem_irregularity,
        mean_irregularity + sem_irregularity,
        color=colors["primary"],
        alpha=0.3,
    )

    # Highlight co-cluster active window
    ax_c.axvspan(230, 240, alpha=0.2, color="yellow", label="Co-cluster window")

    ax_c.set_xlabel("Time (min post-fertilization)", fontsize=10)
    ax_c.set_ylabel("Shape irregularity index", fontsize=10)
    ax_c.set_title(
        "C. Morphological feature dynamics", fontsize=12, fontweight="bold", pad=20
    )
    ax_c.legend(loc="upper right", fontsize=8)
    ax_c.grid(True, alpha=0.3)

    # ===== Panel D: Velocity Field Analysis =====
    ax_d = fig.add_subplot(gs[2, :2])

    # Simulate cross-midline velocity data
    time_points_vel = np.linspace(220, 250, 15)  # Fewer time points for clarity

    # Co-cluster cells vs control cells
    n_cocluster = 20
    n_control = 30

    velocity_data = []
    time_labels = []
    group_labels = []

    for t_idx, t in enumerate(time_points_vel):
        # Co-cluster cells: higher cross-midline velocity during event
        if 230 <= t <= 240:  # During intercalation
            cocluster_vel = np.random.normal(2.5, 1.2, n_cocluster)
        else:
            cocluster_vel = np.random.normal(0.2, 0.8, n_cocluster)

        # Control cells: minimal cross-midline velocity
        control_vel = np.random.normal(0.1, 0.5, n_control)

        # Add to data
        velocity_data.extend(cocluster_vel)
        velocity_data.extend(control_vel)

        time_labels.extend([t] * (n_cocluster + n_control))
        group_labels.extend(["Co-cluster"] * n_cocluster + ["Control"] * n_control)

    # Create DataFrame for seaborn
    df_vel = pd.DataFrame(
        {"Time": time_labels, "Velocity": velocity_data, "Group": group_labels}
    )

    # Create box plot
    sns.boxplot(
        data=df_vel,
        x="Time",
        y="Velocity",
        hue="Group",
        ax=ax_d,
        palette=[colors["primary"], colors["warning"]],
    )

    # Customize
    ax_d.set_xlabel("Time (min post-fertilization)", fontsize=10)
    ax_d.set_ylabel("Cross-midline velocity (μm/min)", fontsize=10)
    ax_d.set_title("D. Velocity field analysis", fontsize=12, fontweight="bold", pad=20)

    # Rotate x-axis labels
    ax_d.tick_params(axis="x", rotation=45)

    # Add significance markers (simulate statistical test results)
    sig_times = [230, 235, 240]  # Times with significant differences
    for t in sig_times:
        if t in time_points_vel:
            t_idx = list(time_points_vel).index(t)
            ax_d.text(
                t_idx,
                4.5,
                "***",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax_d.legend(loc="upper left", fontsize=8)
    ax_d.grid(True, alpha=0.3)

    # Final adjustments
    plt.tight_layout()
    return fig


# Create the figure
# fig2 = create_figure2_dorsal_intercalation()
# plt.show()

# Save in high resolution
# fig2.savefig('Figure2_DorsalIntercalation.pdf', dpi=300, bbox_inches='tight',
#              facecolor='white', edgecolor='none')
# fig2.savefig('Figure2_DorsalIntercalation.tiff', dpi=300, bbox_inches='tight',
#              facecolor='white', edgecolor='none')

if __name__ == "__main__":
    fig_a, _ = create_panel_a()
    fig_a.savefig(
        "pic/PanelA.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig_a)

    fig_b, _ = create_panel_b()
    fig_b.savefig(
        "pic/PanelB.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig_b)

    fig_c, _ = create_panel_c()
    fig_c.savefig(
        "pic/PanelC.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig_c)

    fig_d, _ = create_panel_d()
    fig_d.savefig(
        "pic/PanelD.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig_d)
