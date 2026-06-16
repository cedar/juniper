import os
import numpy as np
import matplotlib.pyplot as plt
from . import util
from matplotlib import colors
from matplotlib import gridspec
import time
from .util import tprint

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),   
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_history(num_ticks, history, save_plot, step_names):
    if len(history) == 0:
        print("Nothing to plot")
        return
    cmap = truncate_colormap(plt.get_cmap('hsv_r'), 0.29, 1)
    
    num_steps = len(history[0])
    cols = min(10, num_ticks) + 1
    rows_per_step = ((num_ticks-1) // (cols-1) + 1)
    rows = rows_per_step * num_steps

    fig = plt.figure(figsize=(2*cols, 2*rows))
    for step_idx in range(num_steps):
        ax_label = fig.add_subplot(rows, cols, (step_idx * rows_per_step) * cols + 1)
        label = step_names[step_idx]
        if len(label) > 8:
            label = label.replace(".", ".\n")
        ax_label.text(0.5, 0.5, label, fontsize=12, ha='center')
        ax_label.set_axis_off()
        colorbar_displayed = False

        if not np.any([step_mats[step_idx] is not None for step_mats in history]):
            continue
        step_history_notna = [step_mats[step_idx] for step_mats in history if step_mats[step_idx] is not None]
        vmin = min([np.min(mat) for mat in step_history_notna])
        vmax = max([np.max(mat) for mat in step_history_notna])

        for i in range((cols - 1) * rows_per_step):
            if i >= num_ticks:
                ax = fig.add_subplot(rows, cols, step_idx * cols + i + 2)
                ax.set_axis_off()
                continue
            row = (i // (cols-1) + step_idx * rows_per_step)
            col = i % (cols-1)
            dimensionality = len(history[i][step_idx].shape)
            if dimensionality >= 3:
                ax = fig.add_subplot(rows, cols, row * cols + col + 2, projection='3d')
            else:
                ax = fig.add_subplot(rows, cols, row * cols + col + 2)
                
            # Plot data
            data = history[i][step_idx]
            im = None
            if dimensionality > 3:
                # reduce dimensionality to 3
                # sum last axis until its 3d
                while len(data.shape) > 3:
                    data = np.sum(data, axis=-1)
            if dimensionality == 1:
                ax.plot(data)
            elif dimensionality == 2:
                im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
            elif dimensionality >= 3:
                # Let markersize depend on the value of the matrix
                markersize_min = 2
                markersizes = markersize_min + 20 * data
                markersizes = markersizes.at[markersizes < 0].set(markersize_min)
                im = ax.scatter(*np.where(data > np.min(data) - 1), c=data, s = markersizes, vmin=vmin, vmax=vmax, cmap=cmap)
            if step_idx == 0:
                ax.set_title(f"t={i+1}")
            if im is not None and not colorbar_displayed:
                colorbar_displayed = True
                fig.colorbar(im, ax=ax_label, orientation='vertical')
    plt.tight_layout()
    if save_plot:
        folder = os.path.join(util.root(), "output")
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, f"plot_{int(time.time())}.png"))
        tprint("Plot done")
    else:
        plt.show()


def plot_steps(
    recordings,
    time_axis=None,
    snapshot_indices=None,
    scalar_group_keys=None,
    figsize=(10, 4),
    step_names=None,
):
    """Plot scalar traces and snapshots from a time-major recording.

    recordings may be either a raw ``recording[time][key]`` list, a step-major
    list of arrays, or a Recording object with ``recording`` and
    ``key_strings`` attributes.

    time_axis:
        Optional x-axis values for simulation steps. Defaults to step indices.
    snapshot_indices:
        Time-step indices used for non-scalar snapshot plots.
    scalar_group_keys:
        Groups of recording keys that should share one scalar trace axis. When
        plotting raw arrays, integer step indices can be used instead.
    """
    if hasattr(recordings, "recording"):
        if step_names is None and hasattr(recordings, "key_strings"):
            step_names = recordings.key_strings
        recordings = recordings.recording

    if len(recordings) == 0:
        raise ValueError("recordings is empty.")

    steps = _recordings_to_step_arrays(recordings)
    T = steps[0].shape[0]

    if time_axis is None:
        time_axis = np.arange(T)
    else:
        time_axis = np.asarray(time_axis)
        if time_axis.shape[0] != T:
            raise ValueError(f"len(time_axis)={time_axis.shape[0]} does not match T={T}.")

    step_is_scalar = [_is_scalar_step(step) for step in steps]
    scalar_indices = [i for i, is_scalar in enumerate(step_is_scalar) if is_scalar]
    non_scalar_indices = [i for i, is_scalar in enumerate(step_is_scalar) if not is_scalar]

    scalar_groups = _resolve_scalar_group_indices(scalar_group_keys, step_names)
    _validate_scalar_groups(scalar_groups, step_is_scalar)
    scalar_indices_in_groups = {idx for group in scalar_groups for idx in group}
    ungrouped_scalar_indices = [i for i in scalar_indices if i not in scalar_indices_in_groups]

    if snapshot_indices is None:
        snapshot_indices = [T - 1] if len(non_scalar_indices) > 0 else []
    snapshot_indices = list(snapshot_indices)

    n_rows = len(scalar_groups) + len(ungrouped_scalar_indices) + len(non_scalar_indices)
    if n_rows == 0:
        raise ValueError("No steps to plot.")
    n_cols = max(1, len(snapshot_indices))

    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_rows))
    grid = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    current_row = 0

    for group_idx, group in enumerate(scalar_groups):
        ax = fig.add_subplot(grid[current_row, :])
        ys = []
        for idx in group:
            y = _scalar_series(steps[idx], T)
            ys.append(y)
            ax.plot(time_axis, y, label=_step_label(idx, step_names))
        _format_scalar_axis(ax, time_axis, np.asarray(ys))
        ax.set_title("Scalar group: " + ", ".join(_step_label(idx, step_names) for idx in group))
        ax.legend()
        current_row += 1

    for idx in ungrouped_scalar_indices:
        ax = fig.add_subplot(grid[current_row, :])
        y = _scalar_series(steps[idx], T)
        ax.plot(time_axis, y)
        _format_scalar_axis(ax, time_axis, y)
        ax.set_title(_step_label(idx, step_names))
        current_row += 1

    for idx in non_scalar_indices:
        if len(snapshot_indices) == 0:
            current_row += 1
            continue
        for col, snap in enumerate(snapshot_indices):
            if snap < 0 or snap >= T:
                raise IndexError(f"Snapshot index {snap} outside [0, {T - 1}].")
            ax = fig.add_subplot(grid[current_row, col])
            _plot_snapshot(fig, ax, np.asarray(steps[idx][snap]))
            ax.set_title(f"{_step_label(idx, step_names)}, step={snap}, x={time_axis[snap]}")
        current_row += 1

    fig.tight_layout()
    return fig


def _recordings_to_step_arrays(recordings):
    if isinstance(recordings[0], (list, tuple)):
        T = len(recordings)
        n_steps = len(recordings[0])
        return [np.asarray([recordings[t][step_idx] for t in range(T)]) for step_idx in range(n_steps)]

    if hasattr(recordings[0], "shape"):
        return [np.asarray(step) for step in recordings]

    raise TypeError("recordings must be time-major list-of-lists or step-major arrays.")


def _is_scalar_step(step):
    return step.ndim == 1 or (step.ndim == 2 and step.shape[1] == 1)


def _scalar_series(step, T):
    if step.ndim == 1:
        return step
    return step.reshape(T, -1)[:, 0]


def _step_label(idx, step_names):
    return f"Step {idx}" if step_names is None else step_names[idx]


def _resolve_scalar_group_indices(scalar_group_keys, step_names):
    if scalar_group_keys is None:
        return []

    groups = []
    for group in scalar_group_keys:
        group_indices = []
        for key in group:
            if isinstance(key, int):
                group_indices.append(key)
                continue
            if step_names is None:
                raise ValueError("scalar_group_keys must use integer indices when step_names are unavailable.")
            key_name = key if isinstance(key, str) else key.get_path_str()
            group_indices.append(step_names.index(key_name))
        groups.append(group_indices)
    return groups


def _validate_scalar_groups(scalar_groups, step_is_scalar):
    for group in scalar_groups:
        for idx in group:
            if idx not in range(len(step_is_scalar)):
                raise IndexError(f"Index {idx} in scalar_groups is outside recorded steps.")
            if not step_is_scalar[idx]:
                raise ValueError(f"Step {idx} in scalar_groups is not scalar.")


def _format_scalar_axis(ax, t, y):
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    pad = max(0.1, 0.05 * (y_max - y_min))
    ax.plot([], [])
    ax.axhline(y=0, color="black", linestyle="dashed")
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("t")
    ax.set_ylabel("value")


def _plot_snapshot(fig, ax, snapshot_data):
    if snapshot_data.ndim == 1:
        x = np.arange(snapshot_data.shape[0])
        ax.plot(x, snapshot_data)
        ax.axhline(y=0, color="black", linestyle="dashed")
        ax.set_xlim(0, max(0, len(x) - 1))
        ax.set_xlabel("index")
        ax.set_ylabel("value")
    elif snapshot_data.ndim == 2:
        im = ax.imshow(snapshot_data, origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax)
    elif snapshot_data.ndim == 3 and snapshot_data.shape[2] == 3:
        ax.imshow(_normalize_rgb(snapshot_data), origin="lower", aspect="auto")
    elif snapshot_data.ndim == 3 and snapshot_data.shape[2] > 3:
        _plot_channel_grid(fig, ax, snapshot_data)
    else:
        flat = snapshot_data.ravel()
        ax.plot(np.arange(flat.shape[0]), flat)
        ax.set_xlabel("flat index")
        ax.set_ylabel("value")


def _normalize_rgb(data):
    data = np.asarray(data)
    if np.nanmax(data) > 1:
        return data / 255
    return data


def _plot_channel_grid(fig, ax, snapshot_data):
    channels = snapshot_data.shape[2]
    ncols = int(np.ceil(np.sqrt(channels)))
    nrows = int(np.ceil(channels / ncols))
    ax.axis("off")

    parent_bbox = ax.get_position()
    pad = 0.01
    cell_w = (parent_bbox.width - pad * (ncols - 1)) / ncols
    cell_h = (parent_bbox.height - pad * (nrows - 1)) / nrows
    vmin = np.nanmin(snapshot_data)
    vmax = np.nanmax(snapshot_data)
    im = None

    for channel in range(channels):
        row = channel // ncols
        col = channel % ncols
        left = parent_bbox.x0 + col * (cell_w + pad)
        bottom = parent_bbox.y1 - (row + 1) * cell_h - row * pad
        channel_ax = fig.add_axes([left, bottom, cell_w, cell_h])
        im = channel_ax.imshow(snapshot_data[:, :, channel], origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        channel_ax.set_title(f"ch {channel}", fontsize=8)
        channel_ax.set_xticks([])
        channel_ax.set_yticks([])

    if im is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
