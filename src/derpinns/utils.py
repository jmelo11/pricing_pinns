import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

blue_palette = ["#08306B", "#08519C",
                "#3182BD", "#6BAED6", "#9ECAE1", "#C6DBEF"]


def _moving_average(arr, window):
    """Simple moving average that keeps the original length."""
    if window <= 1:
        return np.asarray(arr)
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    # pad the left side so lengths match
    left_pad = np.full(window - 1, smoothed[0])
    return np.concatenate([left_pad, smoothed])


def plot_loss(
    loss_history,
    save_path=None,
    backend="plotly",
    fig_size=(800, 600),
    smooth=True,
    smooth_window=50,
):
    """
    Plot loss history on a logarithmic y‑axis.

    Parameters
    ----------
    loss_history : dict
        {
          'interior_loss': list or np.ndarray,
          'boundary_loss': list or np.ndarray,
          'initial_loss' : list or np.ndarray
        }
    smooth : bool              – turn smoothing on/off
    smooth_window : int >= 1   – moving‑average window size in steps
    """
    # --- grab & optionally smooth the data ----------------------------------
    interior = np.asarray(loss_history["interior_loss"])
    boundary = np.asarray(loss_history["boundary_loss"])
    initial = np.asarray(loss_history["initial_loss"])   # fixed typo

    if smooth:
        interior = _moving_average(interior, smooth_window)
        boundary = _moving_average(boundary, smooth_window)
        initial = _moving_average(initial,  smooth_window)

    x = np.arange(len(interior))

    # --- Plotly backend ------------------------------------------------------
    if backend.lower() == "plotly":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=interior, mode="lines",
                                 name="PDE Loss",
                                 line=dict(color=blue_palette[0])))
        fig.add_trace(go.Scatter(x=x, y=boundary, mode="lines",
                                 name="Boundary Loss",
                                 line=dict(color=blue_palette[2])))
        fig.add_trace(go.Scatter(x=x, y=initial, mode="lines",
                                 name="Initial Loss",
                                 line=dict(color=blue_palette[4])))

        fig.update_layout(
            title=f"Loss History",
            xaxis_title="Epoch",
            yaxis_title="Loss (log‑scale)",
            yaxis_type="log",
            template="plotly_white",
            width=fig_size[0], height=fig_size[1],
            legend=dict(orientation="h", x=0.5, y=1,
                        xanchor="center", yanchor="top")
        )
        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()

    # --- Matplotlib backend --------------------------------------------------
    elif backend.lower() == "matplotlib":
        plt.figure(figsize=(fig_size[0] / 100, fig_size[1] / 100))
        plt.plot(x, interior, label="PDE Loss",     color=blue_palette[0])
        plt.plot(x, boundary, label="Boundary Loss", color=blue_palette[2])
        plt.plot(x, initial,  label="Initial Loss",  color=blue_palette[4])
        plt.yscale("log")
        plt.title(f"Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()


def compare_loss_histories(
    runs,
    labels=None,
    backend="plotly",
    fig_size=(900, 600),
    smooth=True,
    smooth_window=50,
):
    """
    Compare N loss‑history dicts on one figure.

    Parameters
    ----------
    runs : list of dict     – each dict must contain keys
                              'interior_loss', 'boundary_loss', 'initial_loss'
    labels : list of str    – legend labels, default "Run 1", "Run 2", ...
    """

    n_runs = len(runs)
    assert n_runs > 0, "runs list cannot be empty"
    if labels is None:
        labels = [f"Run {i+1}" for i in range(n_runs)]
    assert len(labels) == n_runs, "`labels` length must match `runs` length"

    # colour per loss‑type, dash‑style per run
    colours = {"pde": "#1f77b4", "bdy": "#5fa2d0",
               "init": "#9cc3e6", "tot": "#08306B"}
    dashes = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

    # prepare data -----------------------------------------------------------
    def prep(d):
        i = np.asarray(d["interior_loss"])
        b = np.asarray(d["boundary_loss"])
        n = np.asarray(d["initial_loss"])
        t = i+b+n
        if smooth:
            i, b, n, t = (_moving_average(x, smooth_window)
                          for x in (i, b, n, t))
        return i, b, n, t

    processed = [prep(r) for r in runs]
    x = np.arange(len(processed[0][0]))   # assume equal length

    # ---------------------------------------------------------------- Plotly
    if backend.lower() == "plotly":
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=("wInterior Loss",
                            "Boundary Loss", "Initial Loss", "Total Loss")
        )

        def add(row, y_values, colour, dash, label, showlegend):
            fig.add_trace(
                go.Scatter(
                    x=x, y=y_values, mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=showlegend,
                    line=dict(color=colour, dash=dash)
                ),
                row=row, col=1,
            )

        for run_idx, (ia, ba, na, ta) in enumerate(processed):
            dash = dashes[run_idx % len(dashes)]
            label = labels[run_idx]
            show = True  # only show legend once per run (row 1)
            add(1, ia, colours["pde"],  dash, label, show)
            add(2, ba, colours["bdy"],  dash, label, False)
            add(3, na, colours["init"], dash, label, False)
            add(4, ta, colours["tot"], dash, label, False)

        fig.update_yaxes(type="log")
        fig.update_layout(
            height=fig_size[1], width=fig_size[0],
            title_text="Loss Comparison", legend_title="Run"
        )
        fig.show()

    # ------------------------------------------------------------- Matplotlib
    else:
        _, axes = plt.subplots(3, 1,
                               figsize=(fig_size[0] / 100, fig_size[1] / 100),
                               sharex=True)

        titles = ["PDE / Interior Loss", "Boundary Loss", "Initial Loss"]

        for run_idx, (ia, ba, na) in enumerate(processed):
            dash = dashes[run_idx % len(dashes)]
            label = labels[run_idx]
            axes[0].plot(x, ia, label=label,
                         color=colours["pde"],  linestyle=dash)
            axes[1].plot(x, ba, label=label,
                         color=colours["bdy"],  linestyle=dash)
            axes[2].plot(x, na, label=label,
                         color=colours["init"], linestyle=dash)

        for ax, title in zip(axes, titles):
            ax.set_yscale("log")
            ax.set_title(title, fontsize=10)

        axes[-1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss (log scale)")
        axes[0].legend(loc="upper right")
        plt.tight_layout()
        plt.show()


def plot_all_total_losses(bench, backend='plotly', save_path=None, fig_size=(800, 600)):
    """
    Plots 'total_loss' curves for each (assets, nn_shape) configuration 
    on a single figure, for both LBFGS and ADAM if present.
    """
    if backend == 'plotly':
        fig = go.Figure()

        # Use blue shades for the configurations.
        for idx, (assets, shapes_dict) in enumerate(bench.items()):
            for jdx, (nn_shape, data) in enumerate(shapes_dict.items()):
                color = blue_palette[(idx + jdx) % len(blue_palette)]
                # Plot LBFGS total_loss if present
                if 'lbfgs_loss' in data:
                    lbfgs_losses = data['lbfgs_loss']['total_loss']
                    x_vals = np.arange(len(lbfgs_losses))
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=lbfgs_losses,
                        mode='lines',
                        name=f"LBFGS (assets={assets}, shape={nn_shape})",
                        line=dict(color=color)
                    ))

                # Plot ADAM total_loss if present
                if 'adam_loss' in data:
                    adam_losses = data['adam_loss']['total_loss']
                    x_vals = np.arange(len(adam_losses))
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=adam_losses,
                        mode='lines',
                        name=f"ADAM (assets={assets}, shape={nn_shape})",
                        line=dict(color=color, dash='dash')
                    ))

        fig.update_layout(
            title="Total Loss - All Configurations",
            xaxis_title="Epoch",
            yaxis_title="Total Loss",
            yaxis_type="log",  # log scale on y-axis
            legend_title="Configuration",
            template="plotly_white",
            width=fig_size[0],
            height=fig_size[1],
            legend=dict(
                orientation="h",
                xanchor="center",
                yanchor="top",
                x=0.5,
                y=1
            ),

        )

        if save_path:
            fig.write_image(save_path)

        fig.show()

    else:
        plt.figure(figsize=(8, 6))

        for assets, shapes_dict in bench.items():
            for nn_shape, data in shapes_dict.items():
                color = blue_palette[(
                    int(assets) + int(nn_shape)) % len(blue_palette)]
                if 'lbfgs_loss' in data:
                    lbfgs_losses = data['lbfgs_loss']['total_loss']
                    plt.plot(lbfgs_losses,
                             label=f"LBFGS (assets={assets}, size={nn_shape})",
                             color=color)
                if 'adam_loss' in data:
                    adam_losses = data['adam_loss']['total_loss']
                    plt.plot(adam_losses,
                             label=f"ADAM (assets={assets}, size={nn_shape})",
                             color=color, linestyle='--')
        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")
        plt.title("Total Loss - All Configurations")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def plot_max_errors_continued(bench, assets, save_path=None, fig_size=(800, 600)):
    """
    Plots the concatenated max error curves for sequential training phases (first Adam, then LBFGS)
    for all network configurations for a given Assets.
    """
    fig = go.Figure()

    # Loop over each network configuration for the given Assets.
    for idx, (nn_shape, config) in enumerate(bench[assets].items()):
        base_color = blue_palette[idx % len(blue_palette)]
        # Retrieve the max error histories for each phase (if they exist)
        adam_errors = config.get('adam_loss', {}).get('max_error', None)
        lbfgs_errors = config.get('lbfgs_loss', {}).get('max_error', None)

        total_length_adam = 0
        # Plot Adam phase with a dashed line
        if adam_errors is not None:
            adam_errors = np.array(adam_errors)
            total_length_adam = len(adam_errors)
            x_adam = np.arange(total_length_adam)
            fig.add_trace(go.Scatter(
                x=x_adam,
                y=adam_errors,
                mode='lines',
                name=f"{nn_shape} (Adam)",
                line=dict(color=base_color, dash='dash')
            ))

        # Plot LBFGS phase with a solid line, continuing from the Adam phase
        if lbfgs_errors is not None:
            lbfgs_errors = np.array(lbfgs_errors)
            x_lbfgs = np.arange(len(lbfgs_errors)) + total_length_adam
            fig.add_trace(go.Scatter(
                x=x_lbfgs,
                y=lbfgs_errors,
                mode='lines',
                name=f"{nn_shape} (LBFGS)",
                line=dict(color=base_color, dash='solid')
            ))

    fig.update_layout(
        title=f"Max Error over Iterations (Adam then LBFGS) for {assets} Assets",
        xaxis_title="Iteration",
        yaxis_title="Max Error",
        showlegend=True,
        width=fig_size[0],
        height=fig_size[1],
        legend=dict(
            orientation="h",
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=1
        ),
        template="plotly_white"
    )

    if save_path:
        fig.write_image(save_path)
    fig.show()


def plot_relative_avg_errors(bench, save_path=None, fig_size=(800, 600)):
    '''
    Bar plot where:
      - x-axis: Assets
      - y-axis: average relative error as a percentage
      - Each color (trace) corresponds to a network size.
    '''
    # Sort the asset counts to have a consistent x-axis order.
    asset_counts = sorted(bench.keys(), key=lambda x: int(x))
    # Gather all network sizes that appear in bench.
    all_nn_shapes = set()
    for shapes_dict in bench.values():
        all_nn_shapes.update(shapes_dict.keys())
    all_nn_shapes = sorted(all_nn_shapes)

    fig = go.Figure()

    # For each network size, collect the average error for each asset count.
    for i, nn_shape in enumerate(all_nn_shapes):
        errors = []
        for assets in asset_counts:
            if nn_shape in bench[assets]:
                avg_error = np.abs(bench[assets][nn_shape]['errors']['avg_nn_price'] -
                                   bench[assets][nn_shape]['errors']['avg_mc_price']) / \
                    bench[assets][nn_shape]['errors']['avg_mc_price'] * 100
            else:
                avg_error = None
            errors.append(avg_error)

        fig.add_trace(go.Bar(
            x=asset_counts,
            y=errors,
            name=f"Network Size {nn_shape}",
            marker_color=blue_palette[i % len(blue_palette)]
        ))

    fig.update_layout(
        title="Relative Error Across Assets",
        xaxis_title="Assets",
        yaxis_title="Relative Error",
        barmode='group',
        template="plotly_white",
        width=fig_size[0],
        height=fig_size[1],
        legend=dict(
            orientation="h",
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=1
        )
    )

    # Append a "%" sign to the y-axis tick labels.
    fig.update_yaxes(ticksuffix="%")

    if save_path:
        fig.write_image(save_path)
    fig.show()


def plot_relative_l2_errors(bench, save_path=None, fig_size=(800, 600)):
    '''
    Bar plot where:
      - x-axis: Assets
      - y-axis: average relative error as a percentage
      - Each color (trace) corresponds to a network size.
    '''
    # Sort the asset counts to have a consistent x-axis order.
    asset_counts = sorted(bench.keys(), key=lambda x: int(x))
    # Gather all network sizes that appear in bench.
    all_nn_shapes = set()
    for shapes_dict in bench.values():
        all_nn_shapes.update(shapes_dict.keys())
    all_nn_shapes = sorted(all_nn_shapes)

    fig = go.Figure()

    # For each network size, collect the average error for each asset count.
    idx = np.random.choice(range(200), 100, replace=False)
    for i, nn_shape in enumerate(all_nn_shapes):
        errors = []
        for assets in asset_counts:
            if nn_shape in bench[assets]:
                nn_prices = np.array(bench[assets][nn_shape]['errors']['nn_prices'])[idx].reshape(
                    10, 10)
                mc_prices = np.array(bench[assets][nn_shape]['errors']['mc_prices'])[idx].reshape(
                    10, 10)
                avg_error = np.linalg.norm(
                    nn_prices - mc_prices, 2) / np.linalg.norm(mc_prices, 2) * 100
            else:
                avg_error = None
            errors.append(avg_error)

        fig.add_trace(go.Bar(
            x=asset_counts,
            y=errors,
            name=f"Network Size {nn_shape}",
            marker_color=blue_palette[i % len(blue_palette)]
        ))

    fig.update_layout(
        title="Relative Error Across Assets",
        xaxis_title="Assets",
        yaxis_title="Relative Error",
        barmode='group',
        template="plotly_white",
        width=fig_size[0],
        height=fig_size[1],
        legend=dict(
            orientation="h",
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=1
        )
    )

    # Append a "%" sign to the y-axis tick labels.
    fig.update_yaxes(ticksuffix="%")

    if save_path:
        fig.write_image(save_path)
    fig.show()


def plot_total_training_time(bench, save_path=None, fig_size=(800, 600)):
    # Sort the asset counts to have a consistent x-axis order.
    asset_counts = sorted(bench.keys(), key=lambda x: int(x))
    # Gather all network sizes that appear in bench.
    all_nn_shapes = set()
    for shapes_dict in bench.values():
        all_nn_shapes.update(shapes_dict.keys())
    all_nn_shapes = sorted(all_nn_shapes)

    fig = go.Figure()

    # For each network size, collect the total training time for each asset count.
    for i, nn_shape in enumerate(all_nn_shapes):
        training_times = []
        for assets in asset_counts:
            if nn_shape in bench[assets]:
                training_time = (
                    bench[assets][nn_shape]['adam_time'] + bench[assets][nn_shape]['lbfgs_time']) / 60
            else:
                training_time = None
            training_times.append(training_time)

        fig.add_trace(go.Bar(
            x=asset_counts,
            y=training_times,
            name=f"Network Size {nn_shape}",
            marker_color=blue_palette[i % len(blue_palette)]
        ))

    fig.update_layout(
        title="Total Training Time Across Assets",
        xaxis_title="Assets",
        yaxis_title="Training Time (min)",
        barmode='group',
        template="plotly_white",
        width=fig_size[0],
        height=fig_size[1],
        legend=dict(
            orientation="h",
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=1
        )
    )

    if save_path:
        fig.write_image(save_path)
    fig.show()


def generate_table(bench, save_path=None):
    table = []
    for assets in bench:
        for nn_shape in bench[assets]:
            row = {
                'n_assets': assets,
                'network_size': nn_shape,
                'lbfgs_loss': bench[assets][nn_shape]['lbfgs_loss']['total_loss'][-1],
                'lbfgs_steps': len(bench[assets][nn_shape]['lbfgs_loss']['total_loss']),
                'lbfgs_time': bench[assets][nn_shape]['lbfgs_time'],
                'adam_loss': bench[assets][nn_shape].get('adam_loss', {}).get('total_loss', [None])[-1],
                'adam_steps': len(bench[assets][nn_shape].get('adam_loss', {}).get('total_loss', [])),
                'adam_time': bench[assets][nn_shape].get('adam_time', None),
                'max_error': bench[assets][nn_shape]['errors']['max_error'],
                'avg_error': bench[assets][nn_shape]['errors']['avg_error'],
                'avg_nn_time': bench[assets][nn_shape]['errors']['avg_nn_time'],
                'avg_mc_time': bench[assets][nn_shape]['errors']['avg_mc_time'],
            }
            table.append(row)
    df = pd.DataFrame(table)
    df = df.sort_values(by=['n_assets', 'network_size'])
    df = df.reset_index(drop=True)
    df.to_csv(save_path, index=False)
    return df


def plot_evaluation_times(bench, save_path=None, fig_size=None, scale_nn=1.0, use_log=True):
    """
    Bar plot showing evaluation times per price for Monte Carlo and the NN.

    - x-axis: number of assets.
    - Two bars per asset count: one for Monte Carlo (using 'avg_mc_time') and one for NN (using 'avg_nn_time').

    Options:
      - use_log: If True, uses a logarithmic scale on the y-axis.
      - scale_nn: Multiply the NN evaluation times by this factor (useful if NN values are very small).

    Assumes each errors dict has keys:
        - 'avg_mc_time': average evaluation time per price for Monte Carlo.
        - 'avg_nn_time': average evaluation time per price for the NN.
    """
    # Set default figure size if not provided.
    if fig_size is None:
        fig_size = (1280, 720)

    # Sort the asset counts to have a consistent x-axis order.
    asset_counts = sorted(bench.keys(), key=lambda x: int(x))

    # For each asset count, average the evaluation times over all configurations.
    avg_eval_mc = []
    avg_eval_nn = []
    for assets in asset_counts:
        mc_times = []
        nn_times = []
        for nn_shape, data in bench[assets].items():
            errors = data.get('errors', {})
            if 'avg_mc_time' in errors:
                mc_times.append(errors['avg_mc_time'])
            if 'avg_nn_time' in errors:
                nn_times.append(errors['avg_nn_time'])
        avg_mc = np.mean(mc_times) if mc_times else None
        avg_nn = np.mean(nn_times) if nn_times else None
        avg_eval_mc.append(avg_mc)
        avg_eval_nn.append(avg_nn)

    # Apply scaling to NN times if requested.
    scaled_nn_times = [
        val * scale_nn if val is not None else None for val in avg_eval_nn]

    # Create the grouped bar plot.
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=asset_counts,
        y=avg_eval_mc,
        name="Monte Carlo Evaluation Time",
        marker_color=blue_palette[2]
    ))

    fig.add_trace(go.Bar(
        x=asset_counts,
        y=scaled_nn_times,
        name=f"NN Evaluation Time{' (scaled x' + str(scale_nn) + ')' if scale_nn != 1.0 else ''}",
        marker_color=blue_palette[4]
    ))

    layout_updates = {
        "title": "Evaluation Times per Price: Monte Carlo vs NN",
        "xaxis_title": "Number of Assets",
        "yaxis_title": "Evaluation Time per Price (sec)",
        "barmode": 'group',
        "template": "plotly_white",
        "width": fig_size[0],
        "height": fig_size[1],
        "legend": {
            "orientation": "h",
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
            "y": 1
        },
    }
    if use_log:
        layout_updates["yaxis_type"] = "log"

    fig.update_layout(**layout_updates)

    if save_path:
        fig.write_image(save_path)
    fig.show()


def generate_beamer_latex_tables(df, output_file):
    """
    Processes the DataFrame 'df' and generates a LaTeX file for a beamer presentation.
    One frame is created for each unique network configuration.

    The table in each frame has the columns:
      - N° Assets
      - Total Training Time (min) : computed as (lbfgs_time + adam_time)/60
      - Avg. Error (%)
      - Avg. Eval. Time (NN) (s)
      - Avg. Eval. Time (MC) (s)

    The header is manually created so that the last two columns are under a multi–column header
    "Evaluation time (s)" with sub–headers "NN" and "MC". All numerical values are shown with exactly
    4 decimals and include the appropriate unit (min, s or %).

    Parameters:
      df         : DataFrame containing the following columns:
                   'n_assets', 'network_size', 'lbfgs_time', 'adam_time',
                   'avg_error', 'avg_nn_time', 'avg_mc_time'
      output_file: Path to the output LaTeX file.
    """
    # Ensure that 'n_assets' is numeric so sorting works as intended.
    df['n_assets'] = pd.to_numeric(df['n_assets'], errors='coerce')

    # Get unique network configurations.
    nn_sizes = df['network_size'].unique()

    # Beamer document preamble and footer.
    header = r"""\documentclass{beamer}
\usetheme{default}
\usepackage{booktabs}
\usepackage{siunitx}
\begin{document}
"""
    footer = r"\end{document}"

    latex_frames = ""

    # Process each network configuration.
    for net_size in nn_sizes:
        # Filter for the current configuration.
        sub_df = df[df['network_size'] == net_size].copy()

        # Keep only the needed columns.
        cols = ['n_assets', 'lbfgs_time', 'adam_time',
                'avg_error', 'avg_nn_time', 'avg_mc_time']
        sub_df = sub_df[cols]

        # Compute total training time (in minutes).
        sub_df['total_training_time'] = (
            sub_df['lbfgs_time'] + sub_df['adam_time']) / 60

        # Select columns for display and sort by n_assets numerically.
        display_cols = ['n_assets', 'total_training_time',
                        'avg_error', 'avg_nn_time', 'avg_mc_time']
        sub_df = sub_df[display_cols].sort_values(by='n_assets')

        # Create a formatted DataFrame where each numeric column is represented as a string
        # with exactly 4 decimals and proper unit/appended symbol.
        fmt_df = pd.DataFrame()
        fmt_df['N° Assets'] = sub_df['n_assets'].apply(lambda x: f"{int(x)}")
        fmt_df['Total Training Time'] = sub_df['total_training_time'].apply(
            lambda x: f"{x:.4f} min")
        fmt_df['Avg. Error'] = sub_df['avg_error'].apply(
            lambda x: f"{x:.4f}\\%")
        fmt_df['Avg. Eval. Time (NN)'] = sub_df['avg_nn_time'].apply(
            lambda x: f"{x:.4f} s")
        fmt_df['Avg. Eval. Time (MC)'] = sub_df['avg_mc_time'].apply(
            lambda x: f"{x:.4f} s")

        # Generate table rows by joining the formatted columns with " & " and ending with " \\"
        table_rows = ""
        for idx, row in fmt_df.iterrows():
            # Each row becomes a LaTeX table row.
            row_str = " & ".join(row.values) + r" \\"
            table_rows += row_str + "\n"

        # Create a custom table header with a multi–column header for Evaluation time.
        custom_header = r"""\setlength{\tabcolsep}{2pt} % reduce horizontal spacing if needed
\resizebox{\linewidth}{!}{%
\begin{tabular}{ccccc}
\toprule
\textbf{N° Assets} & \textbf{Total Training Time (min)} & \textbf{Avg. Error (\%)} & \multicolumn{2}{c}{\textbf{Evaluation time (s)}} \\
 &  &  & \textbf{NN} & \textbf{MC} \\
\midrule
"""
        custom_footer = r"""\bottomrule
\end{tabular}
}"""

        # Combine header, rows, and footer.
        table_str = custom_header + table_rows + custom_footer

        # Wrap the table in a beamer frame.
        frame_str = "\n".join([
            r"\begin{frame}{Results: " + net_size + r"}",
            r"\begin{center}",
            table_str,
            r"\end{center}",
            r"\end{frame}"
        ])

        latex_frames += frame_str + "\n\n"

    # Compose the full document.
    full_tex = header + "\n" + latex_frames + "\n" + footer

    # Write to file.
    with open(output_file, 'w') as f:
        f.write(full_tex)

    print(f"LaTeX beamer file successfully written to {output_file}")
