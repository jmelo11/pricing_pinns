from scipy.stats import norm
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.ticker import LogFormatterMathtext
import torch

blue_palette = ["#08306B", "#08519C",
                "#3182BD", "#6BAED6", "#9ECAE1", "#C6DBEF"]


def _moving_average(arr, window):
    if window <= 1:
        return np.asarray(arr)
    c = np.cumsum(np.insert(arr, 0, 0))
    m = (c[window:] - c[:-window]) / float(window)
    return np.concatenate([np.full(window-1, m[0]), m])


# -----------------------------------------------------------------------------
#  Loss history
# -----------------------------------------------------------------------------
def plot_loss(
    loss_history,
    save_path: str = None,
    backend: str = "plotly",
    fig_size=(800, 600),
    smooth: bool = True,
    smooth_window: int = 50,
    yscale: str = "power10",
):
    """
    Parameters
    ----------
    loss_history : dict
      keys 'interior_loss','boundary_loss','initial_loss'
    save_path : str or None
    backend : {'plotly','matplotlib'}
    fig_size : (width_px,height_px)
    smooth : bool
    smooth_window : int
    yscale : {'linear','log','power10'}
      Determines how the y–axis is drawn.
    """
    i = np.asarray(loss_history["interior_loss"])
    b = np.asarray(loss_history["boundary_loss"])
    n = np.asarray(loss_history["initial_loss"])
    if smooth:
        i = _moving_average(i, smooth_window)
        b = _moving_average(b, smooth_window)
        n = _moving_average(n, smooth_window)
    x = np.arange(len(i))

    if backend.lower() == "plotly":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=i, mode="lines",
                                 name="PDE Loss", line=dict(color=blue_palette[0])))
        fig.add_trace(go.Scatter(x=x, y=b, mode="lines",
                                 name="Boundary Loss", line=dict(color=blue_palette[2])))
        fig.add_trace(go.Scatter(x=x, y=n, mode="lines",
                                 name="Initial Loss", line=dict(color=blue_palette[4])))

        # choose axis type
        if yscale == "log":
            ytype = "log"
            efmt = None
            sexp = None
        elif yscale == "power10":
            ytype = "log"
            efmt = "power"
            sexp = "all"
        else:
            ytype = "linear"
            efmt = None
            sexp = None

        fig.update_layout(
            title="Loss History",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis_type=ytype,
            width=fig_size[0], height=fig_size[1],
            template="plotly_white",
            legend=dict(orientation="h", x=0.5, y=1, xanchor="center"),
        )
        if efmt:
            fig.update_yaxes(exponentformat=efmt, showexponent=sexp)

        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()

    else:
        fig, ax = plt.subplots(
            figsize=(fig_size[0]/100, fig_size[1]/100))
        ax.plot(x, i, label="PDE Loss",     color=blue_palette[0])
        ax.plot(x, b, label="Boundary Loss", color=blue_palette[2])
        ax.plot(x, n, label="Initial Loss",  color=blue_palette[4])

        # choose axis scale
        if yscale == "log":
            ax.set_yscale("log")
        elif yscale == "power10":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
        # else linear: do nothing

        ax.set_title("Loss History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()


# -----------------------------------------------------------------------------
#  Loss comparison across runs (3 subplots)
# -----------------------------------------------------------------------------
def compare_loss_histories(
    runs,
    labels=None,
    save_path: str = None,
    backend: str = "plotly",
    fig_size=(900, 600),
    smooth: bool = True,
    smooth_window: int = 50,
    yscale: str = "power10",
):
    """
    Parameters
    ----------
    runs : list of dicts with keys 'interior_loss','boundary_loss','initial_loss'
    labels : list of str
    rest as above
    """
    n = len(runs)
    assert n > 0
    if labels is None:
        labels = [f"Run {i+1}" for i in range(n)]
    assert len(labels) == n

    def prep(d):
        i = np.asarray(d["interior_loss"])
        b = np.asarray(d["boundary_loss"])
        u = np.asarray(d["initial_loss"])
        t = i + b + u
        if smooth:
            i = _moving_average(i, smooth_window)
            b = _moving_average(b, smooth_window)
            u = _moving_average(u, smooth_window)
            t = _moving_average(t, smooth_window)
        return i, b, u, t

    data = [prep(r) for r in runs]
    x = np.arange(len(data[0][0]))
    dashes = ["solid", "dash", "dot", "dashdot", "longdash"]
    colours = [blue_palette[0], blue_palette[1],
               blue_palette[2], blue_palette[3]]

    if backend.lower() == "plotly":
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=("Interior Loss", "Boundary Loss",
                                            "Initial Loss", "Total Loss"))
        for idx, (i, b, u, t) in enumerate(data):
            dash = dashes[idx % len(dashes)]
            name = labels[idx]
            for row, arr, col in zip(range(1, 5), (i, b, u, t), colours):
                fig.add_trace(go.Scatter(x=x, y=arr, mode="lines",
                                         name=name if row == 1 else None,
                                         legendgroup=name,
                                         showlegend=(row == 1),
                                         line=dict(color=col, dash=dash)
                                         ), row=row, col=1)

        # scaling
        for row in (1, 2, 3, 4):
            if yscale == "linear":
                fig.update_yaxes(type="linear", row=row, col=1)
            else:
                fig.update_yaxes(type="log", row=row, col=1)
                if yscale == "power10":
                    fig.update_yaxes(exponentformat="power",
                                     showexponent="all",
                                     row=row, col=1)

        fig.update_layout(
            title="Loss Comparison",
            height=fig_size[1], width=fig_size[0],
            legend_title="Run"
        )
        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()

    else:
        fig, axes = plt.subplots(3, 1,
                                 figsize=(fig_size[0]/100, fig_size[1]/100),
                                 sharex=True)
        titles = ["Interior Loss", "Boundary Loss", "Initial Loss"]
        cols = [blue_palette[0], blue_palette[1], blue_palette[2]]
        for idx, (i, b, u, _) in enumerate(data):
            dash = ['-', '--', ':', '-.'][idx % 4]
            axes[0].plot(x, i, label=labels[idx],
                         color=cols[0], linestyle=dash)
            axes[1].plot(x, b, label=labels[idx],
                         color=cols[1], linestyle=dash)
            axes[2].plot(x, u, label=labels[idx],
                         color=cols[2], linestyle=dash)

        for ax in axes:
            if yscale == "log":
                ax.set_yscale("log")
            elif yscale == "power10":
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(LogFormatterMathtext())

        axes[0].set_title(titles[0])
        axes[1].set_title(titles[1])
        axes[2].set_title(titles[2])
        axes[2].set_xlabel("Epoch")
        axes[0].legend(loc="upper right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()


# -----------------------------------------------------------------------------
#  All‐config total losses
# -----------------------------------------------------------------------------
def plot_all_total_losses(
    bench,
    save_path: str = None,
    backend: str = "plotly",
    fig_size=(800, 600),
    yscale: str = "power10",
):
    """
    Similar to above, but only one subplot.
    """
    if backend.lower() == "plotly":
        fig = go.Figure()
        for idx, (assets, sd) in enumerate(bench.items()):
            for jdx, (shape, data) in enumerate(sd.items()):
                col = blue_palette[(idx+jdx) % len(blue_palette)]
                if 'lbfgs_loss' in data:
                    y = data['lbfgs_loss']['total_loss']
                    fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y,
                                             mode='lines',
                                             name=f"LBFGS ({assets},{shape})",
                                             line=dict(color=col)))
                if 'adam_loss' in data:
                    y = data['adam_loss']['total_loss']
                    fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y,
                                             mode='lines',
                                             name=f"ADAM ({assets},{shape})",
                                             line=dict(color=col, dash='dash')))
        # scaling
        if yscale == "linear":
            fig.update_yaxes(type="linear")
        else:
            fig.update_yaxes(type="log")
            if yscale == "power10":
                fig.update_yaxes(exponentformat="power", showexponent="all")

        fig.update_layout(
            title="Total Loss - All Configs",
            xaxis_title="Epoch",
            yaxis_title="Total Loss",
            width=fig_size[0], height=fig_size[1],
            template="plotly_white",
            legend=dict(orientation="h", x=0.5, y=1, xanchor="center"),
        )
        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()

    else:
        plt.figure(figsize=(fig_size[0]/100, fig_size[1]/100))
        for assets, sd in bench.items():
            for shape, data in sd.items():
                col = blue_palette[(int(assets)+int(shape)) %
                                   len(blue_palette)]
                if 'lbfgs_loss' in data:
                    y = data['lbfgs_loss']['total_loss']
                    plt.plot(y, label=f"LBFGS ({assets},{shape})", color=col)
                if 'adam_loss' in data:
                    y = data['adam_loss']['total_loss']
                    plt.plot(
                        y, '--', label=f"ADAM ({assets},{shape})", color=col)

        if yscale == "log":
            plt.yscale("log")
        elif yscale == "power10":
            ax = plt.gca()
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(LogFormatterMathtext())

        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")
        plt.title("Total Loss - All Configs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# -----------------------------------------------------------------------------
#  Max‐error continuation
# -----------------------------------------------------------------------------
def plot_max_errors_continued(
    bench,
    assets,
    save_path: str = None,
    backend: str = "plotly",
    fig_size=(800, 600),
    yscale: str = "power10",
):
    """
    One subplot, continuous curves.
    """
    if backend.lower() == "plotly":
        fig = go.Figure()
        for idx, (shape, cfg) in enumerate(bench[assets].items()):
            col = blue_palette[idx % len(blue_palette)]
            adam = cfg.get('adam_loss', {}).get('max_error', [])
            lbfgs = cfg.get('lbfgs_loss', {}).get('max_error', [])
            n0 = len(adam)
            if adam:
                fig.add_trace(go.Scatter(x=np.arange(n0), y=adam,
                                         mode='lines', name=f"{shape} (Adam)",
                                         line=dict(color=col, dash='dash')))
            if lbfgs:
                fig.add_trace(go.Scatter(x=np.arange(len(lbfgs))+n0, y=lbfgs,
                                         mode='lines', name=f"{shape} (LBFGS)",
                                         line=dict(color=col, dash='solid')))
        # scaling
        if yscale == "linear":
            fig.update_yaxes(type="linear")
        else:
            fig.update_yaxes(type="log")
            if yscale == "power10":
                fig.update_yaxes(exponentformat="power", showexponent="all")
        fig.update_layout(
            title=f"Max Error (Adam→LBFGS) Assets={assets}",
            xaxis_title="Iteration",
            yaxis_title="Max Error",
            width=fig_size[0], height=fig_size[1],
            legend=dict(orientation="h", x=0.5, y=1, xanchor="center"),
            template="plotly_white"
        )
        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()

    else:
        plt.figure(figsize=(fig_size[0]/100, fig_size[1]/100))
        for idx, (shape, cfg) in enumerate(bench[assets].items()):
            col = blue_palette[idx % len(blue_palette)]
            adam = np.array(cfg.get('adam_loss', {}).get('max_error', []))
            lbfgs = np.array(cfg.get('lbfgs_loss', {}).get('max_error', []))
            n0 = len(adam)
            if adam.size:
                plt.plot(np.arange(n0), adam, '--', color=col,
                         label=f"{shape} (Adam)")
            if lbfgs.size:
                plt.plot(np.arange(len(lbfgs))+n0, lbfgs, '-', color=col,
                         label=f"{shape} (LBFGS)")

        if yscale == "log":
            plt.yscale("log")
        elif yscale == "power10":
            ax = plt.gca()
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(LogFormatterMathtext())

        plt.xlabel("Iteration")
        plt.ylabel("Max Error")
        plt.title(f"Max Error (Adam→LBFGS) Assets={assets}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()


# -----------------------------------------------------------------------------
#  Eval‐time comparison
# -----------------------------------------------------------------------------
def plot_evaluation_times(
    bench,
    save_path: str = None,
    backend: str = "plotly",
    fig_size=(1280, 720),
    scale_nn: float = 1.0,
    use_log: bool = True,
    yscale: str = "power10",
):
    """
    Bar chart; supports linear / log / power10 y–axis.
    """
    # --- compute means ---
    def keys(d):
        # keys are net_assets
        # example: 10x1_1
        # we want to extract the last part
        # example: 1
        # and convert to int
        k = [x.split("_")[0] for x in d.keys()]
        a = [int(x.split("_")[-1]) for x in d.keys()]
        return k, a

    k, a = keys(bench)
    assets = sorted(set(a))
    # sort by number of assets
    assets.sort(key=lambda x: (x, k[a.index(x)]))

    mc, nn = [], []
    for a in assets:
        mt, nt = [], []
        for cfg in bench[a].values():
            e = cfg.get('errors', {})
            if 'avg_mc_time' in e:
                mt.append(e['avg_mc_time'])
            if 'avg_nn_time' in e:
                nt.append(e['avg_nn_time'])
        mc.append(np.mean(mt) if mt else 0)
        nn.append(np.mean(nt)*scale_nn if nt else 0)

    # --- decide final scale ---
    # if use_log=False, we force a linear axis no matter what yscale says
    if not use_log:
        scale = "linear"
    else:
        scale = yscale.lower()
    if scale not in ("linear", "log", "power10"):
        raise ValueError(f"Unknown yscale: {yscale!r}")

    # --- PLOTLY PATH ---
    if backend.lower() == "plotly":
        fig = go.Figure([
            go.Bar(x=assets, y=mc, name="MC",    marker_color=blue_palette[2]),
            go.Bar(x=assets, y=nn,
                   name=f"NN×{scale_nn}", marker_color=blue_palette[4])
        ])
        # set axis type
        fig.update_yaxes(type="linear" if scale == "linear" else "log")
        # power10 = log axis with exponent formatting
        if scale == "power10":
            fig.update_yaxes(exponentformat="power", showexponent="all")

        fig.update_layout(
            title="Eval Time per Price",
            xaxis_title="Assets",
            yaxis_title="Time (s)",
            barmode='group',
            width=fig_size[0],
            height=fig_size[1],
            template="plotly_white",
            legend=dict(orientation="h", x=0.5, y=1, xanchor="center")
        )
        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()
        return

    # --- MATPLOTLIB PATH ---
    ind = np.arange(len(assets))
    width = 0.35
    plt.figure(figsize=(fig_size[0]/100, fig_size[1]/100))
    plt.bar(ind - width/2, mc, width, label="MC", color=blue_palette[2])
    plt.bar(ind + width/2, nn, width,
            label="NN×{:.2f}".format(scale_nn), color=blue_palette[4])

    ax = plt.gca()
    if scale in ("log", "power10"):
        ax.set_yscale("log")
        if scale == "power10":
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
    else:
        ax.set_yscale("linear")

    plt.xticks(ind, assets)
    plt.xlabel("Assets")
    plt.ylabel("Time (s)")
    plt.title("Eval Time per Price")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_solution_surface(
    model,
    params,
    assets: int,
    device,
    dtype,
    save_path: str = None,
    backend: str = "plotly",
    fig_size=(800, 600),
    num_points: int = 100,
):
    """
    Plot u(x₀,x₁,τ) surface.
    """
    x0 = np.linspace(params.x_min, params.x_max, num_points)
    x1 = np.linspace(params.x_min, params.x_max, num_points)
    mid = 0.5*(params.x_min + params.x_max)
    pts = []
    for i in range(num_points):
        for j in range(num_points):
            p = [mid]*assets
            p[0], p[1] = x0[i], x1[j]
            p.append(params.tau)
            pts.append(p)
    X = torch.tensor(pts, dtype=dtype, device=device)
    with torch.no_grad():
        Y = model(X).cpu().numpy().reshape(num_points, num_points)

    if backend.lower() in ("matplotlib", "both"):
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(fig_size[0]/100, fig_size[1]/100))
        ax = fig.add_subplot(111, projection='3d')
        X0, X1 = np.meshgrid(x0, x1)
        surf = ax.plot_surface(X0, X1, Y,
                               cmap='viridis', edgecolor='none')
        ax.set_title(r"$u(x_0,x_1,\tau)$")
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$")
        ax.set_zlabel(r"$u$")
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.tight_layout()
        if save_path and backend.lower() == "matplotlib":
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    if backend.lower() in ("plotly", "both"):
        fig2 = go.Figure(data=[go.Surface(z=Y, x=x0, y=x1)])
        fig2.update_layout(
            title=r"$u(x_0,x_1,\tau)$",
            scene=dict(
                xaxis_title=r"$x_0$",
                yaxis_title=r"$x_1$",
                zaxis_title=r"$u$"
            ),
            width=fig_size[0], height=fig_size[1],
            template="plotly_white"
        )
        if save_path and backend.lower() == "plotly":
            fig2.write_image(save_path)
        else:
            fig2.show()


# your blue palette
blue_palette = ["#08306B", "#08519C",
                "#3182BD", "#6BAED6", "#9ECAE1", "#C6DBEF"]


def _parse_bench(bench, scale_nn):
    """
    Turn your bench dict into a DataFrame with columns
    ['shape','n_assets','total_time','l2_error'].
    """
    rows = []
    for key, v in bench.items():
        # expect key like "64x3_5" → shape="64x3", n_assets=5
        m = re.match(r"(.+?)_(\d+)$", key)
        if not m:
            continue
        shape, n = m.groups()
        n = int(n)

        total_time = v["adam_time"] + v["qn_time"]
        if shape.lower() == "nn":  # if you need to scale one shape
            total_time *= scale_nn

        # assume state["l2_rel_err"] is a list/array
        l2 = v["state"]["l2_rel_err"][-1]

        rows.append({
            "shape": shape,
            "n_assets": n,
            "total_time": total_time,
            "l2_error": l2,
        })
    df = pd.DataFrame(rows)
    return df


def plot_evaluation_times(
    bench,
    save_path: str = None,
    backend: str = "plotly",
    fig_size=(1280, 720),
    scale_nn: float = 1.0,
    use_log: bool = True,
    yscale: str = "power10",
):
    """
    Bar chart of total training time vs. number of assets,
    with one bar per NN shape, colored from blue_palette.
    """
    df = _parse_bench(bench, scale_nn)
    pivot = df.pivot(index="n_assets", columns="shape",
                     values="total_time").sort_index()
    shapes = list(pivot.columns)
    colors = [blue_palette[i % len(blue_palette)] for i in range(len(shapes))]

    if backend.lower().startswith("plotly"):
        import plotly.graph_objects as go
        fig = go.Figure()
        for idx, shape in enumerate(shapes):
            fig.add_trace(
                go.Bar(
                    x=pivot.index,
                    y=pivot[shape],
                    name=shape,
                    marker_color=colors[idx]
                )
            )
        fig.update_layout(
            barmode="group",
            width=fig_size[0],
            height=fig_size[1],
            xaxis_title="Number of assets",
            yaxis_title="Total training time (s)",
            template="plotly_white",
            legend=dict(orientation="h", x=0.5, y=1.05, xanchor="center")
        )
        if use_log:
            fig.update_yaxes(type="log",
                             tickformat="e" if yscale == "power10" else None)

        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()
        return fig

    else:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        fig, ax = plt.subplots(
            figsize=(fig_size[0]/100, fig_size[1]/100), dpi=100
        )
        pivot.plot(kind="bar", ax=ax, color=colors)
        ax.set_xlabel("Number of assets")
        ax.set_ylabel("Total training time (s)")
        if use_log:
            ax.set_yscale("log")
            if yscale == "power10":
                ax.yaxis.set_major_formatter(mtick.LogFormatter(base=10))
        ax.legend(title="Shape", loc="upper center", ncol=len(shapes))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100)
            plt.close(fig)
        else:
            plt.show()
        return fig


def plot_l2_error(
    bench,
    save_path: str = None,
    backend: str = "plotly",
    fig_size=(1280, 720),
    scale_nn: float = 1.0,
    use_log: bool = False,
    yscale: str = None,
):
    """
    Bar chart of final relative L2 error vs. number of assets,
    with one bar per NN shape, colored from blue_palette.
    """
    df = _parse_bench(bench, scale_nn)
    pivot = df.pivot(index="n_assets", columns="shape",
                     values="l2_error").sort_index()
    shapes = list(pivot.columns)
    colors = [blue_palette[i % len(blue_palette)] for i in range(len(shapes))]

    if backend.lower().startswith("plotly"):
        import plotly.graph_objects as go
        fig = go.Figure()
        for idx, shape in enumerate(shapes):
            fig.add_trace(
                go.Bar(
                    x=pivot.index,
                    y=pivot[shape],
                    name=shape,
                    marker_color=colors[idx]
                )
            )
        fig.update_layout(
            barmode="group",
            width=fig_size[0],
            height=fig_size[1],
            xaxis_title="Number of assets",
            yaxis_title="Final relative L₂ error",
            template="plotly_white",
            legend=dict(orientation="h", x=0.5, y=1.05, xanchor="center")
        )
        if use_log:
            fig.update_yaxes(type="log",
                             tickformat="e" if yscale == "power10" else None)

        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()
        return fig

    else:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        fig, ax = plt.subplots(
            figsize=(fig_size[0]/100, fig_size[1]/100), dpi=100
        )
        pivot.plot(kind="bar", ax=ax, color=colors)
        ax.set_xlabel("Number of assets")
        ax.set_ylabel("Final relative L₂ error")
        if use_log:
            ax.set_yscale("log")
            if yscale == "power10":
                ax.yaxis.set_major_formatter(mtick.LogFormatter(base=10))
        ax.legend(title="Shape", loc="upper center", ncol=len(shapes))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100)
            plt.close(fig)
        else:
            plt.show()
        return fig


def compare_solution_surfaces(
    models,
    params,
    assets: int,
    device,
    labels=None,
    save_path: str = None,
    backend: str = "plotly",
    fig_size=(800, 600),
    num_points: int = 100,
    real_solution: bool = False,
    view: tuple = (30, -60),
):
    """
    Compare PINN / FOPINN solution surfaces.

    If assets==1 and real_solution==True, then
      • x-axis is S = K * exp(x)
      • z-axis is V = K * u(x,τ)
    Otherwise we stay purely in dimensionless (x,τ,u).
    """
    n = len(models)
    assert n > 0
    if labels is None:
        labels = [f"Model {i+1}" for i in range(n)]
    assert len(labels) == n

    # --- build the *dimensionless* grid (x in [x_min,x_max], τ in [0,tau]) ---
    x_vals = np.linspace(params.x_min, params.x_max-1, num_points)
    t_vals = np.linspace(0.0,            params.tau,    num_points)
    XX, TT = np.meshgrid(x_vals, t_vals)
    pts = np.stack([XX, TT], axis=2).reshape(-1, 2)
    X_t = torch.tensor(pts, dtype=torch.float32, device=device)

    # --- evaluate each model and reshape back to (num_points, num_points) ---
    Ys = []
    with torch.no_grad():
        for m in models:
            out = m(X_t)
            u = out[:, 0] if (out.ndim == 2 and out.shape[1]
                              > 1) else out.view(-1)
            Ys.append(u.cpu().numpy().reshape(num_points, num_points))

    # --- if requested, convert to *real* S–V surface ---
    if assets == 1 and real_solution:
        # horizontal axis S = K * e^x
        X0_plot = params.strike * np.exp(XX)
        # vertical axis V = K * u
        Ys = [Y * params.strike for Y in Ys]
        x_label, y_label, z_label = "S", r"$\tau$", r"$V(S,\tau)$"
    else:
        # stay in dimensionless x,τ,u
        X0_plot, TT_plot = XX, TT
        x_label, y_label, z_label = r"$x$", r"$\tau$", r"$u(x,\tau)$"
        # note: if assets>1 you’d handle that case here...

    # prepare colormaps exactly as before
    import matplotlib.cm as cm
    try:
        mpl_cmap = cm.get_cmap("mako", 256)
    except ValueError:
        try:
            import seaborn as sns
            mpl_cmap = sns.color_palette("mako", 256, as_cmap=True)
        except ImportError:
            mpl_cmap = cm.get_cmap("magma", 256)

    vals = mpl_cmap(np.linspace(0, 1, 256))
    plotly_scale = [
        [i/255.0, f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
        for i, (r, g, b, a) in enumerate(vals)
    ]

    # ---- Matplotlib ----
    if backend in ("matplotlib", "both"):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        fig = plt.figure(
            figsize=((fig_size[0]/100)*n, fig_size[1]/100)
        )
        axs = []
        for idx, (Y, lbl) in enumerate(zip(Ys, labels), start=1):
            ax = fig.add_subplot(1, n, idx, projection="3d")
            surf = ax.plot_surface(
                X0_plot, TT, Y,
                cmap=mpl_cmap, edgecolor="none"
            )
            ax.set_title(lbl)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            ax.view_init(elev=view[0], azim=view[1])
            axs.append(ax)

        # single colorbar, outside the last panel
        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(surf, cax=cax, label=z_label)

        plt.tight_layout(rect=(0, 0, 0.8, 1.0))
        if save_path and backend == "matplotlib":
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    # ---- Plotly ----
    if backend in ("plotly", "both"):
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots

        fig2 = make_subplots(
            rows=1, cols=n,
            specs=[[{"type": "surface"}]*n],
            subplot_titles=labels
        )
        for i, Y in enumerate(Ys, start=1):
            fig2.add_trace(
                go.Surface(
                    z=Y,
                    x=X0_plot[0],
                    y=TT[:, 0],
                    colorscale=plotly_scale,
                    showscale=(i == n)
                ), row=1, col=i
            )

        # unify axes + camera
        elev, azim = view
        r = 2.5
        eye = dict(
            x=r*np.cos(np.radians(azim))*np.cos(np.radians(elev)),
            y=r*np.sin(np.radians(azim))*np.cos(np.radians(elev)),
            z=r*np.sin(np.radians(elev)),
        )
        for i in range(1, n+1):
            scene = fig2.layout[f"scene{i}"]
            scene.update(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label,
                camera=dict(eye=eye)
            )

        fig2.update_layout(
            width=fig_size[0]*n,
            height=fig_size[1],
            title_text="Solution surface comparison",
            template="plotly_white"
        )
        if save_path and backend == "plotly":
            fig2.write_image(save_path)
        else:
            fig2.show()


# ─────────────────────────────  Black–Scholes helper
def black_scholes_option_price(S, K, T, r, sigma, option_type="call"):
    S = np.maximum(S.astype(np.float32), 1e-12)
    K = np.float32(K)
    T = np.maximum(T.astype(np.float32), 1e-12)
    r = np.float32(r)
    sigma = np.float32(sigma)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ─────────────────────────────  error-heat-map routine
def compare_error_heatmaps(
    models,
    params,
    device,
    labels=None,
    save_path: str | None = None,
    backend: str = "matplotlib",  # "matplotlib" | "plotly" | "both"
    fig_size=(6, 3),  # inches per panel  (width , height)
    num_S: int = 100,  # resolution in x-direction
    num_t: int = 100,  # resolution in t-direction
    cmap: str = "mako",
):
    """
    Heat-map of | V_true − V_pred | for several networks.

    Network   input : (x , t) with  x = log(S / K)
    Network  output : log(u)       with  u = V / K

    *Absolute* error is plotted (dollar units).
    """
    # ─────────  labels
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(models))]
    if len(labels) != len(models):
        raise ValueError("`labels` length must match `models` length")

    # ─────────  shorthand
    r, K, sigma = params.r, params.strike, params.sigma
    option_type = getattr(params, "option_type", "call")

    # ─────────  grid   (shape:  x → rows ,  t → columns  ⇒  (num_S , num_t))
    x_vals = np.linspace(params.x_min, params.x_max, num_S, dtype=np.float32)
    S_vals = np.exp(x_vals) * K
    t_vals = np.linspace(0.0, params.tau, num_t, dtype=np.float32)

    # (num_S, num_t)
    XX, TT = np.meshgrid(x_vals, t_vals, indexing="ij")

    # ─────────  flatten grid for one forward pass per model
    pts = torch.from_numpy(
        np.stack([XX.ravel(order="C"), TT.ravel(order="C")], axis=1)
    ).to(device)                                                # (N, 2)  N = num_S*num_t

    # ─────────  network predictions  →  dimensional price
    V_preds = []
    with torch.no_grad():
        for net in models:
            out = net(pts)                               # (N,) or (N, m)
            if out.ndim == 2:
                out = out[:, 0]                          # take only log(u)
            out = out.view(-1)                           # ensure flat
            V_pred = out * K                  # V = K · exp(log u)
            V_pred = V_pred.cpu().numpy().reshape(
                num_S, num_t, order="C"
            ).astype(np.float32)
            V_preds.append(V_pred)

    # ─────────  analytic Black–Scholes surface
    Tau = TT     # remaining time grid
    V_true = black_scholes_option_price(
        S_vals[:, None], K, Tau, r, sigma, option_type
    ).astype(np.float32)                                 # (num_S, num_t)

    # ─────────  error maps & global relative L₂ norms
    Err_maps = [np.abs(Vp - V_true) for Vp in V_preds]
    rel_L2 = [np.linalg.norm(Vp - V_true) /
              np.linalg.norm(V_true) for Vp in V_preds]

    # ─────────  colour-map
    import matplotlib.cm as cm
    try:
        mpl_cmap = cm.get_cmap(cmap, 256)
    except ValueError:                                   # seaborn fallback
        import seaborn as sns
        mpl_cmap = sns.color_palette(cmap, 256, as_cmap=True)

    # ──────────────────────────────────────────────────────────────────
    #  MATPLOTLIB
    # ──────────────────────────────────────────────────────────────────
    backend = backend.lower()
    if backend in ("matplotlib", "both"):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, axes = plt.subplots(
            1, len(models),
            figsize=(fig_size[0] * len(models), fig_size[1]),
            squeeze=False
        )
        axes = axes[0]

        for ax, err, lbl, r2 in zip(axes, Err_maps, labels, rel_L2):
            im = ax.imshow(
                err.T,                                       # transpose -> x horizontal
                extent=[params.x_min, params.x_max, 0, params.tau],
                origin="lower", aspect="auto", cmap=mpl_cmap
            )
            ax.set_title(f"{lbl}\nRelative $L_2$ Error = {r2:.2e}")
            ax.set_xlabel(r"$x=\log(S/K)$")
            ax.set_ylabel(r"$\tau$")

        # single colour-bar in its own axis right of last subplot
        divider = make_axes_locatable(axes[-1])
        cax = divider.append_axes("right", size="2.5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(r"$|{\rm Error}|$")

        fig.tight_layout()
        if save_path and backend == "matplotlib":
            fig.savefig(save_path, dpi=120)
            plt.close(fig)
        else:
            plt.show()

    # ──────────────────────────────────────────────────────────────────
    #  PLOTLY
    # ──────────────────────────────────────────────────────────────────
    if backend in ("plotly", "both"):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # convert matplotlib cmap to plotly scale
        vals = mpl_cmap(np.linspace(0, 1, 256))
        plotly_scale = [[i / 255, f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
                        for i, (r, g, b, _) in enumerate(vals)]

        fig_p = make_subplots(
            rows=1, cols=len(models),
            subplot_titles=[f"{lb}<br>rel L₂ err = {er:.2e}"
                            for lb, er in zip(labels, rel_L2)]
        )

        for i, (err, show_scale) in enumerate(
            zip(Err_maps, [False]*(len(models)-1) + [True]), start=1
        ):
            fig_p.add_trace(
                go.Heatmap(
                    z=err.T,                   # Plotly uses z[y,x]
                    x=x_vals, y=t_vals,
                    colorscale=plotly_scale,
                    showscale=show_scale,
                    colorbar=dict(title="|error|")
                ),
                row=1, col=i
            )

        fig_p.update_xaxes(title_text="x = log(S/K)")
        fig_p.update_yaxes(title_text="t")
        fig_p.update_layout(
            width=fig_size[0] * 100 * len(models),
            height=fig_size[1] * 100,
            template="plotly_white",
            title="Absolute-error heat maps"
        )

        if save_path and backend == "plotly":
            fig_p.write_image(save_path)
        else:
            fig_p.show()


def compare_error_histories(
    runs,
    labels=None,
    save_path: str = None,
    backend: str = "plotly",
    fig_size=(900, 600),
    smooth: bool = True,
    smooth_window: int = 50,
    yscale: str = "power10",
):
    """
    Plot L₂ relative-error over training for multiple runs,
    allowing runs of different lengths.

    Parameters
    ----------
    runs : list of dicts with key 'l2_rel_err'
    labels : list of str, one per run
    save_path : if given, write figure to this path
    backend : 'plotly' or 'matplotlib'
    fig_size : (width, height) pixels for Plotly, inches (dpi=100) for Matplotlib
    smooth : apply moving‐average smoothing?
    smooth_window : window size for smoothing
    yscale : 'linear', 'log', or 'power10'
    """
    n = len(runs)
    assert n > 0, "Need at least one run to compare"
    if labels is None:
        labels = [f"Run {i+1}" for i in range(n)]
    assert len(labels) == n

    # prepare data (only L2)
    histories = []
    for d in runs:
        l2 = np.asarray(d["l2_rel_err"], dtype=float)
        if smooth:
            l2 = _moving_average(l2, smooth_window)
        histories.append(l2)

    # color cycle
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    backend = backend.lower()

    if backend == "plotly":
        import plotly.graph_objs as go

        fig = go.Figure()
        for idx, l2 in enumerate(histories):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(l2)),
                    y=l2,
                    mode="lines",
                    name=labels[idx],
                    line=dict(color=colors[idx % len(colors)])
                )
            )

        # y-scale
        if yscale == "linear":
            fig.update_yaxes(type="linear")
        else:
            fig.update_yaxes(type="log")
            if yscale == "power10":
                fig.update_yaxes(
                    exponentformat="power",
                    showexponent="all"
                )

        fig.update_layout(
            title="$L_2$ Relative Error over Training",
            xaxis_title="Iteration",
            yaxis_title="rel $L_2$ error",
            width=fig_size[0], height=fig_size[1],
            legend_title="Run",
            template="plotly_white"
        )

        if save_path:
            fig.write_image(save_path)
        else:
            fig.show()

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogFormatterMathtext

        # convert px → inches for Matplotlib at dpi=100
        fig, ax = plt.subplots(
            1, 1,
            figsize=(fig_size[0]/100, fig_size[1]/100),
            squeeze=True
        )

        for idx, l2 in enumerate(histories):
            ax.plot(
                np.arange(len(l2)),
                l2,
                label=labels[idx],
                color=colors[idx % len(colors)]
            )

        # y-scale
        if yscale == "linear":
            ax.set_yscale("linear")
        elif yscale == "log":
            ax.set_yscale("log")
        elif yscale == "power10":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
        else:
            raise ValueError(f"Unknown yscale {yscale!r}")

        ax.set_title("$L_2$ Relative Error over Training")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Relative $L_2$ Error")
        ax.legend(loc="upper right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()

    else:
        raise ValueError(f"Unsupported backend: {backend!r}")
