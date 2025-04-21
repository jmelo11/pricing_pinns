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
    X = torch.tensor(pts, dtype=torch.float32, device=device)
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

        total_time = v["adam_time"] + v["ssbroyden_time"]
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
