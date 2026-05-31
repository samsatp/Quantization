"""
Vector Quantization Explorer
Visualizes Scalar Quantization (Lloyd-Max) and Product Quantization (k-means per sub-space).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Vector Quantization Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Algorithms
# ─────────────────────────────────────────────────────────────────────────────

def lloyd_max(data: np.ndarray, n_levels: int, max_iter: int = 60, tol: float = 1e-9):
    """
    Scalar Lloyd-Max quantizer for 1-D data.
    Returns (centroids, full_boundaries, history_of_centroids).
    full_boundaries has length n_levels+1 with ±inf at the ends.
    """
    lo, hi = data.min(), data.max()
    centroids = np.linspace(lo, hi, n_levels)
    history = [centroids.copy()]

    for _ in range(max_iter):
        interior = (centroids[:-1] + centroids[1:]) / 2
        labels = np.digitize(data, interior)
        new_c = np.array([
            data[labels == k].mean() if np.any(labels == k) else centroids[k]
            for k in range(n_levels)
        ])
        history.append(new_c.copy())
        if np.max(np.abs(new_c - centroids)) < tol:
            break
        centroids = new_c

    interior_final = (centroids[:-1] + centroids[1:]) / 2
    full_bounds = np.concatenate([[-np.inf], interior_final, [np.inf]])
    return centroids, full_bounds, history


def quantize_sq(vec: np.ndarray, centroids_all, bounds_all):
    """Quantize a vector using per-dimension SQ codebooks."""
    q = np.empty_like(vec)
    bins = np.empty(len(vec), dtype=int)
    for d in range(len(vec)):
        interior = bounds_all[d][1:-1]
        idx = int(np.digitize(vec[d], interior))
        q[d] = centroids_all[d][idx]
        bins[d] = idx
    return q, bins


def kmeans(data: np.ndarray, k: int, max_iter: int = 60, tol: float = 1e-9, seed: int = 0):
    """
    K-means on N×d data.
    Returns (centroids, labels, history_of_centroids).
    history[i] has shape (k, d).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), k, replace=False)
    centroids = data[idx].astype(float).copy()
    history = [centroids.copy()]

    for _ in range(max_iter):
        dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_c = np.array([
            data[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])
        history.append(new_c.copy())
        if np.max(np.abs(new_c - centroids)) < tol:
            break
        centroids = new_c

    return centroids, labels, history


def product_quantize(data: np.ndarray, n_subspaces: int, k: int, seed: int = 0):
    """
    Product quantization: split D dims into n_subspaces sub-spaces, k-means on each.
    Returns list of dicts {centroids, labels, history, dims}.
    """
    n_vec, n_dim = data.shape
    base = n_dim // n_subspaces
    extra = n_dim % n_subspaces
    sub_dims, start = [], 0
    for m in range(n_subspaces):
        end = start + base + (1 if m < extra else 0)
        sub_dims.append(list(range(start, end)))
        start = end

    results = []
    for m, dims in enumerate(sub_dims):
        sub_data = data[:, dims]
        c, l, h = kmeans(sub_data, k, seed=seed + m * 1000)
        results.append({"centroids": c, "labels": l, "history": h, "dims": dims})
    return results


def quantize_pq(vec: np.ndarray, pq_results):
    """Quantize a vector using PQ codebooks."""
    q = np.empty_like(vec)
    bin_indices = []
    for res in pq_results:
        c, dims = res["centroids"], res["dims"]
        sub_vec = vec[dims]
        idx = int(np.argmin(np.linalg.norm(c - sub_vec, axis=1)))
        q[dims] = c[idx]
        bin_indices.append(idx)
    return q, bin_indices


def labels_from_centroids(data_sub: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Recompute cluster assignments for a given centroid state."""
    dists = np.linalg.norm(data_sub[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

def make_data(n_vec: int, n_dim: int, seed: int, dist: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dist == "Gaussian":
        return rng.normal(0, 1, (n_vec, n_dim))
    if dist == "Uniform":
        return rng.uniform(-3, 3, (n_vec, n_dim))
    if dist == "Bimodal":
        m = rng.random((n_vec, n_dim)) > 0.5
        return np.where(m,
                        rng.normal(-1.8, 0.5, (n_vec, n_dim)),
                        rng.normal( 1.8, 0.5, (n_vec, n_dim)))
    # Mixed: cycle Gaussian / Uniform / Bimodal
    out = np.zeros((n_vec, n_dim))
    for d in range(n_dim):
        t = d % 3
        if t == 0:
            out[:, d] = rng.normal(0, 1, n_vec)
        elif t == 1:
            out[:, d] = rng.uniform(-2.5, 2.5, n_vec)
        else:
            m = rng.random(n_vec) > 0.5
            out[:, d] = np.where(m,
                                 rng.normal(-1.8, 0.5, n_vec),
                                 rng.normal( 1.8, 0.5, n_vec))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

for key, val in [("seed", 42), ("qseed", 777)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Controls")

    quant_type = st.radio(
        "Quantization type",
        ["Scalar Quantization (SQ)", "Product Quantization (PQ)"],
        horizontal=False,
    )
    is_pq = quant_type.startswith("Product")

    st.divider()
    st.markdown("**Dataset**")
    n_vec = st.slider("Number of vectors",  20, 300,  80, step=10)
    n_dim = st.slider("Vector dimensions",   2,  12,   6)
    dist  = st.selectbox("Distribution",
                         ["Gaussian", "Uniform", "Bimodal", "Mixed"], index=3)

    st.divider()

    if not is_pq:
        st.markdown("**SQ parameters**")
        n_lvl = st.slider("Quantization levels / dim", 2, 16, 4)
        sel_d = st.slider("Dimension for convergence demo", 0, n_dim - 1, 0)
    else:
        st.markdown("**PQ parameters**")
        max_sub = n_dim
        n_sub   = st.slider("Number of sub-spaces (M)", 1, max_sub, min(2, max_sub))
        k_sub   = st.slider("Centroids per sub-space (K)", 2, 32, 4)
        sel_sub = st.slider("Sub-space for convergence demo", 0, n_sub - 1, 0)
        st.caption(
            f"Sub-space dims: "
            + ", ".join(
                str(n_dim // n_sub + (1 if m < n_dim % n_sub else 0))
                for m in range(n_sub)
            )
        )

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("New dataset", use_container_width=True):
            st.session_state.seed = int(np.random.randint(0, 99_999))
    with col_b:
        if st.button("New query vec", use_container_width=True):
            st.session_state.qseed = int(np.random.randint(0, 99_999))
    st.caption(f"Dataset seed : {st.session_state.seed}")
    st.caption(f"Query seed   : {st.session_state.qseed}")

    st.divider()
    if not is_pq:
        st.markdown(
            "**Scalar Quantization**\n\n"
            "Each dimension is quantized independently. "
            "**Lloyd-Max** iterates:\n"
            "1. Boundaries = midpoints between centroids\n"
            "2. Centroids = conditional mean in each bin\n\n"
            "Bits per vector: D × log₂(L)"
        )
    else:
        st.markdown(
            "**Product Quantization**\n\n"
            "Dimensions are split into M sub-spaces. "
            "**K-means** finds K centroids per sub-space. "
            "A vector is encoded as M indices.\n\n"
            "Bits per vector: M × log₂(K)\n\n"
            "When M = D, PQ reduces to per-dimension k-means ≈ SQ."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Compute
# ─────────────────────────────────────────────────────────────────────────────

data = make_data(n_vec, n_dim, st.session_state.seed, dist)
dim_labels = [f"d{i}" for i in range(n_dim)]
PALETTE    = px.colors.qualitative.Set2
CLUSTER_PALETTE = px.colors.qualitative.Plotly

rng_q = np.random.default_rng(st.session_state.qseed)
qvec  = rng_q.normal(0, 1.2, n_dim)

if not is_pq:
    centroids_all, bounds_all, history_all = [], [], []
    for d in range(n_dim):
        c, b, h = lloyd_max(data[:, d], n_lvl)
        centroids_all.append(c)
        bounds_all.append(b)
        history_all.append(h)
    qvec_q, qvec_bins = quantize_sq(qvec, centroids_all, bounds_all)
else:
    pq_results = product_quantize(data, n_sub, k_sub, seed=st.session_state.seed)
    qvec_q, qvec_bins = quantize_pq(qvec, pq_results)
    sub_labels = [
        "m" + str(m) + "(" + "+".join(f"d{d}" for d in res["dims"]) + ")"
        for m, res in enumerate(pq_results)
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

st.title("Vector Quantization Explorer")
if not is_pq:
    st.markdown(
        "**Scalar Quantization** maps each dimension independently using the "
        "**Lloyd-Max algorithm**, which finds optimal bin boundaries and centroids "
        "minimising mean-squared error."
    )
else:
    st.markdown(
        "**Product Quantization** splits the vector into M sub-spaces and applies "
        "**k-means** independently in each sub-space. "
        "A vector is represented by M centroid indices — one per sub-space."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Dataset  (shared)
# ─────────────────────────────────────────────────────────────────────────────

st.header("1 · Dataset")
c1, c2 = st.columns([3, 1])

with c1:
    fig = go.Figure(go.Heatmap(
        z=data, x=dim_labels, y=[f"v{i}" for i in range(n_vec)],
        colorscale="RdBu_r", zmid=0,
        colorbar=dict(title="value", thickness=12),
    ))
    fig.update_layout(
        title=f"{n_vec} vectors  ×  {n_dim} dimensions  ({dist} distribution)",
        xaxis_title="Dimension", yaxis_title="Vector index",
        height=min(500, max(250, n_vec * 4)),
        margin=dict(l=60, r=20, t=45, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("**First 10 rows**")
    st.dataframe(
        pd.DataFrame(data[:10], columns=dim_labels).style.format("{:.3f}"),
        height=340,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Distributions  (shared)
# ─────────────────────────────────────────────────────────────────────────────

st.header("2 · Value Distributions")

fig = go.Figure()
for d in range(n_dim):
    fig.add_trace(go.Violin(
        x=[dim_labels[d]] * n_vec, y=data[:, d],
        name=dim_labels[d],
        box_visible=True, meanline_visible=True,
        fillcolor=PALETTE[d % len(PALETTE)],
        line_color="rgba(0,0,0,0.35)",
        opacity=0.75, showlegend=False,
        points="outliers", jitter=0.3, pointpos=0,
    ))
fig.update_layout(
    xaxis_title="Dimension", yaxis_title="Value",
    title="Per-dimension distribution (violin + box + mean line)",
    height=370, margin=dict(t=45, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Convergence
# ─────────────────────────────────────────────────────────────────────────────

if not is_pq:
    # ── SQ: Lloyd-Max convergence for selected dimension ──────────────────────
    st.header(f"3 · Lloyd-Max Convergence — dimension {sel_d}")

    history  = history_all[sel_d]
    dim_data = data[:, sel_d]
    n_steps  = len(history)

    if n_steps <= 7:
        show = list(range(n_steps))
    else:
        show = sorted(set([0, 1, 2, 3] + list(range(max(4, n_steps - 3), n_steps))))
    n_show = len(show)

    counts, bin_edges = np.histogram(dim_data, bins=30)
    bar_w = bin_edges[1] - bin_edges[0]
    bar_x = (bin_edges[:-1] + bin_edges[1:]) / 2
    y_max = counts.max() * 1.2
    x_lo, x_hi = dim_data.min() - 0.4, dim_data.max() + 0.4

    fig = make_subplots(
        rows=1, cols=n_show,
        subplot_titles=["Init" if s == 0 else f"Step {s}" for s in show],
        shared_yaxes=True, horizontal_spacing=0.03,
    )
    for ci, step in enumerate(show, 1):
        c = history[step]
        interior = (c[:-1] + c[1:]) / 2 if len(c) > 1 else np.array([])

        fig.add_trace(go.Bar(
            x=bar_x, y=counts, width=bar_w,
            marker_color="lightsteelblue", marker_line_color="white",
            marker_line_width=0.4, showlegend=False,
        ), row=1, col=ci)

        for bv in interior:
            fig.add_trace(go.Scatter(
                x=[bv, bv], y=[0, y_max], mode="lines",
                line=dict(color="dimgray", dash="dash", width=1), showlegend=False,
            ), row=1, col=ci)

        for cv in c:
            fig.add_trace(go.Scatter(
                x=[cv, cv], y=[0, y_max], mode="lines",
                line=dict(color="crimson", width=2.5), showlegend=False,
            ), row=1, col=ci)

    fig.update_xaxes(range=[x_lo, x_hi])
    fig.update_yaxes(range=[0, y_max])
    fig.update_layout(
        height=290,
        title=(
            f"Centroids (red) converging over {n_steps - 1} iterations  |  "
            "dashed gray = partition boundaries"
            + (f"  |  showing {n_show} of {n_steps}" if n_show < n_steps else "")
        ),
        bargap=0.04, margin=dict(t=65, b=30, l=40, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show centroid values at each displayed step"):
        st.dataframe(pd.DataFrame(
            {"Init" if s == 0 else f"Step {s}": history[s] for s in show},
            index=[f"L{k}" for k in range(n_lvl)],
        ).style.format("{:.4f}"))

else:
    # ── PQ: K-means convergence for selected sub-space ────────────────────────
    res      = pq_results[sel_sub]
    dims_sel = res["dims"]
    sub_dim  = len(dims_sel)
    hist_pq  = res["history"]
    n_steps  = len(hist_pq)

    st.header(
        f"3 · K-means Convergence — sub-space {sel_sub}  "
        f"(dims {dims_sel},  sub-dim = {sub_dim})"
    )

    if n_steps <= 7:
        show = list(range(n_steps))
    else:
        show = sorted(set([0, 1, 2, 3] + list(range(max(4, n_steps - 3), n_steps))))
    n_show = len(show)

    sub_data = data[:, dims_sel]

    if sub_dim == 1:
        # 1-D: same histogram style as SQ
        d1 = sub_data[:, 0]
        counts, bin_edges = np.histogram(d1, bins=30)
        bar_w = bin_edges[1] - bin_edges[0]
        bar_x = (bin_edges[:-1] + bin_edges[1:]) / 2
        y_max = counts.max() * 1.2
        x_lo, x_hi = d1.min() - 0.4, d1.max() + 0.4

        fig = make_subplots(
            rows=1, cols=n_show,
            subplot_titles=["Init" if s == 0 else f"Step {s}" for s in show],
            shared_yaxes=True, horizontal_spacing=0.03,
        )
        for ci, step in enumerate(show, 1):
            c1d = hist_pq[step][:, 0]
            fig.add_trace(go.Bar(
                x=bar_x, y=counts, width=bar_w,
                marker_color="lightsteelblue", marker_line_color="white",
                marker_line_width=0.4, showlegend=False,
            ), row=1, col=ci)
            for cv in c1d:
                fig.add_trace(go.Scatter(
                    x=[cv, cv], y=[0, y_max], mode="lines",
                    line=dict(color="crimson", width=2.5), showlegend=False,
                ), row=1, col=ci)

        fig.update_xaxes(range=[x_lo, x_hi])
        fig.update_yaxes(range=[0, y_max])
        fig.update_layout(
            height=290,
            title=f"K-means centroids converging ({n_steps - 1} iters)  |  red = centroids",
            bargap=0.04, margin=dict(t=65, b=30, l=40, r=10),
        )

    else:
        # 2-D+ scatter: use first 2 dims of sub-space for display
        xs, ys = sub_data[:, 0], sub_data[:, 1]
        pad = 0.3
        x_lo2, x_hi2 = xs.min() - pad, xs.max() + pad
        y_lo2, y_hi2 = ys.min() - pad, ys.max() + pad

        fig = make_subplots(
            rows=1, cols=n_show,
            subplot_titles=["Init" if s == 0 else f"Step {s}" for s in show],
            shared_yaxes=True, horizontal_spacing=0.04,
        )
        for ci, step in enumerate(show, 1):
            c_step = hist_pq[step]
            lbl    = labels_from_centroids(sub_data, c_step)

            # Data points coloured by cluster
            for j in range(k_sub):
                mask = lbl == j
                if not mask.any():
                    continue
                fig.add_trace(go.Scatter(
                    x=xs[mask], y=ys[mask], mode="markers",
                    marker=dict(
                        size=5, opacity=0.55,
                        color=CLUSTER_PALETTE[j % len(CLUSTER_PALETTE)],
                    ),
                    showlegend=False,
                ), row=1, col=ci)

            # Centroids
            fig.add_trace(go.Scatter(
                x=c_step[:, 0], y=c_step[:, 1], mode="markers",
                marker=dict(
                    size=12, symbol="x", color="crimson",
                    line=dict(width=2, color="white"),
                ),
                showlegend=False,
            ), row=1, col=ci)

        fig.update_xaxes(range=[x_lo2, x_hi2])
        fig.update_yaxes(range=[y_lo2, y_hi2])
        fig.update_layout(
            height=330,
            title=(
                f"K-means in sub-space {sel_sub} (showing dims {dims_sel[0]}, {dims_sel[1]}) — "
                "colours = cluster, × = centroid"
            ),
            margin=dict(t=65, b=30, l=40, r=10),
        )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show centroid coordinates at each displayed step"):
        for s in show:
            step_label = "Init" if s == 0 else f"Step {s}"
            c = hist_pq[s]
            df = pd.DataFrame(
                c, columns=[f"dim_{dims_sel[d]}" for d in range(sub_dim)],
                index=[f"C{j}" for j in range(k_sub)],
            )
            st.markdown(f"**{step_label}**")
            st.dataframe(df.style.format("{:.4f}"))

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Summary
# ─────────────────────────────────────────────────────────────────────────────

if not is_pq:
    # ── SQ: all-dimension violin + centroids ──────────────────────────────────
    st.header("4 · Lloyd-Max Results — All Dimensions")

    fig = go.Figure()
    for d in range(n_dim):
        fig.add_trace(go.Violin(
            x=[dim_labels[d]] * n_vec, y=data[:, d],
            side="positive", fillcolor=PALETTE[d % len(PALETTE)],
            line_color="rgba(120,120,120,0.4)", opacity=0.22,
            showlegend=False, points=False, name=dim_labels[d],
        ))
        c   = centroids_all[d]
        bnd = bounds_all[d][1:-1]
        fig.add_trace(go.Scatter(
            x=[dim_labels[d]] * len(bnd), y=bnd, mode="markers",
            marker=dict(symbol="line-ew-open", size=20, color="dimgray",
                        line=dict(width=1.8, color="dimgray")),
            name="boundary" if d == 0 else None,
            showlegend=(d == 0), legendgroup="bnd",
        ))
        fig.add_trace(go.Scatter(
            x=[dim_labels[d]] * len(c), y=c, mode="markers",
            marker=dict(symbol="line-ew-open", size=28, color="crimson",
                        line=dict(width=2.8, color="crimson")),
            name="centroid" if d == 0 else None,
            showlegend=(d == 0), legendgroup="ctr",
        ))

    fig.update_layout(
        xaxis_title="Dimension", yaxis_title="Value",
        title="Data distribution with Lloyd-Max centroids (red) and boundaries (gray)",
        height=430, margin=dict(t=45, b=40),
        legend=dict(orientation="h", y=-0.14, x=0.3),
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    # ── PQ: cluster scatter per sub-space ─────────────────────────────────────
    st.header("4 · K-means Results — All Sub-spaces")
    st.caption(
        "Colours = cluster assignment.  × = centroid.  "
        "Sub-spaces with 1 dimension are shown as 1-D strips."
    )

    ncols = min(n_sub, 4)
    nrows = int(np.ceil(n_sub / ncols))
    # Build subplot grid
    sub_titles = [f"sub {m}: {pq_results[m]['dims']}" for m in range(n_sub)]
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=sub_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    for m, res in enumerate(pq_results):
        r = m // ncols + 1
        col = m % ncols + 1
        dims_m = res["dims"]
        sub_dim_m = len(dims_m)
        labels_m  = res["labels"]
        c_m       = res["centroids"]

        if sub_dim_m == 1:
            d0 = data[:, dims_m[0]]
            for j in range(k_sub):
                mask = labels_m == j
                if not mask.any():
                    continue
                fig.add_trace(go.Scatter(
                    x=d0[mask],
                    y=np.full(mask.sum(), j * 0.05),
                    mode="markers",
                    marker=dict(size=5, opacity=0.5,
                                color=CLUSTER_PALETTE[j % len(CLUSTER_PALETTE)]),
                    showlegend=False,
                ), row=r, col=col)
            # centroid ticks
            for j, cv in enumerate(c_m[:, 0]):
                fig.add_trace(go.Scatter(
                    x=[cv, cv], y=[-0.08, 0.08], mode="lines",
                    line=dict(color="crimson", width=3),
                    showlegend=False,
                ), row=r, col=col)

        else:
            xs_m, ys_m = data[:, dims_m[0]], data[:, dims_m[1]]
            for j in range(k_sub):
                mask = labels_m == j
                if not mask.any():
                    continue
                fig.add_trace(go.Scatter(
                    x=xs_m[mask], y=ys_m[mask], mode="markers",
                    marker=dict(size=5, opacity=0.55,
                                color=CLUSTER_PALETTE[j % len(CLUSTER_PALETTE)]),
                    showlegend=False,
                ), row=r, col=col)
            fig.add_trace(go.Scatter(
                x=c_m[:, 0], y=c_m[:, 1], mode="markers",
                marker=dict(size=12, symbol="x", color="crimson",
                            line=dict(width=2, color="white")),
                showlegend=False,
            ), row=r, col=col)

    fig.update_layout(
        height=max(280, nrows * 270),
        margin=dict(t=50, b=30, l=40, r=10),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Codebook
# ─────────────────────────────────────────────────────────────────────────────

if not is_pq:
    # ── SQ codebook ───────────────────────────────────────────────────────────
    st.header("5 · Codebook")

    cb_arr = np.column_stack(centroids_all)
    cb_df  = pd.DataFrame(cb_arr, columns=dim_labels,
                          index=[f"L{k}" for k in range(n_lvl)])
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            "Each cell = centroid for that level × dimension.  \n"
            f"Codebook size: **{n_lvl} levels × {n_dim} dims** = "
            f"{n_lvl * n_dim} entries.  \n"
            f"Bits per vector: **{n_dim} × log₂({n_lvl}) = "
            f"{n_dim * int(np.ceil(np.log2(max(n_lvl, 2))))} bits**."
        )
        st.dataframe(
            cb_df.style.format("{:.4f}").background_gradient(cmap="RdBu_r", axis=None),
            height=300,
        )
    with c2:
        fig = go.Figure(go.Heatmap(
            z=cb_arr, x=dim_labels, y=cb_df.index.tolist(),
            colorscale="RdBu_r", zmid=0,
            text=np.round(cb_arr, 3), texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="centroid", thickness=12),
        ))
        fig.update_layout(
            title="Codebook heatmap  (row = level, col = dimension)",
            xaxis_title="Dimension", yaxis_title="Quantization level",
            height=300, margin=dict(t=45, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    # ── PQ codebook ───────────────────────────────────────────────────────────
    st.header("5 · Codebook")

    bits_pq = n_sub * int(np.ceil(np.log2(max(k_sub, 2))))
    st.markdown(
        f"PQ encodes each vector as **{n_sub} indices** (one per sub-space).  \n"
        f"Each index selects from **{k_sub} centroids**.  \n"
        f"Bits per vector: **{n_sub} × log₂({k_sub}) = {bits_pq} bits**."
    )

    # Show each sub-space codebook as a small heatmap in a grid
    ncols_cb = min(n_sub, 4)
    nrows_cb = int(np.ceil(n_sub / ncols_cb))

    for row_i in range(nrows_cb):
        cols = st.columns(ncols_cb)
        for col_i in range(ncols_cb):
            m = row_i * ncols_cb + col_i
            if m >= n_sub:
                break
            res_m   = pq_results[m]
            c_m     = res_m["centroids"]      # (k_sub, sub_dim_m)
            dims_m  = res_m["dims"]
            sub_dim_m = len(dims_m)
            col_names = [f"d{d}" for d in dims_m]

            cb_m_df = pd.DataFrame(c_m, columns=col_names,
                                   index=[f"C{j}" for j in range(k_sub)])
            with cols[col_i]:
                st.markdown(f"**Sub-space {m}** — dims {dims_m}")
                if sub_dim_m == 1:
                    st.dataframe(
                        cb_m_df.style.format("{:.4f}")
                                     .background_gradient(cmap="RdBu_r", axis=None),
                        height=min(350, 55 + 35 * k_sub),
                    )
                else:
                    fig = go.Figure(go.Heatmap(
                        z=c_m, x=col_names,
                        y=[f"C{j}" for j in range(k_sub)],
                        colorscale="RdBu_r", zmid=0,
                        text=np.round(c_m, 3), texttemplate="%{text}",
                        textfont=dict(size=9),
                        colorbar=dict(thickness=8),
                        showscale=False,
                    ))
                    fig.update_layout(
                        height=min(350, 60 + 30 * k_sub),
                        margin=dict(t=10, b=10, l=5, r=5),
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Quantize a new vector
# ─────────────────────────────────────────────────────────────────────────────

st.header("6 · Quantize a New Vector")

if not is_pq:
    st.markdown(
        "Each dimension is looked up independently in the SQ codebook and "
        "replaced with its bin's centroid."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Original vector**")
        st.dataframe(
            pd.DataFrame({"Dim": dim_labels, "Value": qvec})
              .style.format({"Value": "{:.4f}"}),
            hide_index=True, height=min(400, 50 + 35 * n_dim),
        )
    with c2:
        st.markdown("**Quantized vector + error**")
        mse = np.mean((qvec_q - qvec) ** 2)
        st.dataframe(
            pd.DataFrame({
                "Dim":       dim_labels,
                "Bin":       [f"L{qvec_bins[d]}" for d in range(n_dim)],
                "Quantized": qvec_q,
                "|Error|":   np.abs(qvec_q - qvec),
            }).style.format({"Quantized": "{:.4f}", "|Error|": "{:.4f}"}),
            hide_index=True, height=min(400, 50 + 35 * n_dim),
        )
        st.metric("MSE (this vector)", f"{mse:.5f}")
    with c3:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Original",  x=dim_labels, y=qvec,   marker_color="steelblue"))
        fig.add_trace(go.Bar(name="Quantized", x=dim_labels, y=qvec_q, marker_color="tomato", opacity=0.85))
        fig.update_layout(
            barmode="group", title="Original vs Quantized",
            xaxis_title="Dimension", yaxis_title="Value",
            height=300, margin=dict(t=40, b=30),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 1-D number-line lookup
    st.markdown("**Per-dimension codebook lookup**")
    st.caption(
        "Red ticks = centroids  |  gray dashes = boundaries  |  "
        "blue circle = original value  |  red diamond = quantized value"
    )

    fig = make_subplots(
        rows=n_dim, cols=1, subplot_titles=dim_labels,
        shared_xaxes=False, vertical_spacing=0.06,
    )
    for d in range(n_dim):
        c   = centroids_all[d]
        bnd = bounds_all[d][1:-1]
        dmin, dmax = data[:, d].min(), data[:, d].max()
        span = dmax - dmin + 1e-12

        def norm(v, _dmin=dmin, _span=span):
            return (v - _dmin) / _span

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 0], mode="lines",
            line=dict(color="lightgray", width=1), showlegend=False,
        ), row=d + 1, col=1)

        all_bnd_n = [0.0] + [float(norm(b)) for b in bnd] + [1.0]
        for k in range(n_lvl):
            x0, x1 = all_bnd_n[k], all_bnd_n[k + 1]
            fig.add_shape(
                type="rect", x0=x0, x1=x1, y0=-0.5, y1=0.5,
                fillcolor="rgba(200,210,230,0.45)" if k % 2 == 0 else "rgba(230,210,200,0.45)",
                line_width=0, row=d + 1, col=1,
            )

        for bv in bnd:
            bv_n = float(norm(bv))
            fig.add_trace(go.Scatter(
                x=[bv_n, bv_n], y=[-0.45, 0.45], mode="lines",
                line=dict(color="dimgray", dash="dash", width=1), showlegend=False,
            ), row=d + 1, col=1)

        for cv in c:
            cv_n = float(norm(cv))
            fig.add_trace(go.Scatter(
                x=[cv_n, cv_n], y=[-0.4, 0.4], mode="lines",
                line=dict(color="crimson", width=2.5), showlegend=False,
            ), row=d + 1, col=1)
            fig.add_annotation(
                x=cv_n, y=0.55, text=f"{cv:.2f}",
                showarrow=False, font=dict(size=8, color="crimson"),
                row=d + 1, col=1,
            )

        ov_n = float(norm(qvec[d]))
        fig.add_trace(go.Scatter(
            x=[ov_n], y=[0.2], mode="markers",
            marker=dict(size=11, color="steelblue", symbol="circle",
                        line=dict(width=1.5, color="white")),
            showlegend=False, hovertemplate=f"original: {qvec[d]:.4f}",
        ), row=d + 1, col=1)

        qv_n = float(norm(qvec_q[d]))
        fig.add_trace(go.Scatter(
            x=[qv_n], y=[-0.2], mode="markers",
            marker=dict(size=11, color="tomato", symbol="diamond",
                        line=dict(width=1.5, color="white")),
            showlegend=False, hovertemplate=f"quantized: {qvec_q[d]:.4f}  (L{qvec_bins[d]})",
        ), row=d + 1, col=1)

        fig.add_annotation(
            ax=ov_n, ay=0.18, x=qv_n, y=-0.18,
            xref=f"x{d+1}", yref=f"y{d+1}",
            axref=f"x{d+1}", ayref=f"y{d+1}",
            arrowhead=2, arrowwidth=1.2,
            arrowcolor="rgba(100,100,100,0.6)", showarrow=True,
        )
        fig.add_annotation(
            x=0, y=-0.55, text=f"{dmin:.2f}",
            showarrow=False, font=dict(size=8, color="gray"),
            row=d + 1, col=1,
        )
        fig.add_annotation(
            x=1, y=-0.55, text=f"{dmax:.2f}",
            showarrow=False, font=dict(size=8, color="gray"),
            row=d + 1, col=1,
        )

    fig.update_xaxes(showticklabels=False, range=[-0.05, 1.05], showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, range=[-0.75, 0.8],  showgrid=False, zeroline=False)
    fig.update_layout(
        height=max(200, n_dim * 90),
        margin=dict(t=20, b=20, l=30, r=10),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    # ── PQ query quantization ─────────────────────────────────────────────────
    st.markdown(
        "The query vector is split into M sub-vectors. "
        "Each sub-vector is replaced by the nearest centroid in its sub-space."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Original vector**")
        rows = []
        for m, res in enumerate(pq_results):
            for d in res["dims"]:
                rows.append({"Sub-space": f"m{m}", "Dim": dim_labels[d], "Value": qvec[d]})
        st.dataframe(
            pd.DataFrame(rows).style.format({"Value": "{:.4f}"}),
            hide_index=True, height=min(420, 55 + 35 * n_dim),
        )

    with c2:
        st.markdown("**Quantized vector + error**")
        rows_q = []
        for m, res in enumerate(pq_results):
            for di, d in enumerate(res["dims"]):
                rows_q.append({
                    "Sub-space": f"m{m}",
                    "Dim":       dim_labels[d],
                    "Centroid":  f"C{qvec_bins[m]}",
                    "Quantized": qvec_q[d],
                    "|Error|":   abs(qvec_q[d] - qvec[d]),
                })
        mse_pq = np.mean((qvec_q - qvec) ** 2)
        st.dataframe(
            pd.DataFrame(rows_q)
              .style.format({"Quantized": "{:.4f}", "|Error|": "{:.4f}"}),
            hide_index=True, height=min(420, 55 + 35 * n_dim),
        )
        st.metric("MSE (this vector)", f"{mse_pq:.5f}")

    with c3:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Original",  x=dim_labels, y=qvec,   marker_color="steelblue"))
        fig.add_trace(go.Bar(name="Quantized", x=dim_labels, y=qvec_q, marker_color="tomato", opacity=0.85))
        # Sub-space boundary annotations
        for m, res in enumerate(pq_results[:-1]):
            boundary_dim = res["dims"][-1]
            fig.add_vline(
                x=boundary_dim + 0.5,
                line_dash="dot", line_color="gray", line_width=1,
            )
        fig.update_layout(
            barmode="group", title="Original vs Quantized (dashed = sub-space boundaries)",
            xaxis_title="Dimension", yaxis_title="Value",
            height=300, margin=dict(t=45, b=30),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Per-sub-space lookup plot
    st.markdown("**Per-sub-space nearest-centroid lookup**")
    st.caption(
        "1-D sub-spaces: strip plot.  2-D+ sub-spaces: scatter.  "
        "Blue = query sub-vector  |  ✕ = all centroids  |  red ✕ = assigned centroid."
    )

    ncols_lk = min(n_sub, 4)
    nrows_lk = int(np.ceil(n_sub / ncols_lk))

    fig = make_subplots(
        rows=nrows_lk, cols=ncols_lk,
        subplot_titles=[f"sub {m}" for m in range(n_sub)],
        horizontal_spacing=0.06, vertical_spacing=0.15,
    )

    for m, res in enumerate(pq_results):
        r   = m // ncols_lk + 1
        col = m % ncols_lk + 1
        dims_m    = res["dims"]
        sub_dim_m = len(dims_m)
        c_m       = res["centroids"]
        assigned  = qvec_bins[m]
        sub_vec   = qvec[dims_m]

        if sub_dim_m == 1:
            d0   = data[:, dims_m[0]]
            lbl_m = res["labels"]
            # data strip
            for j in range(k_sub):
                mask = lbl_m == j
                if not mask.any():
                    continue
                fig.add_trace(go.Scatter(
                    x=d0[mask], y=np.zeros(mask.sum()),
                    mode="markers",
                    marker=dict(size=4, opacity=0.3,
                                color=CLUSTER_PALETTE[j % len(CLUSTER_PALETTE)]),
                    showlegend=False,
                ), row=r, col=col)
            # all centroids
            for j, cv in enumerate(c_m[:, 0]):
                fig.add_trace(go.Scatter(
                    x=[cv, cv], y=[-0.1, 0.1], mode="lines",
                    line=dict(
                        color="crimson" if j == assigned else "dimgray",
                        width=3 if j == assigned else 1.5,
                    ),
                    showlegend=False,
                ), row=r, col=col)
            # query value
            fig.add_trace(go.Scatter(
                x=[sub_vec[0]], y=[0.15], mode="markers",
                marker=dict(size=10, color="steelblue", symbol="circle",
                            line=dict(width=1.5, color="white")),
                showlegend=False,
            ), row=r, col=col)

        else:
            xs_m = data[:, dims_m[0]]
            ys_m = data[:, dims_m[1]]
            lbl_m = res["labels"]
            for j in range(k_sub):
                mask = lbl_m == j
                if not mask.any():
                    continue
                fig.add_trace(go.Scatter(
                    x=xs_m[mask], y=ys_m[mask], mode="markers",
                    marker=dict(size=4, opacity=0.25,
                                color=CLUSTER_PALETTE[j % len(CLUSTER_PALETTE)]),
                    showlegend=False,
                ), row=r, col=col)
            # all centroids (gray X), assigned centroid (red X)
            for j in range(k_sub):
                is_assigned = (j == assigned)
                fig.add_trace(go.Scatter(
                    x=[c_m[j, 0]], y=[c_m[j, 1]], mode="markers",
                    marker=dict(
                        size=14 if is_assigned else 10,
                        symbol="x",
                        color="crimson" if is_assigned else "dimgray",
                        line=dict(width=3 if is_assigned else 1.5,
                                  color="crimson" if is_assigned else "dimgray"),
                    ),
                    showlegend=False,
                ), row=r, col=col)
            # query sub-vector
            fig.add_trace(go.Scatter(
                x=[sub_vec[0]], y=[sub_vec[1]], mode="markers",
                marker=dict(size=12, color="steelblue", symbol="circle",
                            line=dict(width=2, color="white")),
                showlegend=False,
            ), row=r, col=col)

    fig.update_layout(
        height=max(260, nrows_lk * 260),
        margin=dict(t=50, b=20, l=40, r=10),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
if not is_pq:
    st.markdown(
        "**How Lloyd-Max works:** "
        "Set boundaries = midpoints between centroids, then update centroids = "
        "conditional mean in each bin. Repeating this decreases MSE until convergence."
    )
else:
    st.markdown(
        "**How Product Quantization works:** "
        "Split D-dim vectors into M sub-vectors, run k-means independently in each "
        "sub-space to get K centroids. Encode each vector as M indices. "
        "Total codebook size is M×K centroids vs K^D for full VQ — "
        "exponential savings with controlled accuracy loss."
    )
