"""
Vector Quantization Explorer
Interactive Streamlit app demonstrating scalar (per-dimension) quantization
using the Lloyd-Max algorithm.
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
# Core algorithm
# ─────────────────────────────────────────────────────────────────────────────

def lloyd_max(data: np.ndarray, n_levels: int, max_iter: int = 60, tol: float = 1e-9):
    """
    Scalar Lloyd-Max quantizer for 1-D data.
    Returns (centroids, full_boundaries, history_of_centroids).
    full_boundaries has length n_levels+1 with ±inf at the ends.
    """
    lo, hi = data.min(), data.max()
    # Uniform initialisation
    centroids = np.linspace(lo, hi, n_levels)
    history = [centroids.copy()]

    for _ in range(max_iter):
        # Partition boundaries = midpoints between consecutive centroids
        interior = (centroids[:-1] + centroids[1:]) / 2
        # Assign every sample to its nearest bin  (0 … n_levels-1)
        labels = np.digitize(data, interior)
        # New centroids = conditional means
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


def quantize_vector(vec: np.ndarray, centroids_all, bounds_all):
    """Quantize a single vector using per-dimension codebooks."""
    q = np.empty_like(vec)
    bins = np.empty(len(vec), dtype=int)
    for d in range(len(vec)):
        interior = bounds_all[d][1:-1]          # finite boundaries
        idx = int(np.digitize(vec[d], interior))
        q[d] = centroids_all[d][idx]
        bins[d] = idx
    return q, bins


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
    # Mixed: cycle Gaussian / Uniform / Bimodal across dimensions
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
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────

for key, val in [("seed", 42), ("qseed", 777)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Controls")

    n_vec = st.slider("Number of vectors",          20, 300,  80, step=10)
    n_dim = st.slider("Vector dimensions",            2,  12,   5)
    n_lvl = st.slider("Quantization levels / dim",   2,  16,   4)
    dist  = st.selectbox("Distribution",
                         ["Gaussian", "Uniform", "Bimodal", "Mixed"],
                         index=3)
    sel_d = st.slider("Dimension for step-by-step demo", 0, n_dim - 1, 0)

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
    st.markdown(
        "**About**\n\n"
        "Scalar quantization replaces each dimension value with the nearest "
        "**centroid** from a learned codebook. "
        "The **Lloyd-Max** algorithm iterates:\n"
        "1. Set boundaries = midpoints between centroids\n"
        "2. Re-compute centroids = conditional mean inside each bin\n"
        "until convergence."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Compute
# ─────────────────────────────────────────────────────────────────────────────

data = make_data(n_vec, n_dim, st.session_state.seed, dist)

centroids_all, bounds_all, history_all = [], [], []
for d in range(n_dim):
    c, b, h = lloyd_max(data[:, d], n_lvl)
    centroids_all.append(c)
    bounds_all.append(b)
    history_all.append(h)

rng_q   = np.random.default_rng(st.session_state.qseed)
qvec    = rng_q.normal(0, 1.2, n_dim)
qvec_q, qvec_bins = quantize_vector(qvec, centroids_all, bounds_all)

dim_labels = [f"d{d}" for d in range(n_dim)]
PALETTE    = px.colors.qualitative.Set2

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

st.title("Vector Quantization Explorer")
st.markdown(
    "**Scalar quantization** maps each vector dimension independently. "
    "The **Lloyd-Max algorithm** finds optimal bin boundaries and centroids "
    "(representatives) that minimise mean-squared quantization error. "
    "Use the sidebar to change the dataset or algorithm parameters."
)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Dataset matrix
# ─────────────────────────────────────────────────────────────────────────────

st.header("1 · Dataset")
c1, c2 = st.columns([3, 1])

with c1:
    fig = go.Figure(go.Heatmap(
        z=data,
        x=dim_labels,
        y=[f"v{i}" for i in range(n_vec)],
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="value", thickness=12),
    ))
    fig.update_layout(
        title=f"{n_vec} vectors  ×  {n_dim} dimensions  ({dist} distribution)",
        xaxis_title="Dimension",
        yaxis_title="Vector index",
        height=min(500, max(250, n_vec * 4)),
        margin=dict(l=60, r=20, t=45, b=40),
    )
    st.plotly_chart(fig, width='stretch')

with c2:
    st.markdown("**First 10 rows**")
    df_show = pd.DataFrame(data[:10], columns=dim_labels)
    st.dataframe(df_show.style.format("{:.3f}"), height=340)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Value distributions
# ─────────────────────────────────────────────────────────────────────────────

st.header("2 · Value Distributions")

fig = go.Figure()
for d in range(n_dim):
    fig.add_trace(go.Violin(
        x=[dim_labels[d]] * n_vec,
        y=data[:, d],
        name=dim_labels[d],
        box_visible=True,
        meanline_visible=True,
        fillcolor=PALETTE[d % len(PALETTE)],
        line_color="rgba(0,0,0,0.35)",
        opacity=0.75,
        showlegend=False,
        points="outliers",
        jitter=0.3,
        pointpos=0,
    ))
fig.update_layout(
    xaxis_title="Dimension",
    yaxis_title="Value",
    title="Per-dimension distribution (violin + box + mean line)",
    height=370,
    margin=dict(t=45, b=40),
)
st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Lloyd-Max convergence (selected dimension)
# ─────────────────────────────────────────────────────────────────────────────

st.header(f"3 · Lloyd-Max Convergence — dimension {sel_d}")

history   = history_all[sel_d]
dim_data  = data[:, sel_d]
n_steps   = len(history)   # includes initialisation as step 0

# Choose which steps to display (init + first few + last few)
if n_steps <= 7:
    show = list(range(n_steps))
else:
    show = sorted(set([0, 1, 2, 3] + list(range(max(4, n_steps - 3), n_steps))))
n_show = len(show)

counts, bin_edges = np.histogram(dim_data, bins=30)
bar_w  = bin_edges[1] - bin_edges[0]
bar_x  = (bin_edges[:-1] + bin_edges[1:]) / 2
y_max  = counts.max() * 1.2
x_lo   = dim_data.min() - 0.4
x_hi   = dim_data.max() + 0.4

titles = ["Init" if s == 0 else f"Step {s}" for s in show]
fig = make_subplots(
    rows=1, cols=n_show,
    subplot_titles=titles,
    shared_yaxes=True,
    horizontal_spacing=0.03,
)

for ci, step in enumerate(show, 1):
    c = history[step]
    interior = (c[:-1] + c[1:]) / 2 if len(c) > 1 else np.array([])

    # Histogram bars
    fig.add_trace(go.Bar(
        x=bar_x, y=counts,
        width=bar_w,
        marker_color="lightsteelblue",
        marker_line_color="white",
        marker_line_width=0.4,
        showlegend=False,
    ), row=1, col=ci)

    # Partition boundaries (dashed gray)
    for bv in interior:
        fig.add_trace(go.Scatter(
            x=[bv, bv], y=[0, y_max],
            mode="lines",
            line=dict(color="dimgray", dash="dash", width=1),
            showlegend=False,
        ), row=1, col=ci)

    # Centroids (solid red)
    for cv in c:
        fig.add_trace(go.Scatter(
            x=[cv, cv], y=[0, y_max],
            mode="lines",
            line=dict(color="crimson", width=2.5),
            showlegend=False,
        ), row=1, col=ci)

fig.update_xaxes(range=[x_lo, x_hi])
fig.update_yaxes(range=[0, y_max])
fig.update_layout(
    height=290,
    title=(
        f"Centroids (red) converging over {n_steps - 1} iterations  |  "
        "dashed gray = partition boundaries"
        + (f"  |  showing {n_show} of {n_steps} steps" if n_show < n_steps else "")
    ),
    bargap=0.04,
    margin=dict(t=65, b=30, l=40, r=10),
)
st.plotly_chart(fig, width='stretch')

with st.expander("Show centroid values at each displayed step"):
    step_df = pd.DataFrame(
        {(f"Init" if s == 0 else f"Step {s}"): history[s] for s in show},
        index=[f"L{k}" for k in range(n_lvl)],
    )
    st.dataframe(step_df.style.format("{:.4f}"))

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Summary across all dimensions
# ─────────────────────────────────────────────────────────────────────────────

st.header("4 · Lloyd-Max Results — All Dimensions")

fig = go.Figure()

for d in range(n_dim):
    # Background violin
    fig.add_trace(go.Violin(
        x=[dim_labels[d]] * n_vec,
        y=data[:, d],
        side="positive",
        fillcolor=PALETTE[d % len(PALETTE)],
        line_color="rgba(120,120,120,0.4)",
        opacity=0.22,
        showlegend=False,
        points=False,
        name=dim_labels[d],
    ))

    c   = centroids_all[d]
    bnd = bounds_all[d][1:-1]   # finite interior boundaries

    # Boundaries
    fig.add_trace(go.Scatter(
        x=[dim_labels[d]] * len(bnd),
        y=bnd,
        mode="markers",
        marker=dict(
            symbol="line-ew-open", size=20, color="dimgray",
            line=dict(width=1.8, color="dimgray"),
        ),
        name="boundary" if d == 0 else None,
        showlegend=(d == 0),
        legendgroup="bnd",
    ))

    # Centroids
    fig.add_trace(go.Scatter(
        x=[dim_labels[d]] * len(c),
        y=c,
        mode="markers",
        marker=dict(
            symbol="line-ew-open", size=28, color="crimson",
            line=dict(width=2.8, color="crimson"),
        ),
        name="centroid" if d == 0 else None,
        showlegend=(d == 0),
        legendgroup="ctr",
    ))

fig.update_layout(
    xaxis_title="Dimension",
    yaxis_title="Value",
    title="Data distribution with Lloyd-Max centroids (red) and boundaries (gray)",
    height=430,
    margin=dict(t=45, b=40),
    legend=dict(orientation="h", y=-0.14, x=0.3),
)
st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Codebook
# ─────────────────────────────────────────────────────────────────────────────

st.header("5 · Codebook")

cb_arr = np.column_stack(centroids_all)          # (n_lvl, n_dim)
cb_df  = pd.DataFrame(
    cb_arr,
    columns=dim_labels,
    index=[f"L{k}" for k in range(n_lvl)],
)

c1, c2 = st.columns([1, 2])
with c1:
    st.markdown(
        "Each cell is the **centroid** (representative value) for that "
        "quantization level in that dimension.  "
        "A quantized vector is encoded as a list of level indices."
    )
    st.dataframe(
        cb_df.style
             .format("{:.4f}")
             .background_gradient(cmap="RdBu_r", axis=None),
        height=300,
    )
with c2:
    fig = go.Figure(go.Heatmap(
        z=cb_arr,
        x=dim_labels,
        y=cb_df.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(cb_arr, 3),
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="centroid", thickness=12),
    ))
    fig.update_layout(
        title="Codebook heatmap  (row = quantization level, col = dimension)",
        xaxis_title="Dimension",
        yaxis_title="Quantization level",
        height=300,
        margin=dict(t=45, b=30),
    )
    st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Quantize a new vector
# ─────────────────────────────────────────────────────────────────────────────

st.header("6 · Quantize a New Vector")
st.markdown(
    "A freshly drawn query vector is quantized dimension-by-dimension: "
    "each value is looked up in the codebook and replaced with the nearest centroid."
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Original vector**")
    orig_df = pd.DataFrame({
        "Dim":   dim_labels,
        "Value": qvec,
    })
    st.dataframe(
        orig_df.style.format({"Value": "{:.4f}"}),
        hide_index=True, height=min(400, 50 + 35 * n_dim),
    )

with c2:
    st.markdown("**Quantized vector + error**")
    q_df = pd.DataFrame({
        "Dim":       dim_labels,
        "Bin":       [f"L{qvec_bins[d]}" for d in range(n_dim)],
        "Quantized": qvec_q,
        "|Error|":   np.abs(qvec_q - qvec),
    })
    mse = np.mean((qvec_q - qvec) ** 2)
    st.dataframe(
        q_df.style.format({"Quantized": "{:.4f}", "|Error|": "{:.4f}"}),
        hide_index=True, height=min(400, 50 + 35 * n_dim),
    )
    st.metric("MSE (this vector)", f"{mse:.5f}")

with c3:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Original", x=dim_labels, y=qvec,
        marker_color="steelblue",
    ))
    fig.add_trace(go.Bar(
        name="Quantized", x=dim_labels, y=qvec_q,
        marker_color="tomato", opacity=0.85,
    ))
    fig.update_layout(
        barmode="group",
        title="Original vs Quantized",
        xaxis_title="Dimension",
        yaxis_title="Value",
        height=300,
        margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, width='stretch')

# ── Per-dimension 1-D number-line lookup ──────────────────────────────────────

st.markdown("**Per-dimension codebook lookup** — where does the query value fall?")
st.caption(
    "Each row is one dimension's number line (normalized to [0,1]). "
    "Red ticks = centroids | gray dashes = boundaries | "
    "blue circle = original value | red diamond = quantized value."
)

fig = make_subplots(
    rows=n_dim, cols=1,
    subplot_titles=dim_labels,
    shared_xaxes=False,
    vertical_spacing=0.06,
)

for d in range(n_dim):
    c   = centroids_all[d]
    bnd = bounds_all[d][1:-1]
    dmin, dmax = data[:, d].min(), data[:, d].max()
    span = dmax - dmin + 1e-12

    def norm(v):
        return (v - dmin) / span

    # Baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0],
        mode="lines",
        line=dict(color="lightgray", width=1),
        showlegend=False,
    ), row=d + 1, col=1)

    # Bin shading via alternating rectangles
    all_bnd_n = [0.0] + [float(norm(b)) for b in bnd] + [1.0]
    for k in range(n_lvl):
        x0, x1 = all_bnd_n[k], all_bnd_n[k + 1]
        color = "rgba(200,210,230,0.45)" if k % 2 == 0 else "rgba(230,210,200,0.45)"
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=-0.5, y1=0.5,
            fillcolor=color,
            line_width=0,
            row=d + 1, col=1,
        )

    # Boundaries (dashed ticks)
    for bv in bnd:
        bv_n = float(norm(bv))
        fig.add_trace(go.Scatter(
            x=[bv_n, bv_n], y=[-0.45, 0.45],
            mode="lines",
            line=dict(color="dimgray", dash="dash", width=1),
            showlegend=False,
        ), row=d + 1, col=1)

    # Centroids (solid red ticks)
    for cv in c:
        cv_n = float(norm(cv))
        fig.add_trace(go.Scatter(
            x=[cv_n, cv_n], y=[-0.4, 0.4],
            mode="lines",
            line=dict(color="crimson", width=2.5),
            showlegend=False,
        ), row=d + 1, col=1)
        # centroid label (value)
        fig.add_annotation(
            x=cv_n, y=0.55, text=f"{cv:.2f}",
            showarrow=False, font=dict(size=8, color="crimson"),
            row=d + 1, col=1,
        )

    # Original value
    ov_n = float(norm(qvec[d]))
    fig.add_trace(go.Scatter(
        x=[ov_n], y=[0.2],
        mode="markers",
        marker=dict(size=11, color="steelblue", symbol="circle",
                    line=dict(width=1.5, color="white")),
        showlegend=False,
        hovertemplate=f"original: {qvec[d]:.4f}",
    ), row=d + 1, col=1)

    # Quantized value
    qv_n = float(norm(qvec_q[d]))
    fig.add_trace(go.Scatter(
        x=[qv_n], y=[-0.2],
        mode="markers",
        marker=dict(size=11, color="tomato", symbol="diamond",
                    line=dict(width=1.5, color="white")),
        showlegend=False,
        hovertemplate=f"quantized: {qvec_q[d]:.4f}  (L{qvec_bins[d]})",
    ), row=d + 1, col=1)

    # Arrow connecting original → quantized
    fig.add_annotation(
        ax=ov_n, ay=0.18,
        x=qv_n, y=-0.18,
        xref=f"x{d+1}", yref=f"y{d+1}",
        axref=f"x{d+1}", ayref=f"y{d+1}",
        arrowhead=2, arrowwidth=1.2,
        arrowcolor="rgba(100,100,100,0.6)",
        showarrow=True,
    )

    # Axis range label (actual values at 0 and 1)
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
st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "**How Lloyd-Max works in one sentence:** "
    "Repeatedly set boundaries to midpoints between centroids, "
    "then update centroids to the conditional mean of their bin — "
    "this decreases MSE monotonically until convergence."
)
