# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.13.15",
#     "plotly>=5.20",
#     "numpy>=1.26",
#     "jax[cpu]>=0.4.28",
# ]
# ///

import marimo as mo

app = mo.App()

# ------------------------------------------------
# ⚡️ Notebook title and description (Markdown cell)
# ------------------------------------------------


@app.cell
def title_and_description():
    mo.md(
        """
        # Constrained Adversarial Alignment  
        _Toy experiments illustrating concave‐Pareto optimisation tricks from the “Doubly Aligned Multilingual Parser” paper._
        
        This is a **Marimo** port of the original Colab script.  All non‑essential visualisations have been removed to focus on the optimisation demo.
        """
    )


# -----------------
# 🌈 Color palette
# -----------------


@app.cell
def color_palette():
    colors = {
        "blue": "#1877F2",  # Categorical Blue
        "orange": "#F0701A",  # Categorical Orange
        "purple": "#5A24C7",  # Categorical Purple
        "pink": "#E42C97",  # Categorical Pink
        "cyan": "#0099E6",  # Categorical Cyan
        "teal": "#0EAC96",  # Categorical Teal
        "light_purple": "#AB76FF",  # Categorical Light Purple
        "burgundy": "#B50550",  # Categorical Burgundy
        "dark_blue": "#00487C",  # Categorical Dark Blue
        "brown": "#783301",  # Categorical Brown
        "background": "#f5f7fa",  # Light background
    }
    return colors


# ---------------------------------------------
# 📦 Core imports (shared across cells)
# ---------------------------------------------


@app.cell
def imports():
    import plotly.graph_objects as go
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import grad, jit
    import plotly.io as pio

    pio.templates.default = "plotly_white"
    return go, np, jax, jnp, grad, jit


# ----------------------------------------------------------
# 🧮 Optimisation toy‑problem: concave Pareto front (JAX)
# ----------------------------------------------------------


@app.cell
def pareto_losses(jnp, np, jit):
    eps = np.sqrt(2) / 2

    @jit
    def l1(t1, t2):
        t1 = jnp.clip(t1, 0, np.pi / 2)
        return jnp.sin(t1) * (1 + t2**2)

    @jit
    def l2(t1, t2):
        t1 = jnp.clip(t1, 0, np.pi / 2)
        return jnp.cos(t1) * (1 + t2**2)

    return l1, l2, eps


@app.cell
def optimise_and_plot(go, np, jnp, grad, jit, pareto_losses, color_palette):
    colors = color_palette
    l1, l2, eps = pareto_losses

    @jit
    def combined_loss(t, alpha):
        return l1(t[0], t[1]) - alpha * (eps - l2(t[0], t[1]))

    combined_grad = grad(combined_loss, argnums=0)

    def optimise(initial_point, alpha, lr=0.02, steps=200):
        t = initial_point
        traj = [t]
        for _ in range(steps):
            g = combined_grad((t[0], t[1]), alpha)
            t = (t[0] - lr * g[0], t[1] - lr * g[1])
            traj.append(t)
        return np.array(traj)

    # Two alpha settings to show failure on concave front
    start = (np.pi / 4, 1.059)
    alphas = [0.99, 1.01]
    trajectories = [optimise(start, a) for a in alphas]

    # ── Build loss‑space figure ────────────────────────────────────
    fig = go.Figure()

    pareto_t1 = np.linspace(0, np.pi / 2, 120)
    pareto_l1 = [l1(t1, 0) for t1 in pareto_t1]
    pareto_l2 = [l2(t1, 0) for t1 in pareto_t1]
    fig.add_trace(
        go.Scatter(
            x=pareto_l1,
            y=pareto_l2,
            mode="lines",
            line=dict(color="black", width=3),
            name="Pareto front",
        )
    )

    palette = [colors["blue"], colors["orange"]]
    for idx, (traj, a) in enumerate(zip(trajectories, alphas)):
        losses = np.array([(l1(*p), l2(*p)) for p in traj])
        fig.add_trace(
            go.Scatter(
                x=losses[:, 0],
                y=losses[:, 1],
                mode="lines+markers",
                line=dict(color=palette[idx], width=2),
                marker=dict(color=palette[idx], size=5),
                name=f"α = {a}",
            )
        )

    fig.update_layout(
        title="Gradient descent with different α values",
        xaxis_title="loss₁",
        yaxis_title="loss₂",
        height=600,
        width=800,
        paper_bgcolor=colors["background"],
        plot_bgcolor=colors["background"],
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


@app.cell
def show_pareto_fig(mo, optimise_and_plot):
    mo.output.replace(optimise_and_plot)


# ------------------------------------
# 📜 Citation block (Markdown cell)
# ------------------------------------


@app.cell
def citation():
    mo.md(
        """
        ---
        #### Citation
        ```bibtex
        @inproceedings{held-etal-2023-damp,
          title   = {DAMP: Doubly Aligned Multilingual Parser for Task-Oriented Dialogue},
          author  = {Held, William and Hidey, Christopher and Liu, Fei and Zhu, Eric and Goel, Rahul and Yang, Diyi and Shah, Rushin},
          booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
          year = {2023},
          address = {Toronto, Canada},
          url = {https://aclanthology.org/2023.acl-long.199/}
        }
        ```
        """
    )


# Run the app when executed as a script
if __name__ == "__main__":
    app.run()
