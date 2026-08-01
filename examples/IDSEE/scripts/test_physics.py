"""
BCOS case studies
=================

Three representative scenarios:

Scenario 1:
    Poorly positioned defender.

Scenario 2:
    Well positioned defender.

Scenario 3:
    Two defenders with different spatial advantage.

Computes:

    S(t), P_def(t), U(t)

    p_BC(x,t)

    J_BC(t) = U(t) * h_BC(t)

    J_D(x,t) = U(t) * a * F_D(x,t) * p_BC(x,t)
             = U(t) * lambda_D(x,t) * p_BC(x,t)

    (the factor a = pi/(sqrt(3)*s) makes J_D directly comparable to
    J_BC, since h_BC = a*F_BC and h_D = a*m(t) = a * sum_x F_D(x,t)*p_BC(x,t)
    both already carry that factor -- J_D(x,t) must carry it too, or
    the two fluxes are not on the same scale)

All quantities share the same solver time grid.
"""


import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


from BCOS.model.path import build_path

from BCOS.model.arrival import (
    F,
    p_bc,
)

from BCOS.model.hazard import (
    ballcarrier_hazard,
    defender_hazard,
)

from BCOS.model.solver_bcos import integrate_point



# ============================================================
# PARAMETERS
# ============================================================

PARAMS = dict(

    s=0.45,

    tau_r_bc=0.2,
    tau_r_def=0.7,

    v_bc_max=7.62,
    v_def_max=8.54,

    alpha_rot=1.0,

    MAX_POINTS=300,
    dx=0.1,

    dt=0.04,
    T_max=10,

    tol=1e-6,
)



# ============================================================
# SCENARIOS
# ============================================================

SCENARIOS = {


    "Scenario 1": dict(

        x_bc0=jnp.array([0.0,0.0]),

        v_bc0=jnp.array([6.0,0.0]),

        x_target=jnp.array([20.0,0.0]),


        r_def=jnp.array([

            [-5.0,-10.0],

        ]),


        v_def=jnp.array([

            [0.5,1.0],

        ]),
    ),



    "Scenario 2": dict(

        x_bc0=jnp.array([0.0,0.0]),

        v_bc0=jnp.array([6.0,0.0]),

        x_target=jnp.array([20.0,0.0]),


        r_def=jnp.array([

            [10.0,0.5],

        ]),


        v_def=jnp.array([

            [-1.0,-0.3],

        ]),
    ),



    "Scenario 3": dict(

        x_bc0=jnp.array([0.0,0.0]),

        v_bc0=jnp.array([6.0,0.0]),

        x_target=jnp.array([20.0,0.0]),


        r_def=jnp.array([

            [9.0,1.0],

            [16.0,8.0],

        ]),


        v_def=jnp.array([

            [-1.0,-0.2],

            [-0.5,-1.0],

        ]),
    ),

}



# ============================================================
# ANALYSIS
# ============================================================


def analyze_scenario(
    scenario,
    params=PARAMS,
):


    x_bc0 = scenario["x_bc0"]
    v_bc0 = scenario["v_bc0"]

    x_target = scenario["x_target"]

    r_def = scenario["r_def"]
    v_def = scenario["v_def"]



    x_path, mask = build_path(
        x_target,
        x_bc0,

        MAX_POINTS=params["MAX_POINTS"],
        dx=params["dx"],
    )



    trajectory = integrate_point(

        x_target,

        x_bc0,
        v_bc0,

        r_def,
        v_def,

        MAX_POINTS=params["MAX_POINTS"],
        dx=params["dx"],

        s=params["s"],

        tau_r_bc=params["tau_r_bc"],
        tau_r_def=params["tau_r_def"],

        alpha_rot=params["alpha_rot"],

        v_bc_max=params["v_bc_max"],
        v_def_max=params["v_def_max"],

        dt=params["dt"],
        T_max=params["T_max"],

        tol=params["tol"],
    )



    S_t = trajectory[0]

    P_def_t = trajectory[1]

    t_grid = trajectory[2]



    U_t = (
        1.0
        - S_t
        - jnp.sum(P_def_t,axis=1)
    )



    # --------------------------------------------------------
    # BC spatial distribution
    # --------------------------------------------------------

    px_series=[]


    for t in t_grid:


        px_series.append(

            p_bc(

                x_bc0,
                v_bc0,

                x_path,

                mask,

                t,

                params["s"],
                params["tau_r_bc"],

                params["v_bc_max"],
                params["alpha_rot"],
            )
        )


    px_series=jnp.stack(
        px_series,
        axis=0
    )



    # --------------------------------------------------------
    # hazards
    # --------------------------------------------------------

    def h_bc(t):

        return ballcarrier_hazard(

            x_bc0,
            v_bc0,

            x_target,

            t,

            params["s"],
            params["tau_r_bc"],

            params["v_bc_max"],
            params["alpha_rot"],
        )



    def h_def(t):

        return defender_hazard(

            x_bc0,
            v_bc0,

            r_def,
            v_def,

            x_path,
            mask,

            t,

            params["s"],

            params["tau_r_def"],
            params["tau_r_bc"],

            params["v_def_max"],
            params["v_bc_max"],

            params["alpha_rot"],
        )



    h_bc_series=jax.vmap(h_bc)(t_grid)

    h_def_series=jax.vmap(h_def)(t_grid)



    bc_flux=U_t*h_bc_series



    # --------------------------------------------------------
    # spatial flux
    #
    # J_D(x,t) = U(t) * a * F_D(x,t) * p_BC(x,t)
    #
    # NOTE: the factor a = pi/(sqrt(3)*s) is required here so that
    # J_D is on the same scale as bc_flux (= U*h_BC = U*a*F_BC), and
    # so that sum_x J_D(x,t) == U(t)*h_def_series(t) exactly (the
    # aggregate hazard already computed via defender_hazard).
    # --------------------------------------------------------

    a_hazard = jnp.pi / (jnp.sqrt(3.0) * params["s"])

    spatial_flux=[]


    for i,t in enumerate(t_grid):


        F_def=F(

            r_def,
            v_def,

            x_path,

            t,

            params["s"],
            params["tau_r_def"],

            params["v_def_max"],
            params["alpha_rot"],
        )


        J_D = (

            a_hazard
            *
            U_t[i]
            *
            F_def
            *
            px_series[i][None,:]

        )


        spatial_flux.append(J_D)



    spatial_flux=jnp.stack(
        spatial_flux,
        axis=0
    )



    return {

        "scenario":scenario,

        "x_path":x_path,

        "mask":mask,


        "t_grid":t_grid,


        "S_t":S_t,

        "P_def_t":P_def_t,

        "U_t":U_t,


        "px_series":px_series,


        "h_bc_series":h_bc_series,

        "h_def_series":h_def_series,


        "bc_flux":bc_flux,


        "spatial_flux":spatial_flux,
    }


# ============================================================
# PLOTTING UTILITIES
# ============================================================


def path_coordinate(x_path):

    """
    Curvilinear coordinate along BC path.
    """

    ds = np.linalg.norm(
        np.diff(
            np.asarray(x_path),
            axis=0
        ),
        axis=1
    )

    return np.concatenate(
        (
            [0.0],
            np.cumsum(ds)
        )
    )


def reaction_trimmed_path(results, params=PARAMS):
    """
    Returns (distance, valid_mask_indices) for the path restricted to
    points at or beyond x_start = x_bc0 + v_bc0*tau_r_bc, using only
    valid (masked) path points. Shared by all spatial plots so the
    trimming logic lives in exactly one place.
    """

    x_path = np.asarray(results["x_path"])
    mask = np.asarray(results["mask"]) > 0
    scenario = results["scenario"]

    x_valid = x_path[mask]

    x_reaction = (
        np.asarray(scenario["x_bc0"])
        + np.asarray(scenario["v_bc0"]) * params["tau_r_bc"]
    )

    i0 = int(np.argmin(np.linalg.norm(x_valid - x_reaction, axis=1)))

    distance = path_coordinate(x_valid[i0:])

    return distance, mask, i0


# ============================================================
# Geometry
# ============================================================


def plot_geometry(
    results,
    title=""
):

    scenario = results["scenario"]


    x_bc0 = np.asarray(
        scenario["x_bc0"]
    )

    v_bc0 = np.asarray(
        scenario["v_bc0"]
    )

    x_target = np.asarray(
        scenario["x_target"]
    )


    r_def = np.asarray(
        scenario["r_def"]
    )

    v_def = np.asarray(
        scenario["v_def"]
    )


    x_path = np.asarray(
        results["x_path"]
    )

    mask = np.asarray(
        results["mask"]
    ) > 0



    plt.figure(
        figsize=(7,6)
    )


    plt.plot(
        *x_bc0,
        "ko",
        markersize=10,
        label="BC start"
    )


    plt.quiver(
        *x_bc0,
        *v_bc0,
        angles="xy",
        scale_units="xy",
        scale=1
    )


    plt.plot(
        *x_target,
        "*",
        markersize=18,
        label="Target"
    )


    plt.plot(
        x_path[mask,0],
        x_path[mask,1],
        "--",
        label="BC path"
    )


    for i in range(
        len(r_def)
    ):

        plt.plot(
            *r_def[i],
            "x",
            markersize=12,
            markeredgewidth=3,
            label=f"D{i+1}"
        )


        plt.quiver(
            *r_def[i],
            *v_def[i],
            angles="xy",
            scale_units="xy",
            scale=1
        )


    plt.axis(
        "equal"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.title(
        title
    )

    plt.tight_layout()

    plt.show()



# ============================================================
# Hazards
# ============================================================


def plot_hazards(
    results,
    title=""
):

    t = np.asarray(
        results["t_grid"]
    )


    h_bc = np.asarray(
        results["h_bc_series"]
    )


    h_def = np.asarray(
        results["h_def_series"]
    )


    plt.figure(
        figsize=(8,4.5)
    )


    plt.plot(
        t,
        h_bc,
        linewidth=2,
        label=r"$h_{BC}(t)$"
    )


    for i in range(
        h_def.shape[1]
    ):

        plt.plot(
            t,
            h_def[:,i],
            linewidth=2,
            label=fr"$h_{{D{i+1}}}(t)$"
        )


    plt.xlabel(
        "time (s)"
    )

    plt.ylabel(
        "hazard"
    )

    plt.title(
        title+" -- hazards"
    )


    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()



# ============================================================
# Competing risks
# ============================================================


def plot_competing_risk(
    results,
    title=""
):

    t = np.asarray(
        results["t_grid"]
    )


    S = np.asarray(
        results["S_t"]
    )

    P = np.asarray(
        results["P_def_t"]
    )

    U = np.asarray(
        results["U_t"]
    )


    plt.figure(
        figsize=(8,4.5)
    )


    plt.plot(
        t,
        S,
        linewidth=2,
        label=r"$S(t)$"
    )


    for i in range(
        P.shape[1]
    ):

        plt.plot(
            t,
            P[:,i],
            linewidth=2,
            label=fr"$P_{{D{i+1}}}(t)$"
        )


    plt.plot(
        t,
        U,
        "--",
        linewidth=2,
        label=r"$U(t)$"
    )


    plt.xlabel(
        "time (s)"
    )

    plt.ylabel(
        "probability"
    )


    plt.title(
        title+" -- competing risk"
    )


    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


def plot_bc_spatial_distribution(
    results,
    snapshot_times=(0.5, 1.0, 1.5, 2.0),
    title="",
    params=PARAMS,
):

    px_series = np.asarray(results["px_series"])
    t_grid = np.asarray(results["t_grid"])

    distance, mask, i0 = reaction_trimmed_path(results, params)
    px_valid = px_series[:, mask][:, i0:]

    plt.figure(
        figsize=(8,4.5)
    )


    for t in snapshot_times:

        # nearest point on the actual solver time grid, instead of
        # int(t/dt) which truncates (systematic bias) and silently
        # assumes params["dt"] matches the dt used to build t_grid
        idx = int(np.argmin(np.abs(t_grid - t)))

        plt.plot(
            distance,
            px_valid[idx],
            linewidth=2,
            label=f"t={t_grid[idx]:.2f}s"
        )


    plt.xlabel(
        "distance along BC path (m)"
    )

    plt.ylabel(
        r"$p_{BC}(x,t)$"
    )


    plt.title(
        title+" -- BC spatial distribution"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


def plot_temporal_fluxes(
    results,
    title=""
):

    t_grid = np.asarray(
        results["t_grid"]
    )


    bc_flux = np.asarray(
        results["bc_flux"]
    )


    spatial_flux = np.asarray(
        results["spatial_flux"]
    )


    # --------------------------------------------------------
    # Defender temporal flux
    #
    # spatial_flux:
    # (N_time, N_defenders, N_points)
    #
    # J_D(t) = sum_x J_D(x,t)
    #
    # By construction this now equals U(t)*h_def_series(t) exactly
    # (same scale as bc_flux = U(t)*h_bc_series(t)), since spatial_flux
    # already carries the factor a = pi/(sqrt(3)*s).
    # --------------------------------------------------------

    def_flux = np.sum(
        spatial_flux,
        axis=2
    )



    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(8,4.5)
    )


    plt.plot(
        t_grid,
        bc_flux,
        linewidth=2,
        label=r"$J_{BC}(t)$"
    )


    for i in range(
        def_flux.shape[1]
    ):

        plt.plot(
            t_grid,
            def_flux[:,i],
            linewidth=2,
            label=fr"$J_{{D{i+1}}}(t)$"
        )


    plt.xlabel(
        "time (s)"
    )

    plt.ylabel(
        "probability flux"
    )


    plt.title(
        title+" -- temporal probability fluxes"
    )


    plt.grid(
        alpha=0.3
    )


    plt.legend()

    plt.tight_layout()

    plt.show()



# ==================
# Defender flux 
# ==================


def plot_defender_flux(
    results,
    title="",
    params=PARAMS,
    max_alpha=0.9,
):
    """
    Single (distance, time) panel comparing all defenders' spatio-
    temporal flux J_D(x,t) at once. Each defender gets a distinct
    base color; opacity at each (x,t) point is proportional to that
    defender's flux, normalized to the SAME global scale across all
    defenders (so intensity is directly comparable). Overlapping
    hot zones blend colors, making it easy to see, at a glance,
    which defender dominates where/when, and where two defenders
    are simultaneously "in the race".
    """

    t_grid = np.asarray(results["t_grid"])
    spatial_flux = np.asarray(results["spatial_flux"])  # (N_t, N_def, N_points_full)

    distance, mask, i0 = reaction_trimmed_path(results, params)
    flux_valid = spatial_flux[:, :, mask][:, :, i0:]  # (N_t, N_def, N_points_valid)

    n_def = flux_valid.shape[1]
    vmax = flux_valid.max()
    if vmax <= 0:
        vmax = 1.0

    # normalized alpha per defender, shape (N_def, N_t, N_points_valid)
    alpha = np.clip(flux_valid / vmax, 0.0, 1.0) * max_alpha
    alpha = np.moveaxis(alpha, 1, 0)

    colors = plt.cm.tab10(np.linspace(0, 1, n_def))[:, :3]

    # start from a white canvas, alpha-composite each defender on top
    img = np.ones(flux_valid.shape[0:1] + flux_valid.shape[2:3] + (3,))  # (N_t, N_points, 3)
    for i in range(n_def):
        a_i = alpha[i][:, :, None]
        img = img * (1.0 - a_i) + colors[i][None, None, :] * a_i

    fig, ax = plt.subplots(figsize=(9, 6.5))

    ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=[distance[0], distance[-1], t_grid[0], t_grid[-1]],
    )

    for i in range(n_def):
        ax.plot([], [], color=colors[i], linewidth=8, label=f"D{i+1}")

    ax.set_xlabel("distance along BC path (m)")
    ax.set_ylabel("time (s)")
    ax.set_title(title + " -- defender flux comparison (opacity = flux magnitude)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


def plot_defender_flux(
    results,
    title=""
):

    spatial_flux = np.asarray(
        results["spatial_flux"]
    )

    x_path = np.asarray(
        results["x_path"]
    )

    mask = np.asarray(
        results["mask"]
    ) > 0

    t_grid = np.asarray(
        results["t_grid"]
    )


    # --------------------------------------------------------
    # Spatial coordinate along BC path
    # --------------------------------------------------------

    x_valid = x_path[mask]

    distance = np.concatenate(
        (
            [0.0],
            np.cumsum(
                np.linalg.norm(
                    np.diff(
                        x_valid,
                        axis=0
                    ),
                    axis=1
                )
            )
        )
    )


    X, T = np.meshgrid(
        distance,
        t_grid
    )


    # --------------------------------------------------------
    # Figure style
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9,6)
    )

    fig.patch.set_facecolor(
        "#eaeaea"
    )

    ax.set_facecolor(
        "#f7f7f7"
    )


    # --------------------------------------------------------
    # Colour maps
    # --------------------------------------------------------

    cmaps = [
        "Reds",
        "Blues",
        "Greens",
        "Purples",
        "Oranges",
    ]


    n_def = spatial_flux.shape[1]


    # --------------------------------------------------------
    # Plot all defenders
    # --------------------------------------------------------

    for i in range(n_def):

        J = spatial_flux[:, i, :]

        J = J[:, mask]


        # avoid empty/zero fields

        if np.max(J) <= 0:
            continue


        levels = np.linspace(
            0.1*np.max(J),
            np.max(J),
            8
        )


        ax.contour(
            X,
            T,
            J,
            levels=levels,
            cmap=cmaps[i % len(cmaps)],
            linewidths=2
        )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "distance along BC path (m)"
    )

    ax.set_ylabel(
        "time (s)"
    )


    ax.set_title(
        title +
        " -- defender spatial flux $J_D(x,t)$"
    )


    ax.grid(
        alpha=0.3
    )


    plt.tight_layout()

    plt.show()
    
# ============================================================
# RUN EXAMPLE
# ============================================================


example = "Scenario 3"


res = analyze_scenario(
    SCENARIOS[example],
    PARAMS
)


plot_geometry(
    res,
    example
)


plot_hazards(
    res,
    example
)

plot_temporal_fluxes(
    res,
    example
)


plot_competing_risk(
    res,
    example
)


plot_bc_spatial_distribution(
    res,
    snapshot_times=(0.5, 1.0, 1.5, 2.0),
    title=example
)


plot_defender_flux(
    res,
    title=example
)







