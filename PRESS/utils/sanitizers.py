import re
import pandas as pd
import jax.numpy as jnp


def validate_tracking_data(
    tracking_data: pd.DataFrame,
) -> None:
    """
    Validate canonical wide tracking format.

    The function does not modify data.
    It raises ValueError if the format is invalid.
    """
    
    if not isinstance(
        tracking_data,
        pd.DataFrame,
    ):
        raise ValueError(
            "tracking_data must be a pandas DataFrame."
        )    

    required = [
        "frame",
        "player_possession",
        "ball_x",
        "ball_y",
    ]

    missing = [
        col for col in required
        if col not in tracking_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Frame integrity
    if tracking_data["frame"].duplicated().any():
        raise ValueError(
            "Frame column contains duplicates."
        )

    # Player columns
    pattern = re.compile(
        r"^(home|away)_(\d+)_(x|y|vx|vy|v)$"
    )

    players = {}

    for col in tracking_data.columns:

        match = pattern.match(col)

        if match:
            player = (
                f"{match.group(1)}_"
                f"{match.group(2)}"
            )

            variable = match.group(3)

            players.setdefault(
                player,
                set()
            ).add(variable)

    expected = {
        "x",
        "y",
        "vx",
        "vy",
        "v",
    }

    for player, variables in players.items():

        missing = expected - variables

        if missing:
            raise ValueError(
                f"{player} missing columns: {missing}"
            )

    if len(players) == 0:
        raise ValueError(
            "No player columns found."
        )

    # Possession consistency
    valid_players = set(players.keys())

    invalid_possession = (
        tracking_data["player_possession"]
        .dropna()
        .loc[
            lambda x:
            ~x.isin(valid_players)
        ]
    )

    if not invalid_possession.empty:
        raise ValueError(
            "player_possession contains "
            "unknown players."
        )

    # Numeric check
    numeric_cols = [
        c for c in tracking_data.columns
        if (
            c.endswith("_x")
            or c.endswith("_y")
            or c.endswith("_vx")
            or c.endswith("_vy")
            or c.endswith("_v")
        )
    ]

    non_numeric = [
        c for c in numeric_cols
        if not pd.api.types.is_numeric_dtype(
            tracking_data[c]
        )
    ]

    if non_numeric:
        raise ValueError(
            f"Non numeric tracking columns: {non_numeric}"
        )
        

def validate_events_data(
    events_data,
):
    """
    Validate events data according to
    PRESS conventions.
    """

    # ----------------------------------
    # DataFrame
    # ----------------------------------

    if not isinstance(
        events_data,
        pd.DataFrame,
    ):
        raise ValueError(
            "events_data must be a pandas DataFrame."
        )
    
    # ----------------------------------
    # Required columns
    # ----------------------------------
        
    if "frame" not in events_data.columns:

        raise ValueError(
            "Missing required column: frame."
        )

    # ----------------------------------
    # Player / team convention
    # ----------------------------------

    player_ids = (
        events_data["player_id"]
        .astype(str)
    )

    if not player_ids.str.match(
        r"^(home|away)_\d+$"
    ).all():

        raise ValueError(
            "Invalid player_id convention. "
            "Expected home_<shirt_number> "
            "or away_<shirt_number>."
        )

    if not events_data["team_id"].isin(
        ["home", "away"]
    ).all():

        raise ValueError(
            "Invalid team_id convention. "
            "Expected 'home' or 'away'."
        )

    player_teams = (
        player_ids
        .str.split("_")
        .str[0]
    )

    if not player_teams.eq(
        events_data["team_id"]
    ).all():

        raise ValueError(
            "player_id and team_id are inconsistent."
        )

    # ----------------------------------
    # Pass
    # ----------------------------------

    passes = events_data[
        events_data["event_type"] == "pass"
    ]

    if not passes.empty:

        if passes["is_successful"].isna().any():

            raise ValueError(
                "Pass events require "
                "is_successful."
            )

        if not passes["is_successful"].map(
            lambda x:
                isinstance(
                    x,
                    (bool, jnp.bool_),
                )
        ).all():

            raise ValueError(
                "Pass events require boolean "
                "is_successful."
            )

        successful_passes = passes[
            passes["is_successful"]
        ]

        if not successful_passes.empty:

            if (
                "receiver_player_id"
                not in events_data.columns
            ):
                raise ValueError(
                    "Successful pass events require "
                    "receiver_player_id."
                )

            receivers = (
                successful_passes[
                    "receiver_player_id"
                ]
            )

            if receivers.isna().any():

                raise ValueError(
                    "Successful pass events require "
                    "receiver_player_id."
                )

            if not receivers.astype(str).str.match(
                r"^(home|away)_\d+$"
            ).all():

                raise ValueError(
                    "Invalid receiver_player_id "
                    "convention."
                )

    return True
  


def validate_kappa_surface(
    kappa_surface,
):
    """
    Validate empirical kappa(v,d) surface.

    Expected structure:

    {
        "speed_edges": jnp.ndarray,
        "distance_edges": jnp.ndarray,
        "kappa": jnp.ndarray,
        "counts": jnp.ndarray (optional)
    }

    kappa shape:
        (len(speed_edges)-1,
         len(distance_edges)-1)
    """

    if not isinstance(
        kappa_surface,
        dict,
    ):
        raise ValueError(
            "kappa_surface must be a dictionary."
        )

    required_keys = [
        "speed_edges",
        "distance_edges",
        "kappa",
    ]

    for key in required_keys:

        if key not in kappa_surface:

            raise ValueError(
                f"Missing key '{key}' in kappa_surface."
            )

    speed_edges = jnp.asarray(
        kappa_surface["speed_edges"]
    )

    distance_edges = jnp.asarray(
        kappa_surface["distance_edges"]
    )

    K = jnp.asarray(
        kappa_surface["kappa"]
    )

    # ----------------------------------
    # Edge validation
    # ----------------------------------

    for name, edges in [
        ("speed_edges", speed_edges),
        ("distance_edges", distance_edges),
    ]:

        if edges.ndim != 1:

            raise ValueError(
                f"{name} must be a 1D array."
            )

        if not jnp.issubdtype(
            edges.dtype,
            jnp.number,
        ):

            raise ValueError(
                f"{name} must be numeric."
            )

        if not jnp.isfinite(
            edges
        ).all():

            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        if not jnp.diff(
            edges
        ).__gt__(0).all():

            raise ValueError(
                f"{name} must be strictly increasing."
            )

    # ----------------------------------
    # Kappa validation
    # ----------------------------------

    expected_shape = (
        len(speed_edges) - 1,
        len(distance_edges) - 1,
    )

    if K.shape != expected_shape:

        raise ValueError(
            f"kappa shape {K.shape} does not match "
            f"expected {expected_shape}."
        )

    if not jnp.issubdtype(
        K.dtype,
        jnp.number,
    ):

        raise ValueError(
            "kappa must be numeric."
        )

    if not jnp.isfinite(
        K
    ).all():

        raise ValueError(
            "kappa contains NaN or infinite values."
        )

    if jnp.any(
        K < 0
    ):

        raise ValueError(
            "kappa values must be non-negative."
        )

    return True
  
  
