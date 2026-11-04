import pandas as pd

SPLITS = [
    "team_coverage_man_zone",
    "team_coverage_type",
    "in_motion_at_ball_snap",
    "route_ran",
]


def _pass_depth(pass_length: pd.Series) -> pd.Series:
    """short: < 15, medium: 15-30 (inclusive), deep: > 30. NaN stays NaN."""
    depth = pd.Series(pd.NA, index=pass_length.index, dtype="object")
    depth[pass_length < 15] = "short"
    depth[(pass_length >= 15) & (pass_length <= 30)] = "medium"
    depth[pass_length > 30] = "deep"
    return depth


def catch_rate(plays_df: pd.DataFrame) -> pd.DataFrame:
    targeted = (
        plays_df[plays_df['target'] == True]
        .assign(
            catch=lambda d: d['pass_result'] == 'C',
            pass_depth=lambda d: _pass_depth(d['pass_length']),
        )
    )

    def _agg(keys):
        g = targeted.groupby(keys).agg(
            catches=('catch', 'sum'),
            targets=('target', 'sum'),
        )
        g["catch_rate"] = g["catches"] / g["targets"]
        return g

    catch_rate = _agg('nfl_id')

    # e.g. catch_rate_short, targets_deep
    by_depth = _agg(['nfl_id', 'pass_depth'])[['catch_rate', 'targets']].unstack('pass_depth')
    by_depth.columns = [f"{stat}_{depth}" for stat, depth in by_depth.columns]
    by_depth = by_depth.reindex(
        columns=[f"{s}_{d}" for s in ('catch_rate', 'targets') for d in ('short', 'medium', 'deep')]
    )

    return catch_rate.join(by_depth, how='left')


def average_depth_of_target(plays_df: pd.DataFrame) -> pd.DataFrame:
    """aDOT: mean pass_length over the player's targets (targets without a pass_length are skipped)."""
    adot = (
        plays_df[plays_df['target'] == True]
        .groupby('nfl_id')
        .agg(
            air_yards=('pass_length', 'sum'),
            targets_with_depth=('pass_length', 'count'),
        )
    )
    adot["adot"] = adot["air_yards"] / adot["targets_with_depth"]
    return adot


def average_separation(plays_df: pd.DataFrame) -> pd.DataFrame:
    """Mean separation_at_pass_forward over the player's targets (targets without a value are skipped)."""
    separation = (
        plays_df[plays_df['target'] == True]
        .groupby('nfl_id')
        .agg(
            avg_separation=('separation_at_pass_forward', 'mean'),
            targets_with_separation=('separation_at_pass_forward', 'count'),
        )
    )
    return separation


def yac_per_reception(plays_df: pd.DataFrame) -> pd.DataFrame:
    yac_per_reception = (
        plays_df
        .assign(catch=lambda d: (d["target"] == True) & (d["pass_result"] == "C"))
        .query("catch == True")
        .groupby('nfl_id')
        .agg (
            yac=('yards_after_catch', 'sum'),
            receptions=('catch', 'sum')
        )
    )
    yac_per_reception["yac_per_reception"] = yac_per_reception["yac"] / yac_per_reception["receptions"]
    return yac_per_reception


def epa_per_target(plays_df: pd.DataFrame) -> pd.DataFrame:
    epa_per_target = (
        plays_df[plays_df['target'] == True]
        .groupby('nfl_id')
        .agg (
            epa=('expected_points_added', 'sum'),
            targets=('target', 'sum')
        )
    )
    epa_per_target["epa_per_target"] = epa_per_target["epa"] / epa_per_target["targets"]
    return epa_per_target


def route_distribution(plays_df: pd.DataFrame) -> pd.DataFrame:
    route_ratio = (
        plays_df.groupby("nfl_id")["route_ran"]
        .value_counts(normalize=True)
        .rename("route_ratio")
        .to_frame()
    )
    route_ratio_wide = (
        route_ratio
        .reset_index()
        .pivot_table(
            index="nfl_id",
            columns="route_ran",
            values="route_ratio",
            fill_value=0 )
        .add_prefix("route_ratio_")
        .rename_axis(columns=None)
        .reset_index()
    )
    return route_ratio_wide


def _per_route_run(
    plays_df: pd.DataFrame,
    value_col: str,
    metric_name: str,
    splits: list[str] | None = None,
) -> pd.DataFrame:
    """sum(value_col) / routes run, globally and for each value of each split column.

    The denominator counts every route the player ran (non-null `route_ran`),
    targeted or not. Splits where the player ran no routes stay NaN.
    """
    splits = SPLITS if splits is None else splits

    def _agg(keys):
        g = plays_df.groupby(keys).agg(
            total=(value_col, "sum"),
            routes_run=("route_ran", "count"),
        )
        g[metric_name] = g["total"] / g["routes_run"]
        return g[[metric_name, "routes_run"]]

    result = _agg("nfl_id")

    for col in splits:
        wide = _agg(["nfl_id", col]).unstack(col)
        # e.g. yprr_team_coverage_man_zone_MAN_COVERAGE, routes_run_route_ran_GO
        wide.columns = [f"{stat}_{col}_{val}" for stat, val in wide.columns]
        result = result.join(wide, how="left")

    return result


def yards_per_route_run(
    plays_df: pd.DataFrame, splits: list[str] | None = None
) -> pd.DataFrame:
    return _per_route_run(plays_df, "rec_yards", "yprr", splits)


def targets_per_route_run(
    plays_df: pd.DataFrame, splits: list[str] | None = None
) -> pd.DataFrame:
    return _per_route_run(plays_df, "target", "tprr", splits)
