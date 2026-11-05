"""Helpers for the WR usage-profile analysis (notebooks/UsageProfile.ipynb).

Pipeline: route counts per player-season -> minimum-sample floor -> shrinkage
(Dirichlet for the route mix, beta for slot%, normal for aDOT) -> CLR + PCA for
the route mix -> weighted feature matrix -> KMeans.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.decomposition import PCA

# WHEEL and ANGLE are too rare among WRs to be their own dimension; they are
# short releases toward the flat, so they are pooled with FLAT.
ROUTE_COLLAPSE = {"WHEEL": "FLAT_WHEEL", "ANGLE": "FLAT_WHEEL", "FLAT": "FLAT_WHEEL"}


def collapse_routes(route_ran: pd.Series, mapping: dict = ROUTE_COLLAPSE) -> pd.Series:
    return route_ran.replace(mapping)


def route_counts(plays_df: pd.DataFrame, by=("nfl_id", "season"), route_col="route") -> pd.DataFrame:
    """Wide table of route counts: one row per `by` group, one column per route."""
    routes = plays_df[plays_df[route_col].notna()]
    counts = routes.groupby(list(by))[route_col].value_counts().unstack(fill_value=0)
    counts.columns.name = None
    return counts.sort_index(axis=1)


def _shares(counts: pd.DataFrame) -> pd.DataFrame:
    return counts.div(counts.sum(axis=1), axis=0)


def choose_route_floor(
    counts: pd.DataFrame,
    n_grid=(25, 50, 75, 100, 125, 150, 200, 250, 300),
    min_source_routes: int = 400,
    tol: float = 0.10,
    n_boot: int = 200,
    seed: int = 0,
) -> tuple[int, pd.DataFrame]:
    """Bootstrap the noise of a route distribution measured on n routes.

    For every high-volume player-season (>= min_source_routes) we treat its full
    route mix as the "truth", draw n routes from it, and measure the total
    variation distance (TVD) between the sampled and true mix. The floor is the
    smallest n whose median TVD is <= tol.
    """
    rng = np.random.default_rng(seed)
    source = _shares(counts[counts.sum(axis=1) >= min_source_routes]).to_numpy()
    rows = []
    for n in n_grid:
        tvd = []
        for p in source:
            sample = rng.multinomial(n, p, size=n_boot) / n
            tvd.append(0.5 * np.abs(sample - p).sum(axis=1))
        tvd = np.concatenate(tvd)
        rows.append({"n_routes": n, "median_tvd": np.median(tvd), "p90_tvd": np.quantile(tvd, 0.9)})
    curve = pd.DataFrame(rows)
    passing = curve.loc[curve["median_tvd"] <= tol, "n_routes"]
    floor = int(passing.iloc[0]) if len(passing) else int(n_grid[-1])
    return floor, curve


def estimate_dirichlet_alpha(counts: pd.DataFrame) -> tuple[float, pd.Series]:
    """Method-of-moments estimate of the Dirichlet-multinomial prior strength.

    Under a Dirichlet(alpha * pi) prior on each row's true mix, the observed
    shares satisfy  E[sum_j (p_ij - pi_j)^2] = S * (rho + (1 - rho) / n_i),
    with S = sum_j pi_j (1 - pi_j) and rho = 1 / (1 + alpha).
    alpha acts as "pseudo-routes": the sample size at which a row's own mix and
    the league mix get equal weight. Works for 2 columns too (beta-binomial).
    """
    n = counts.sum(axis=1).to_numpy(dtype=float)
    prior = counts.sum(axis=0) / counts.to_numpy().sum()
    p = _shares(counts).to_numpy()
    pi = prior.to_numpy()

    S = (pi * (1 - pi)).sum()
    D = ((p - pi) ** 2).sum(axis=1).mean()
    inv_n = (1 / n).mean()
    rho = (D / S - inv_n) / (1 - inv_n)
    rho = float(np.clip(rho, 1e-6, 1 - 1e-6))
    return 1 / rho - 1, prior


def shrink_composition(counts: pd.DataFrame, alpha: float, prior: pd.Series) -> pd.DataFrame:
    """(count + alpha * league_share) / (n + alpha), row-wise. Rows still sum to 1."""
    prior = prior.reindex(counts.columns)
    shrunk = counts.add(alpha * prior, axis=1)
    return shrunk.div(shrunk.sum(axis=1), axis=0)


def clr(shares: pd.DataFrame) -> pd.DataFrame:
    """Centered log-ratio: log(share) - mean(log(share)) per row. Requires no zeros."""
    logs = np.log(shares)
    return logs.sub(logs.mean(axis=1), axis=0)


def inverse_clr(clr_values: np.ndarray) -> np.ndarray:
    e = np.exp(clr_values - clr_values.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def stabilization_k(values: pd.Series, groups: pd.Series) -> float:
    """k = sigma^2_within / tau^2_between for a per-observation metric (e.g. pass_length per target).

    A group's mean based on n observations has reliability n / (n + k).
    """
    d = pd.DataFrame({"v": values, "g": groups}).dropna()
    g = d.groupby("g")["v"].agg(["mean", "var", "count"])
    g = g[g["count"] > 1]
    w = g["count"]
    sigma2_within = (g["var"] * (w - 1)).sum() / (w - 1).sum()
    grand = np.average(g["mean"], weights=w)
    observed_between = np.average((g["mean"] - grand) ** 2, weights=w)
    noise = np.average(sigma2_within / w, weights=w)
    tau2_between = max(observed_between - noise, 1e-9)
    return sigma2_within / tau2_between


def shrink_mean(mean: pd.Series, n: pd.Series, k: float, prior: float) -> pd.Series:
    """(n * mean + k * prior) / (n + k); groups with n = 0 get the prior."""
    n = n.fillna(0)
    return (mean.fillna(prior) * n + prior * k) / (n + k)


class UsageFeatures:
    """Builds the clustering matrix and maps points back to readable units.

    Blocks:
      - route mix: CLR(shrunk shares) -> PCA (keeps `pca_variance` of variance),
        rescaled so the block's total variance equals `route_weight`
      - logit(slot_pct), z-scored (variance 1)
      - aDOT, z-scored (variance 1)
    """

    def __init__(self, route_weight: float = 2.0, pca_variance: float = 0.90, seed: int = 0):
        self.route_weight = route_weight
        self.pca_variance = pca_variance
        self.seed = seed

    def fit_transform(self, shares: pd.DataFrame, slot_pct: pd.Series, adot: pd.Series) -> pd.DataFrame:
        self.routes = list(shares.columns)
        self.pca = PCA(n_components=self.pca_variance, svd_solver="full", random_state=self.seed)
        scores = self.pca.fit_transform(clr(shares).to_numpy())
        self.route_scale = np.sqrt(self.route_weight / scores.var(axis=0).sum())

        slot_logit = logit(slot_pct.to_numpy())
        self.slot_mu, self.slot_sd = slot_logit.mean(), slot_logit.std()
        self.adot_mu, self.adot_sd = adot.mean(), adot.std(ddof=0)
        return self.transform(shares, slot_pct, adot)

    def transform(self, shares, slot_pct, adot) -> pd.DataFrame:
        scores = self.pca.transform(clr(shares[self.routes]).to_numpy()) * self.route_scale
        X = np.column_stack([
            scores,
            (logit(slot_pct.to_numpy()) - self.slot_mu) / self.slot_sd,
            (adot.to_numpy() - self.adot_mu) / self.adot_sd,
        ])
        cols = [f"route_pc{i + 1}" for i in range(scores.shape[1])] + ["slot_z", "adot_z"]
        return pd.DataFrame(X, index=shares.index, columns=cols)

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        """Feature-space points (e.g. KMeans centroids) -> route shares, slot_pct, adot."""
        X = np.asarray(X)
        n_pc = self.pca.n_components_
        clr_values = self.pca.inverse_transform(X[:, :n_pc] / self.route_scale)
        out = pd.DataFrame(inverse_clr(clr_values), columns=self.routes)
        out["slot_pct"] = expit(X[:, n_pc] * self.slot_sd + self.slot_mu)
        out["adot"] = X[:, n_pc + 1] * self.adot_sd + self.adot_mu
        return out


def nearest_to_centroids(X: pd.DataFrame, labels: np.ndarray, centers: np.ndarray, n: int = 3) -> pd.DataFrame:
    """The n members of each cluster closest (Euclidean, feature space) to their centroid."""
    dist = np.linalg.norm(X.to_numpy() - centers[labels], axis=1)
    d = pd.DataFrame({"cluster": labels, "distance": dist}, index=X.index)
    return d.sort_values("distance").groupby("cluster", group_keys=False).head(n).sort_values(["cluster", "distance"])
