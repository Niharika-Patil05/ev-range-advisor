"""Split protocols as first-class objects, one per level of the distribution-shift ladder.

Each protocol knows its own name, its level, and what claim it licenses. Making them
objects rather than loose calls to `GroupKFold` means an experiment records *which*
protocol produced a number, and a result can never be quoted at a level it was not
tested at (docs/RESEARCH_PLAN_V2.md section 4.1).

    L0  RandomRows          leaky control -- included ON PURPOSE, to measure what a
                            naive split fabricates (experiment E0)
    L1  RouteWise           unseen routes            -- current standard practice
    L2  TemperatureRegime   unseen temperature band  -- train warm, test cold
    L3  LeaveOneVehicleOut  unseen vehicle
    L4  CrossVehicleModel   unseen vehicle model     -- BEV -> PHEV-electric transfer
        LeaveOneBatteryOut  unseen cell              -- the battery-side analogue
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

Fold = tuple[np.ndarray, np.ndarray]


class SplitProtocol(ABC):
    """A named evaluation protocol. `split` yields positional index arrays.

    `disjoint_keys` declares what the protocol GUARANTEES to hold apart, so an
    experiment can assert exactly that and nothing more. This matters: a
    leave-one-vehicle-out split does not automatically separate routes, and
    asserting route disjointness there would be asserting something the protocol
    never promised.
    """

    level: str = "??"
    name: str = "unnamed"
    disjoint_keys: tuple[str, ...] = ("route_id",)

    @abstractmethod
    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        ...

    def describe(self) -> str:
        return f"{self.level} {self.name}"

    @staticmethod
    def _positions(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.asarray(mask))


class RandomRows(SplitProtocol):
    """L0 -- shuffled row split. LEAKY BY DESIGN.

    A route's other trips sit in the training set, so the model can memorise route
    geometry. This exists only so E0 can quantify how much accuracy that fabricates.
    Never use it for a reported result.
    """

    level, name = "L0", "random rows (leaky control)"
    disjoint_keys = ()          # guarantees nothing; that is the point

    def __init__(self, test_size: float = 0.25, seed: int = 0, n_splits: int = 1):
        self.test_size, self.seed, self.n_splits = test_size, seed, n_splits

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        rng = np.random.default_rng(self.seed)
        n = len(df)
        for _ in range(self.n_splits):
            perm = rng.permutation(n)
            cut = int(round(n * self.test_size))
            yield perm[cut:], perm[:cut]


class RouteWise(SplitProtocol):
    """L1 -- no route appears in both train and test."""

    level, name = "L1", "unseen routes"

    def __init__(self, group_col: str = "route_id", n_splits: int = 5,
                 test_size: float = 0.25, seed: int = 0, mode: str = "kfold"):
        self.group_col, self.n_splits = group_col, n_splits
        self.test_size, self.seed, self.mode = test_size, seed, mode

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        groups = df[self.group_col].to_numpy()
        if self.mode == "kfold":
            yield from GroupKFold(self.n_splits).split(df, groups=groups)
        else:
            yield from GroupShuffleSplit(
                n_splits=self.n_splits, test_size=self.test_size, random_state=self.seed
            ).split(df, groups=groups)


class TemperatureRegime(SplitProtocol):
    """L2 -- train on warm segments, test on the coldest tertile.

    Design note worth defending in the viva: a naive temperature split leaks routes.
    A commute is driven year-round, so the same route appears in both warm and cold
    weather. A model could then score well by recognising the route rather than by
    handling the cold, confounding temperature shift with route memorisation.

    Filtering on temperature first does not fix this: because nearly every route has
    *some* cold segments, every route would enter the test set and no training data
    would remain. So the partition is done in two stages, routes first:

        1. partition ROUTES into train-routes and test-routes (as in L1);
        2. test  = cold segments belonging to test-routes;
           train = warm segments belonging to train-routes.

    The result is a genuine temperature shift evaluated on genuinely unseen routes.
    It discards the warm segments of test-routes and the cold segments of
    train-routes; `last_attrition_` records exactly how much, and experiments
    report it rather than quietly losing the data.
    """

    level, name = "L2", "unseen temperature regime (train warm, test cold)"

    def __init__(self, temp_col: str = "temp_c", group_col: str = "route_id",
                 cold_quantile: float = 1 / 3, route_test_fraction: float = 0.3,
                 seed: int = 0, enforce_route_disjoint: bool = True):
        self.temp_col, self.group_col = temp_col, group_col
        self.cold_quantile = cold_quantile
        self.route_test_fraction = route_test_fraction
        self.seed = seed
        self.enforce_route_disjoint = enforce_route_disjoint
        self.last_attrition_: dict[str, float] = {}

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        temp = df[self.temp_col].to_numpy(float)
        threshold = float(np.quantile(temp, self.cold_quantile))
        cold = temp <= threshold

        if self.enforce_route_disjoint:
            routes = pd.unique(df[self.group_col])
            order = np.random.default_rng(self.seed).permutation(len(routes))
            n_test = max(1, min(len(routes) - 1, round(len(routes) * self.route_test_fraction)))
            test_routes = set(routes[order[:n_test]])
            in_test_route = df[self.group_col].isin(test_routes).to_numpy()
            test_mask = cold & in_test_route
            train_mask = ~cold & ~in_test_route
        else:
            test_mask, train_mask = cold, ~cold

        train_idx = self._positions(df, train_mask)
        test_idx = self._positions(df, test_mask)

        self.last_attrition_ = {
            "cold_threshold_c": threshold,
            "n_cold_total": int(cold.sum()),
            "n_warm_total": int((~cold).sum()),
            "n_train_kept": int(len(train_idx)),
            "n_test_kept": int(len(test_idx)),
            "n_discarded": int(len(df) - len(train_idx) - len(test_idx)),
        }
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                f"TemperatureRegime produced an empty split ({self.last_attrition_}). "
                f"Too few routes, or no route has segments in both temperature bands."
            )
        yield train_idx, test_idx

    def describe(self) -> str:
        return (f"{self.level} {self.name}; cold_quantile={self.cold_quantile}, "
                f"route-disjoint={self.enforce_route_disjoint}")


class LeaveOneGroupOut(SplitProtocol):
    """Generic leave-one-entity-out; the base for vehicle and battery protocols."""

    level, name = "L?", "leave-one-group-out"

    def __init__(self, group_col: str):
        self.group_col = group_col

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        groups = df[self.group_col].to_numpy()
        for held in pd.unique(groups):
            test = groups == held
            yield self._positions(df, ~test), self._positions(df, test)


class LeaveOneVehicleOut(LeaveOneGroupOut):
    """L3 -- unseen vehicle.

    With only three BEVs in VED this rests on n=3. That is weak, and every result
    produced under it must print n alongside the number.
    """

    level, name = "L3", "unseen vehicle (leave-one-vehicle-out)"
    disjoint_keys = ("vehicle_id",)     # NOT route_id -- see the class docstring

    def __init__(self, group_col: str = "vehicle_id"):
        super().__init__(group_col)


class LeaveOneVehicleOutRouteDisjoint(SplitProtocol):
    """L3b -- unseen vehicle AND unseen routes.

    Plain L3 holds out a vehicle but not its roads. On this data 16-30% of each
    L3 test set sits on a route that also appears in training, so L3 measures
    vehicle transfer with some route familiarity retained. That is a legitimate
    thing to measure -- it isolates the vehicle effect while holding the road
    roughly constant -- but it is NOT "unseen vehicle on unseen ground", and
    reporting it as such would overstate generalization.

    L3b additionally drops from training every route the held-out vehicle drives.
    It is the harder and cleaner claim, at the cost of training data. Both are
    reported, because they answer different questions.
    """

    level, name = "L3b", "unseen vehicle AND unseen routes"
    disjoint_keys = ("vehicle_id", "route_id")

    def __init__(self, vehicle_col: str = "vehicle_id", route_col: str = "route_id"):
        self.vehicle_col, self.route_col = vehicle_col, route_col
        self.last_attrition_: dict[str, float] = {}

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        veh = df[self.vehicle_col].to_numpy()
        for held in pd.unique(veh):
            test_mask = veh == held
            test_routes = set(df.loc[test_mask, self.route_col].unique())
            train_mask = (~test_mask) & ~df[self.route_col].isin(test_routes).to_numpy()
            train_idx = self._positions(df, train_mask)
            test_idx = self._positions(df, test_mask)
            self.last_attrition_ = {
                "held_out_vehicle": held,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_dropped_for_route_disjointness": int((~test_mask).sum() - len(train_idx)),
            }
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx


class LeaveOneBatteryOut(LeaveOneGroupOut):
    """Battery-side protocol: a cell is predicted by a model that never saw it."""

    level, name = "LOBO", "unseen battery cell"

    def __init__(self, group_col: str = "battery_id"):
        super().__init__(group_col)


class CrossVehicleModel(SplitProtocol):
    """L4 -- fit on one set of vehicle models, test on another.

    Used for the BEV -> PHEV-electric-mode transfer that replaced the dropped
    cross-vehicle-class experiment (docs/PHASE0_VERIFICATION.md section A6).
    Vehicle parameters are re-identified per model; they are never shared.
    """

    level, name = "L4", "unseen vehicle model (cross-model transfer)"

    def __init__(self, model_col: str = "vehicle_model",
                 train_models: list[str] | None = None,
                 test_models: list[str] | None = None):
        self.model_col = model_col
        self.train_models, self.test_models = train_models, test_models

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        col = df[self.model_col]
        if self.train_models is None or self.test_models is None:
            raise ValueError("CrossVehicleModel needs explicit train_models and test_models.")
        overlap = set(self.train_models) & set(self.test_models)
        if overlap:
            raise ValueError(f"train and test vehicle models overlap: {sorted(overlap)}")
        train = self._positions(df, col.isin(self.train_models).to_numpy())
        test = self._positions(df, col.isin(self.test_models).to_numpy())
        if len(train) == 0 or len(test) == 0:
            raise ValueError("CrossVehicleModel produced an empty split.")
        yield train, test

    def describe(self) -> str:
        return (f"{self.level} {self.name}; train={self.train_models} -> test={self.test_models}")


LADDER: dict[str, type[SplitProtocol]] = {
    "L0": RandomRows,
    "L1": RouteWise,
    "L2": TemperatureRegime,
    "L3": LeaveOneVehicleOut,
    "L3b": LeaveOneVehicleOutRouteDisjoint,
    "L4": CrossVehicleModel,
}
