from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import random
import re
from typing import Callable, Sequence

import numpy as np
import pandas as pd


DEFAULT_WINDOW_START = 100
DEFAULT_WINDOW_HOURS = 24
DEFAULT_SEARCH_BUDGET = 80
DEFAULT_TRIALS = 5
DEFAULT_SEED = 7
DEFAULT_BEAM_WIDTH = 4


@dataclass(frozen=True)
class Job:
    run_time: int
    gpu_num: float


@dataclass(frozen=True)
class Candidate:
    region: str
    start_sec: int


@dataclass(frozen=True)
class SearchResult:
    cost: float
    candidate: Candidate | None


@dataclass
class SignalSeries:
    values: np.ndarray
    prefix: np.ndarray

    @classmethod
    def from_hourly(cls, hourly_signal: Sequence[float]) -> "SignalSeries":
        values = np.repeat(np.asarray(hourly_signal, dtype=float), 3600)
        prefix = np.zeros(len(values) + 1, dtype=float)
        prefix[1:] = np.cumsum(values)
        return cls(values=values, prefix=prefix)

    def cost(self, start_sec: int, run_time: int, gpu_num: float) -> float:
        end_sec = start_sec + run_time
        if start_sec < 0 or end_sec > len(self.values):
            return math.inf
        total_intensity = self.prefix[end_sec] - self.prefix[start_sec]
        return (total_intensity * gpu_num) / 3600.0


class SchedulingProblem:
    def __init__(
        self,
        signals: dict[str, SignalSeries],
        *,
        time_move_radius: int = 20_000,
        switch_probability: float = 0.35,
    ) -> None:
        if not signals:
            raise ValueError("At least one region signal is required.")
        self.signals = signals
        self.regions = tuple(signals.keys())
        self.time_move_radius = time_move_radius
        self.switch_probability = switch_probability

    def feasible_regions(self, job: Job) -> list[str]:
        return [region for region in self.regions if len(self.signals[region].values) >= job.run_time]

    def latest_start(self, region: str, run_time: int) -> int:
        return len(self.signals[region].values) - run_time

    def clamp_start(self, region: str, start_sec: int, run_time: int) -> int:
        return max(0, min(self.latest_start(region, run_time), start_sec))

    def random_candidate(self, job: Job, rng: random.Random) -> Candidate | None:
        regions = self.feasible_regions(job)
        if not regions:
            return None
        region = rng.choice(regions)
        start_sec = rng.randint(0, self.latest_start(region, job.run_time))
        return Candidate(region=region, start_sec=start_sec)

    def neighbor(self, job: Job, current: Candidate, rng: random.Random) -> Candidate:
        region = current.region
        if len(self.regions) > 1 and rng.random() < self.switch_probability:
            alternatives = [name for name in self.feasible_regions(job) if name != current.region]
            if alternatives:
                region = rng.choice(alternatives)
        move = rng.randint(-self.time_move_radius, self.time_move_radius)
        start_sec = self.clamp_start(region, current.start_sec + move, job.run_time)
        return Candidate(region=region, start_sec=start_sec)

    def crossover(self, job: Job, left: Candidate, right: Candidate, rng: random.Random) -> Candidate:
        region = rng.choice((left.region, right.region))
        if rng.random() < 0.5:
            start_sec = left.start_sec
        else:
            start_sec = right.start_sec
        if rng.random() < 0.5:
            start_sec = (left.start_sec + right.start_sec) // 2
        start_sec = self.clamp_start(region, start_sec, job.run_time)
        return Candidate(region=region, start_sec=start_sec)

    def mutate(self, job: Job, candidate: Candidate, rng: random.Random) -> Candidate:
        return self.neighbor(job, candidate, rng)

    def cost(self, job: Job, candidate: Candidate | None) -> float:
        if candidate is None:
            return math.inf
        return self.signals[candidate.region].cost(candidate.start_sec, job.run_time, job.gpu_num)


Solver = Callable[[SchedulingProblem, Job, random.Random, int], SearchResult]


def _best_random_candidate(problem: SchedulingProblem, job: Job, rng: random.Random, budget: int) -> SearchResult:
    best_candidate = None
    best_cost = math.inf
    for _ in range(max(1, budget)):
        candidate = problem.random_candidate(job, rng)
        cost = problem.cost(job, candidate)
        if cost < best_cost:
            best_cost = cost
            best_candidate = candidate
    return SearchResult(cost=best_cost, candidate=best_candidate)


def random_baseline(problem: SchedulingProblem, job: Job, rng: random.Random, _: int) -> SearchResult:
    return _best_random_candidate(problem, job, rng, budget=1)


def hill_climbing(problem: SchedulingProblem, job: Job, rng: random.Random, budget: int) -> SearchResult:
    current = problem.random_candidate(job, rng)
    if current is None:
        return SearchResult(cost=math.inf, candidate=None)
    current_cost = problem.cost(job, current)
    best_cost = current_cost
    best_candidate = current

    for _ in range(max(0, budget - 1)):
        neighbor = problem.neighbor(job, current, rng)
        neighbor_cost = problem.cost(job, neighbor)
        if neighbor_cost < current_cost:
            current = neighbor
            current_cost = neighbor_cost
        if neighbor_cost < best_cost:
            best_cost = neighbor_cost
            best_candidate = neighbor

    return SearchResult(cost=best_cost, candidate=best_candidate)


def local_beam_search(problem: SchedulingProblem, job: Job, rng: random.Random, budget: int) -> SearchResult:
    beam_width = min(DEFAULT_BEAM_WIDTH, max(1, budget))
    initial_beam = []
    seen = set()
    attempts = 0
    max_attempts = max(beam_width * 6, 6)
    while len(initial_beam) < beam_width and attempts < max_attempts:
        attempts += 1
        candidate = problem.random_candidate(job, rng)
        if candidate is None:
            break
        if candidate in seen:
            continue
        seen.add(candidate)
        initial_beam.append((problem.cost(job, candidate), candidate))

    if not initial_beam:
        return SearchResult(cost=math.inf, candidate=None)

    evaluations = len(initial_beam)
    beam = sorted(initial_beam, key=lambda item: item[0])[:beam_width]
    best_cost, best_candidate = beam[0]

    while evaluations < budget:
        expansions = []
        for _, candidate in beam:
            if evaluations >= budget:
                break
            neighbor = problem.neighbor(job, candidate, rng)
            cost = problem.cost(job, neighbor)
            expansions.append((cost, neighbor))
            evaluations += 1

            if evaluations >= budget:
                break
            second_neighbor = problem.neighbor(job, candidate, rng)
            second_cost = problem.cost(job, second_neighbor)
            expansions.append((second_cost, second_neighbor))
            evaluations += 1

        combined = beam + expansions
        deduped: dict[Candidate, float] = {}
        for cost, candidate in combined:
            previous = deduped.get(candidate)
            if previous is None or cost < previous:
                deduped[candidate] = cost
        beam = sorted(((cost, candidate) for candidate, cost in deduped.items()), key=lambda item: item[0])[:beam_width]
        if beam and beam[0][0] < best_cost:
            best_cost, best_candidate = beam[0]

    return SearchResult(cost=best_cost, candidate=best_candidate)


def simulated_annealing(problem: SchedulingProblem, job: Job, rng: random.Random, budget: int) -> SearchResult:
    current = problem.random_candidate(job, rng)
    if current is None:
        return SearchResult(cost=math.inf, candidate=None)
    current_cost = problem.cost(job, current)
    best_cost = current_cost
    best_candidate = current

    initial_temp = max(current_cost * 0.05, 1.0)
    final_temp = 1e-3
    steps = max(1, budget - 1)
    cooling = (final_temp / initial_temp) ** (1.0 / steps)
    temperature = initial_temp

    for _ in range(steps):
        neighbor = problem.neighbor(job, current, rng)
        neighbor_cost = problem.cost(job, neighbor)
        delta = neighbor_cost - current_cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 1e-9)):
            current = neighbor
            current_cost = neighbor_cost
        if current_cost < best_cost:
            best_cost = current_cost
            best_candidate = current
        temperature *= cooling

    return SearchResult(cost=best_cost, candidate=best_candidate)


def tabu_search(problem: SchedulingProblem, job: Job, rng: random.Random, budget: int) -> SearchResult:
    current = problem.random_candidate(job, rng)
    if current is None:
        return SearchResult(cost=math.inf, candidate=None)
    current_cost = problem.cost(job, current)
    best_cost = current_cost
    best_candidate = current

    tabu_limit = 12
    tabu_list = {current}
    tabu_queue = [current]
    evaluations = 1

    while evaluations < budget:
        remaining = budget - evaluations
        candidate_count = min(8, remaining)
        neighborhood = []
        for _ in range(candidate_count):
            neighbor = problem.neighbor(job, current, rng)
            neighborhood.append((problem.cost(job, neighbor), neighbor))
        evaluations += candidate_count

        allowed = [item for item in neighborhood if item[1] not in tabu_list or item[0] < best_cost]
        if not allowed:
            allowed = neighborhood

        current_cost, current = min(allowed, key=lambda item: item[0])
        if current_cost < best_cost:
            best_cost = current_cost
            best_candidate = current

        tabu_queue.append(current)
        tabu_list.add(current)
        if len(tabu_queue) > tabu_limit:
            expired = tabu_queue.pop(0)
            tabu_list.discard(expired)

    return SearchResult(cost=best_cost, candidate=best_candidate)


def genetic_algorithm(problem: SchedulingProblem, job: Job, rng: random.Random, budget: int) -> SearchResult:
    population_size = min(12, max(4, budget))
    population = [problem.random_candidate(job, rng) for _ in range(population_size)]
    population = [candidate for candidate in population if candidate is not None]
    if not population:
        return SearchResult(cost=math.inf, candidate=None)
    scored = [(problem.cost(job, candidate), candidate) for candidate in population]
    evaluations = len(scored)
    best_cost, best_candidate = min(scored, key=lambda item: item[0])

    def tournament_pick() -> Candidate:
        contenders = rng.sample(scored, k=min(3, len(scored)))
        return min(contenders, key=lambda item: item[0])[1]

    while evaluations < budget:
        parent_a = tournament_pick()
        parent_b = tournament_pick()
        child = problem.crossover(job, parent_a, parent_b, rng)
        if rng.random() < 0.35:
            child = problem.mutate(job, child, rng)

        child_cost = problem.cost(job, child)
        evaluations += 1
        if child_cost < best_cost:
            best_cost = child_cost
            best_candidate = child

        elite = sorted(scored, key=lambda item: item[0])[:2]
        rest = sorted(scored, key=lambda item: item[0])[2:]
        if rest:
            rest[-1] = (child_cost, child)
            scored = elite + rest
        else:
            scored = elite + [(child_cost, child)]

    return SearchResult(cost=best_cost, candidate=best_candidate)


ALGORITHMS: tuple[tuple[str, Solver], ...] = (
    ("Random Baseline", random_baseline),
    ("Hill Climbing", hill_climbing),
    ("Local Beam Search", local_beam_search),
    ("Simulated Annealing", simulated_annealing),
    ("Tabu Search", tabu_search),
    ("Genetic Algorithm", genetic_algorithm),
)


def load_gpu_jobs(path: str = "gpu_jobs.csv") -> list[Job]:
    jobs = pd.read_csv(path, usecols=["run_time", "gpu_num"])
    filtered = jobs[(jobs["gpu_num"] > 0) & (jobs["run_time"] > 0)].copy()
    filtered["run_time"] = filtered["run_time"].astype(int)
    return [Job(run_time=row.run_time, gpu_num=float(row.gpu_num)) for row in filtered.itertuples(index=False)]


def _series_or_zero(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.zeros(len(df), dtype=float)
    return df[column].ffill().bfill().to_numpy(dtype=float)


def get_clean_signal(df: pd.DataFrame, region: str) -> np.ndarray:
    load_col = f"{region}_load_actual_entsoe_transparency"
    if load_col not in df.columns:
        raise KeyError(f"Missing load column for region {region}.")

    load = _series_or_zero(df, load_col)
    solar = _series_or_zero(df, f"{region}_solar_generation_actual")
    wind = _series_or_zero(df, f"{region}_wind_generation_actual")
    
    residual = load - (solar + wind)
    normalized_signal = np.where(load > 0, np.maximum(0, residual) / load, 0.0)
    return normalized_signal

def load_energy(path: str = "time_series_60min_singleindex.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def available_regions(df: pd.DataFrame) -> list[str]:
    suffix = "_load_actual_entsoe_transparency"
    return [column[: -len(suffix)] for column in df.columns if column.endswith(suffix)]

def default_spatial_regions(df: pd.DataFrame) -> list[str]:
    regions = []
    for region in available_regions(df):
        if not re.fullmatch(r"[A-Z]{2}", region):
            continue
        has_solar = f"{region}_solar_generation_actual" in df.columns
        has_wind = f"{region}_wind_generation_actual" in df.columns
        if has_solar or has_wind:
            regions.append(region)
    return regions

def build_problem(
    energy: pd.DataFrame,
    regions: Sequence[str],
    *,
    window_start: int = DEFAULT_WINDOW_START,
    window_hours: int = DEFAULT_WINDOW_HOURS,
) -> SchedulingProblem:
    signals = {}
    for region in regions:
        residual = get_clean_signal(energy, region)
        hourly_window = residual[window_start : window_start + window_hours]
        if len(hourly_window) != window_hours:
            raise ValueError(f"Window {window_start}:{window_start + window_hours} is out of range for {region}.")
        signals[region] = SignalSeries.from_hourly(hourly_window)
    return SchedulingProblem(signals)

def benchmark_algorithms(
    problem: SchedulingProblem,
    jobs: Sequence[Job],
    *,
    search_budget: int = DEFAULT_SEARCH_BUDGET,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    rows = []
    region_shares_by_algorithm: dict[str, dict[str, float]] = {}
    for algorithm_index, (name, solver) in enumerate(ALGORITHMS):
        trial_costs = []
        region_counts = {region: 0 for region in problem.regions}
        for trial in range(trials):
            rng = random.Random(seed + algorithm_index * 1000 + trial)
            total_cost = 0.0
            for job in jobs:
                budget = 1 if name == "Random Baseline" else search_budget
                result = solver(problem, job, rng, budget)
                total_cost += result.cost
                if result.candidate is not None and len(problem.regions) > 1:
                    region_counts[result.candidate.region] += 1
            trial_costs.append(total_cost)

        most_region, most_count = _region_preference(region_counts, prefer_highest=True)
        least_region, least_count = _region_preference(region_counts, prefer_highest=False)
        total_assignments = sum(region_counts.values())
        region_shares_by_algorithm[name] = {
            region: 0.0 if total_assignments == 0 else (count / total_assignments) * 100.0
            for region, count in region_counts.items()
        }
        rows.append(
            {
                "algorithm": name,
                "mean_cost": float(np.mean(trial_costs)),
                "std_cost": float(np.std(trial_costs, ddof=0)),
                "best_cost": float(np.min(trial_costs)),
                "most_preferred_region": most_region,
                "most_preferred_share_pct": 0.0 if total_assignments == 0 else (most_count / total_assignments) * 100.0,
                "least_preferred_region": least_region,
                "least_preferred_share_pct": 0.0 if total_assignments == 0 else (least_count / total_assignments) * 100.0,
            }
        )

    summary = pd.DataFrame(rows)
    random_mean = float(summary.loc[summary["algorithm"] == "Random Baseline", "mean_cost"].iloc[0])
    summary["reduction_vs_random_pct"] = ((random_mean - summary["mean_cost"]) / random_mean) * 100.0
    summary = summary.sort_values("mean_cost", kind="stable").reset_index(drop=True)
    summary["rank"] = summary.index + 1
    columns = ["rank", "algorithm", "mean_cost", "std_cost", "best_cost", "reduction_vs_random_pct"]
    if len(problem.regions) > 1:
        columns.extend(
            [
                "most_preferred_region",
                "most_preferred_share_pct",
                "least_preferred_region",
                "least_preferred_share_pct",
            ]
        )
    result = summary[columns].copy()
    result.attrs["regions"] = list(problem.regions)
    result.attrs["region_shares_by_algorithm"] = region_shares_by_algorithm
    return result


def filter_schedulable_jobs(problem: SchedulingProblem, jobs: Sequence[Job]) -> list[Job]:
    return [job for job in jobs if problem.feasible_regions(job)]


def _region_preference(region_counts: dict[str, int], *, prefer_highest: bool) -> tuple[str, int]:
    if not region_counts:
        return ("N/A", 0)
    ordered = sorted(region_counts.items(), key=lambda item: (item[1], item[0]))
    if prefer_highest:
        region, count = ordered[-1]
    else:
        region, count = ordered[0]
    return region, count


def format_summary(summary: pd.DataFrame) -> str:
    include_regions = "most_preferred_region" in summary.columns
    header = f"{'Rank':>4}  {'Algorithm':<20} {'Mean Cost':>14} {'Std Dev':>12} {'Best Cost':>14} {'Reduction vs Random':>21}"
    if include_regions:
        header += f"  {'Most Used':>14} {'Share':>8}  {'Least Used':>14} {'Share':>8}"
    lines = [header]
    for row in summary.itertuples(index=False):
        line = (
            f"{row.rank:>4}  {row.algorithm:<20} {row.mean_cost:>14,.2f} {row.std_cost:>12,.2f} "
            f"{row.best_cost:>14,.2f} {row.reduction_vs_random_pct:>20.2f}%"
        )
        if include_regions:
            line += (
                f"  {row.most_preferred_region:>14} {row.most_preferred_share_pct:>7.2f}%"
                f"  {row.least_preferred_region:>14} {row.least_preferred_share_pct:>7.2f}%"
            )
        lines.append(line)
    return "\n".join(lines)


def rank_regions_by_preference(summary: pd.DataFrame) -> pd.DataFrame:
    region_shares_by_algorithm = summary.attrs.get("region_shares_by_algorithm", {})
    regions = summary.attrs.get("regions", [])
    if not region_shares_by_algorithm or not regions:
        return pd.DataFrame(columns=["rank", "region", "weighted_score_pct", "mean_share_pct", "best_algorithm", "best_share_pct"])

    weights = {}
    for row in summary.itertuples(index=False):
        if row.algorithm == "Random Baseline":
            continue
        weights[row.algorithm] = max(float(row.reduction_vs_random_pct), 0.0)

    total_weight = sum(weights.values())
    rows = []
    for region in regions:
        weighted_sum = 0.0
        mean_sum = 0.0
        best_algorithm = "N/A"
        best_share = -1.0
        algorithm_count = 0

        for algorithm, shares in region_shares_by_algorithm.items():
            if algorithm == "Random Baseline":
                continue
            share = float(shares.get(region, 0.0))
            weight = weights.get(algorithm, 0.0)
            weighted_sum += share * weight
            mean_sum += share
            algorithm_count += 1
            if share > best_share:
                best_share = share
                best_algorithm = algorithm

        rows.append(
            {
                "region": region,
                "weighted_score_pct": 0.0 if total_weight == 0 else weighted_sum / total_weight,
                "mean_share_pct": 0.0 if algorithm_count == 0 else mean_sum / algorithm_count,
                "best_algorithm": best_algorithm,
                "best_share_pct": max(best_share, 0.0),
            }
        )

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        ["weighted_score_pct", "mean_share_pct", "region"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1
    return ranking[["rank", "region", "weighted_score_pct", "mean_share_pct", "best_algorithm", "best_share_pct"]]


def format_region_ranking(ranking: pd.DataFrame) -> str:
    lines = [
        f"{'Rank':>4}  {'Region':<8} {'Weighted Score':>15} {'Mean Share':>12} {'Best Algorithm':>20} {'Best Share':>12}",
    ]
    for row in ranking.itertuples(index=False):
        lines.append(
            f"{row.rank:>4}  {row.region:<8} {row.weighted_score_pct:>14.2f}% {row.mean_share_pct:>11.2f}% "
            f"{row.best_algorithm:>20} {row.best_share_pct:>11.2f}%"
        )
    return "\n".join(lines)


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-start", type=int, default=DEFAULT_WINDOW_START, help="First hour in the energy trace window.")
    parser.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_HOURS, help="Number of hours in the scheduling window.")
    parser.add_argument("--budget", type=int, default=DEFAULT_SEARCH_BUDGET, help="Objective evaluations per job for each optimization algorithm.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Independent runs per algorithm.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed for reproducibility.")
