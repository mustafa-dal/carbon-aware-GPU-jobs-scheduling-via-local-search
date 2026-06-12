import argparse

from scheduler_core import (
    add_shared_arguments,
    benchmark_algorithms,
    build_problem,
    default_spatial_regions,
    filter_schedulable_jobs,
    format_summary,
    format_region_ranking,
    load_energy,
    load_gpu_jobs,
    rank_regions_by_preference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial scheduling benchmark across multiple countries using residual load.")
    add_shared_arguments(parser)
    args = parser.parse_args()
    print("Starting...")
    jobs = load_gpu_jobs()
    energy = load_energy()
    print(f"Loaded {len(jobs)} GPU jobs and energy data for {len(energy)} regions.")
    countries =default_spatial_regions(energy)
    problem = build_problem(
        energy,
        countries,
        window_start=args.window_start,
        window_hours=args.window_hours,
    )
    schedulable_jobs = filter_schedulable_jobs(problem, jobs)
    summary = benchmark_algorithms(
        problem,
        schedulable_jobs,
        search_budget=args.budget,
        trials=args.trials,
        seed=args.seed,
    )

    print("Spatial Residual-Load Scheduling Benchmark")
    print(f"Countries benchmarked ({len(countries)}): {', '.join(countries)}")
    print(f"GPU jobs benchmarked: {len(schedulable_jobs)}")
    print(f"Excluded jobs longer than the scheduling window: {len(jobs) - len(schedulable_jobs)}")
    print(format_summary(summary))
    region_ranking = rank_regions_by_preference(summary)
    print()
    print("Country Ranking Weighted by Algorithm Efficiency")
    print(format_region_ranking(region_ranking))
    best = summary.iloc[0]
    print()
    print(
        f"Best spatial algorithm: {best['algorithm']} "
        f"with mean cost {best['mean_cost']:.2f}"
    )
    print(
        f"Most preferred region for {best['algorithm']}: {best['most_preferred_region']} "
        f"({best['most_preferred_share_pct']:.2f}% of assignments)"
    )
    print(
        f"Least preferred region for {best['algorithm']}: {best['least_preferred_region']} "
        f"({best['least_preferred_share_pct']:.2f}% of assignments)"
    )


if __name__ == "__main__":
    main()
