# Carbon-Aware GPU Jobs Scheduling via Local Search

A machine learning scheduling system that optimizes GPU job placement across geographic regions to minimize carbon emissions using local search algorithms.

## Overview

This project addresses the problem of scheduling GPU-intensive computational jobs across multiple geographic regions with varying electricity grid carbon intensities. By leveraging residual load data (a proxy for grid carbon intensity), the system intelligently schedules jobs to run during periods and in locations where renewable energy is most abundant, thereby reducing overall carbon emissions.

## Problem Statement

GPU workloads consume significant computational resources and electricity. The carbon footprint of these jobs depends on:
- **When** the job runs (time of day affects grid carbon intensity)
- **Where** the job runs (different regions have different energy mixes)
- **Duration and scale** of the job (GPU count and runtime)

This system solves the scheduling problem using multiple local search algorithms to find near-optimal placements that minimize the total carbon cost.

## Key Features

- **Multi-Algorithm Benchmarking**: Compares multiple local search strategies:
  - Random baseline
  - Hill climbing
  - Local beam search
  - Simulated annealing
  - Tabu search
  - Genetic algorithms

- **Geographic Optimization**: Schedules jobs across multiple countries/regions with region-specific energy intensity signals

- **Flexible Job Specifications**: Handles GPU jobs with:
  - Variable runtime durations
  - Different GPU counts
  - Job-specific time windows

- **Carbon-Aware Metrics**: Uses residual load data as a proxy for grid carbon intensity to make scheduling decisions

## Project Structure

```
.
├── scheduler_core.py           # Core scheduling algorithms and problem definition
├── benchmark_spatial.py        # Spatial benchmarking across regions
├── gpu_jobs.csv               # GPU job dataset with runtimes and GPU counts
├── time_series_60min_singleindex.csv  # Hourly residual load data for regions
└── README.md                  # This file
```

## Files

### `scheduler_core.py`
Contains the main scheduling logic:
- `Job`: Data class for GPU job specifications (runtime, GPU count)
- `Candidate`: Scheduling decision (region, start time)
- `SignalSeries`: Represents hourly carbon intensity signals with efficient prefix sum computation
- `SchedulingProblem`: Defines the optimization landscape and neighborhood structure
- **Solver Functions**:
  - `hill_climbing()`: Greedy local search
  - `local_beam_search()`: Parallel local search maintaining multiple candidates
  - `simulated_annealing()`: Probabilistic search with temperature cooling
  - `tabu_search()`: Search with memory to avoid cycling
  - `genetic_algorithm()`: Population-based evolutionary search
- `benchmark_algorithms()`: Runs comparative evaluation of all algorithms

### `benchmark_spatial.py`
Executes spatial benchmarking:
- Loads GPU job specifications
- Loads energy/carbon intensity data
- Builds multi-region scheduling problems
- Runs benchmarking suite across all algorithms
- Generates region rankings and performance summaries

### `gpu_jobs.csv`
Dataset of GPU jobs with columns:
- Runtime (seconds)
- GPU count

### `time_series_60min_singleindex.csv`
Hourly residual load signals for each region (24-hour windows or longer)

## Installation

### Requirements
- Python 3.10+
- numpy
- pandas

### Setup
```bash
pip install numpy pandas
```

## Usage

### Running Spatial Benchmarking
```bash
python benchmark_spatial.py [options]
```

### Options
- `--window_start`: Start position in time series (default: 100)
- `--window_hours`: Scheduling window duration in hours (default: 24)
- `--budget`: Search budget per job (default: 80)
- `--trials`: Number of independent trials per algorithm (default: 5)
- `--seed`: Random seed for reproducibility (default: 7)

### Example
```bash
python benchmark_spatial.py --window_hours 24 --budget 100 --trials 10
```

## Algorithm Details

### Local Search Methods

**Hill Climbing**
- Starts with a random solution
- Iteratively moves to better neighbors
- Greedy, fast convergence, prone to local optima

**Local Beam Search**
- Maintains a beam of k-best candidates
- Expands each candidate to generate neighbors
- Better global search than hill climbing, higher computational cost

**Simulated Annealing**
- Accepts worse solutions probabilistically (decreasing with temperature)
- Escapes local optima through controlled randomness
- Temperature schedule balances exploration/exploitation

**Tabu Search**
- Maintains memory of recently visited solutions
- Prevents cycling back to previous solutions
- Can escape local optima through diversification

**Genetic Algorithm**
- Population-based evolutionary approach
- Crossover and mutation operators
- Maintains diverse candidate pool

### Problem Formulation

For each GPU job, the system searches for optimal placement:

**Decision Variables:**
- Region selection (which geographic region)
- Start time (when to begin job execution)

**Objective:**
- Minimize: `carbon_cost = residual_load_intensity × gpu_count × job_duration`

**Constraints:**
- Job must complete within available time window
- Start time must be within valid time range for selected region

## Output

The benchmark outputs:
- Number of countries/regions evaluated
- Number of jobs processed
- Algorithm performance comparison
- Best scheduling placements per algorithm
- Country ranking by scheduling efficiency

## Performance Considerations

- **Search Budget**: Controls computation time per job (higher budget = more evaluations)
- **Window Size**: Larger scheduling windows allow more flexibility but increase search space
- **Region Count**: More regions increase problem complexity
- **Job Count**: Scales linearly with total computation required

## Data Format

### GPU Jobs CSV
```
run_time,gpu_num
3600,2
7200,4
```

### Energy Data CSV
Columns represent different regions, rows are hourly residual load values (24+ rows for a time window):
```
Region1,Region2,Region3
50.2,45.3,52.1
48.5,47.2,50.8
...
```

## Research Applications

This project is useful for:
- **Data Center Operations**: Scheduling workloads to minimize carbon emissions
- **Cloud Provider Optimization**: Regional load balancing with environmental awareness
- **Algorithm Research**: Comparing local search methods on scheduling problems
- **Sustainability**: Quantifying carbon savings from intelligent job placement

## Future Enhancements

- Machine learning prediction of future carbon intensity
- Integration with real-time energy grid APIs
- Support for job deadlines and QoS constraints
- Multi-objective optimization (carbon + cost + latency)
- Parallel distributed scheduling

## References

Research background: [Project Report](./Project%20–%20Dal,%20Lis%20-%20Carbon-Aware%20GPU%20Jobs%20Scheduling%20via%20Local%20Search%20methods%20-%2013:05:2026.pdf)

## Authors

- Mustafa Dal
- Mateusz Miroslaw Lis
