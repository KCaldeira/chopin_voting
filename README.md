# Chopin Competition Voting Analysis

Analysis of voting results from the Chopin Piano Competition using multiple ranking and elimination methods. This project implements various voting systems to analyze judge scores and determine winners through different methodologies.

## Overview

This tool analyzes competition voting data where multiple judges score contestants. It handles missing data (NA values) and implements several voting methods:

- **Copeland/Condorcet Method**: Pairwise comparison ranking
- **IRV (Instant Runoff Voting)**: Top-down elimination with fractional vote sharing for ties
- **Bottom Elimination**: Reverse runoff method
- **Favorite Metrics**: Analysis of top picks (both fractional and equal counting)

## Data Format

Input CSV files should follow this structure:

```
Contestant,Judge1,Judge2,Judge3,...
Contestant A,25,24,23,...
Contestant B,NA,23,24,...
Contestant C,22,25,NA,...
```

- First column: Contestant names
- Subsequent columns: Judge scores (numeric values or NA for missing)
- Higher scores are better

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Edit the `__main__` section in `analyze_voting_results.py` to specify:
- `csv_path`: Path to your CSV file
- `candidate_to_inspect`: Name of candidate for detailed analysis

Then run:

```bash
python analyze_voting_results.py
```

### Programmatic Usage

```python
from analyze_voting_results import load_scores, copeland_scores, irv_fractional_top

# Load data
contestants, judges, scores, df = load_scores("path/to/scores.csv")

# Get Copeland ranking
wins, ties, comps = compute_pairwise(scores, contestants, judges)
copeland_ranking, condorcet = copeland_scores(wins, comps, contestants)

# Get IRV winner
irv_winner, irv_elims = irv_fractional_top(scores, df, contestants, judges)
```

## Voting Methods Explained

### 1. Copeland/Condorcet Method

**Pairwise comparison** where each contestant is compared head-to-head with every other contestant:
- For each pair (A, B), count how many judges prefer A over B
- Copeland score: +1 for each opponent beaten, -1 for each loss
- A Condorcet winner beats all other contestants in pairwise comparisons

**Use case**: Identifies the candidate who would win most head-to-head matchups.

### 2. IRV (Instant Runoff Voting)

**Top-down elimination** process:
- Each round, judges give 1 vote total to their top-scoring remaining contestant(s)
- If multiple contestants tie for top score from one judge, they split that vote fractionally
- Contestant with fewest votes is eliminated
- Process repeats until one candidate has majority (>50% of votes)
- Tie-breaker: lowest total raw score, then alphabetical

**Use case**: Ensures winner has majority support through successive elimination of least-preferred candidates.

### 3. Bottom Elimination (Reverse Runoff)

**Bottom-up elimination** process:
- Each round, judges give a "bottom vote" to their lowest-scoring remaining contestant(s)
- Contestants tied for lowest from one judge split that bottom vote fractionally
- Contestant with most bottom votes is eliminated
- Process repeats until one candidate remains
- Tie-breaker: lowest total raw score, then alphabetical

**Use case**: Identifies the candidate who is least often anyone's worst choice.

### 4. Favorite Metrics

**Top pick analysis**:

- **Fractional**: Each judge's vote is split equally among all contestants tied for their maximum score
- **Equal**: Each contestant at a judge's maximum score receives 1 full point (no splitting)

**Use case**: Shows which contestants are most often among judges' top choices.

## Project Structure

```
chopin_voting/
├── analyze_voting_results.py    # Main analysis script
├── data/
│   └── input/                   # Input CSV files
│       ├── Chopin_Stage1_scores_NA.csv
│       ├── Chopin_Stage2_scores_NA.csv
│       ├── Chopin_Stage3_scores_NA.csv
│       └── Chopin_FinalStage_scores_NA.csv
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Available Functions

### Data Loading
- `load_scores(csv_path)`: Load and parse CSV data

### Pairwise/Condorcet
- `compute_pairwise(scores, contestants, judges)`: Compute pairwise comparison matrices
- `copeland_scores(wins, comps, contestants)`: Calculate Copeland scores and identify Condorcet winner

### Elimination Methods
- `irv_fractional_top(scores, df, contestants, judges)`: IRV with fractional vote sharing
- `bottom_elimination_fractional(scores, df, contestants, judges)`: Reverse elimination method

### Favorite Analysis
- `favorites_fractional(scores, df, contestants, judges)`: Fractional top pick counting
- `favorites_equal(scores, df, contestants, judges)`: Equal top pick counting

### Diagnostics
- `judge_top_summary(scores, judges)`: Summary of each judge's scoring patterns
- `favorite_contributions_for_candidate(scores, df, judges, candidate)`: Detailed per-judge analysis for specific candidate
- `total_raw_score(scores, df, contestant)`: Sum of all scores for a contestant

## Example Output

The script produces:

1. Copeland ranking with scores
2. Condorcet winner identification (if exists)
3. IRV winner with elimination order
4. Bottom-elimination winner with elimination order
5. Favorite counts (fractional and equal methods)
6. Judge top-score patterns
7. Detailed per-judge contributions for specified candidate

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0

## Notes

- All methods handle missing data (NA values) by excluding those judge-contestant pairs from calculations
- Tie-breaking is consistent across methods: lower total raw score, then alphabetical order
- Fractional vote sharing prevents judges who give ties from having outsized influence
