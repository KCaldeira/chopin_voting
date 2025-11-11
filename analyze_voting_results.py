import pandas as pd
import numpy as np
import itertools

# ============================================
# 1. Load data
# ============================================

def load_scores(csv_path: str):
    """
    Load a CSV of the form:
      Contestant,Judge1,Judge2,...,JudgeN
      Name1,25,24,...,23
      Name2,NA,23,...,24

    Returns:
      contestants: list of contestant names
      judges: list of judge column names
      scores: DataFrame of numeric scores (rows=contestants, cols=judges)
      df: original DataFrame including 'Contestant'
    """
    df = pd.read_csv(csv_path, na_values=["NA"])
    contestants = df["Contestant"].tolist()
    judges = [c for c in df.columns if c != "Contestant"]
    scores = df[judges].astype(float)
    return contestants, judges, scores, df


# ============================================
# 2. Pairwise / Condorcet (Copeland)
# ============================================

def compute_pairwise(scores: pd.DataFrame, contestants, judges):
    """
    Compute pairwise wins matrix, ties, and comparison counts.

    wins[a,b] = #judges preferring a over b
    ties[a,b] = #judges giving same score to a and b
    comps[a,b] = #judges that scored both a and b (non-NA)
    """
    n = len(contestants)
    wins = np.zeros((n, n), dtype=int)
    ties = np.zeros((n, n), dtype=int)
    comps = np.zeros((n, n), dtype=int)

    for j in judges:
        col = scores[j]
        for a_idx, b_idx in itertools.combinations(range(n), 2):
            sa = col.iloc[a_idx]
            sb = col.iloc[b_idx]
            if pd.isna(sa) or pd.isna(sb):
                continue
            comps[a_idx, b_idx] += 1
            comps[b_idx, a_idx] += 1
            if sa > sb:
                wins[a_idx, b_idx] += 1
            elif sb > sa:
                wins[b_idx, a_idx] += 1
            else:
                ties[a_idx, b_idx] += 1
                ties[b_idx, a_idx] += 1

    return wins, ties, comps


def copeland_scores(wins, comps, contestants):
    """
    Copeland score: +1 per opponent you beat, -1 per opponent that beats you.

    Returns:
      ranked: list of (contestant, copeland_score), sorted descending
      condorcet: list of Condorcet winners (usually 0 or 1 names)
    """
    n = len(contestants)
    copeland = np.zeros(n, dtype=int)

    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            if comps[a, b] == 0:
                continue  # no comparison, ignore
            if wins[a, b] > wins[b, a]:
                copeland[a] += 1
            elif wins[b, a] > wins[a, b]:
                copeland[a] -= 1
            # ties give 0

    order = sorted(range(n), key=lambda i: copeland[i], reverse=True)
    ranked = [(contestants[i], int(copeland[i])) for i in order]

    # Condorcet winner: beats every other candidate
    condorcet = []
    for a in range(n):
        beats_all = True
        for b in range(n):
            if a == b:
                continue
            if comps[a, b] == 0:
                continue
            if wins[a, b] <= wins[b, a]:
                beats_all = False
                break
        if beats_all:
            condorcet.append(contestants[a])

    return ranked, condorcet


# ============================================
# 3. IRV (ranked-choice) with fractional top ties
# ============================================

def total_raw_score(scores: pd.DataFrame, df: pd.DataFrame, contestant: str):
    """Sum of all non-NA scores for a given contestant."""
    row = scores.loc[df["Contestant"] == contestant].iloc[0]
    return row.dropna().sum()


def irv_fractional_top(scores: pd.DataFrame, df: pd.DataFrame, contestants, judges):
    """
    IRV-style elimination:
      - In each round, each judge gives 1 vote total.
      - Among remaining contestants they scored, those with max score
        share the vote equally (fractionally).
      - Candidate with lowest total votes is eliminated each round.
      - Tie-breaker: lowest total raw score, then alphabetical.

    Returns:
      winner: name of the IRV winner
      elimination_order: list of dicts with keys:
          'round', 'eliminated', 'votes'
    """
    remaining = set(contestants)
    elimination_order = []
    round_num = 1

    while True:
        votes = {c: 0.0 for c in remaining}
        active_judges = 0

        for j in judges:
            col = scores[j]
            sub = col[df["Contestant"].isin(remaining)].dropna()
            if sub.empty:
                continue
            active_judges += 1
            max_score = sub.max()
            top_idx = sub[sub == max_score].index
            share = 1.0 / len(top_idx)
            for idx in top_idx:
                cand = df.loc[idx, "Contestant"]
                votes[cand] += share

        if active_judges == 0 or not remaining:
            return None, elimination_order

        total_votes = float(active_judges)

        # Check majority
        winner = None
        for c in remaining:
            if votes[c] > total_votes / 2.0:
                winner = c
                break

        if winner is not None:
            return winner, elimination_order

        # Eliminate lowest vote-getter
        min_vote = min(votes[c] for c in remaining)
        lowest = [c for c in remaining if votes[c] == min_vote]

        if len(lowest) > 1:
            lowest = sorted(
                lowest,
                key=lambda c: (total_raw_score(scores, df, c), c)
            )
        elim = lowest[0]
        elimination_order.append(
            {"round": round_num, "eliminated": elim, "votes": votes[elim]}
        )
        remaining.remove(elim)

        if len(remaining) == 1:
            last = list(remaining)[0]
            return last, elimination_order

        round_num += 1


# ============================================
# 4. Bottom-elimination (reverse runoff)
# ============================================

def bottom_elimination_fractional(scores: pd.DataFrame, df: pd.DataFrame, contestants, judges):
    """
    Reverse elimination:
      - In each round, for each judge, find their lowest score among remaining.
      - Everyone tied at that lowest score gets equal share of a "bottom vote".
      - Candidate with highest total bottom-votes is eliminated.
      - Tie-breaker: lowest total raw score, then alphabetical.

    Returns:
      winner: name of survivor
      elimination_order: list of dicts with keys:
          'round', 'eliminated', 'bottom_score'
    """
    remaining = set(contestants)
    elimination_order = []
    round_num = 1

    while len(remaining) > 1:
        bottom_counts = {c: 0.0 for c in remaining}

        for j in judges:
            col = scores[j]
            sub = col[df["Contestant"].isin(remaining)].dropna()
            if sub.empty:
                continue
            min_score = sub.min()
            bottom_idx = sub[sub == min_score].index
            share = 1.0 / len(bottom_idx)
            for idx in bottom_idx:
                cand = df.loc[idx, "Contestant"]
                bottom_counts[cand] += share

        max_bottom = max(bottom_counts[c] for c in remaining)
        candidates_max_bottom = [c for c in remaining if bottom_counts[c] == max_bottom]

        if len(candidates_max_bottom) > 1:
            candidates_max_bottom = sorted(
                candidates_max_bottom,
                key=lambda c: (total_raw_score(scores, df, c), c)
            )
        elim = candidates_max_bottom[0]
        elimination_order.append(
            {"round": round_num, "eliminated": elim, "bottom_score": bottom_counts[elim]}
        )
        remaining.remove(elim)
        round_num += 1

    winner = list(remaining)[0]
    return winner, elimination_order


# ============================================
# 5. Favorite metrics (top picks)
# ============================================

def favorites_fractional(scores: pd.DataFrame, df: pd.DataFrame, contestants, judges):
    """
    For each judge:
      - Find max score among contestants they scored.
      - All contestants with that max share 1 vote (fractionally).
    Returns dict: {contestant: favorite_share}
    """
    fav = {c: 0.0 for c in contestants}

    for j in judges:
        col = scores[j]
        valid = col.dropna()
        if valid.empty:
            continue
        max_score = valid.max()
        top_idx = valid[valid == max_score].index
        share = 1.0 / len(top_idx)
        for idx in top_idx:
            cand = df.loc[idx, "Contestant"]
            fav[cand] += share

    return fav


def favorites_equal(scores: pd.DataFrame, df: pd.DataFrame, contestants, judges):
    """
    For each judge:
      - Find max score among contestants they scored.
      - All contestants with that max each get 1 favorite point (no sharing).
    Returns dict: {contestant: favorite_count}
    """
    fav = {c: 0.0 for c in contestants}

    for j in judges:
        col = scores[j]
        valid = col.dropna()
        if valid.empty:
            continue
        max_score = valid.max()
        top_idx = valid[valid == max_score].index
        for idx in top_idx:
            cand = df.loc[idx, "Contestant"]
            fav[cand] += 1.0

    return fav


# ============================================
# 6. Diagnostics: judge top-score summary
# ============================================

def judge_top_summary(scores: pd.DataFrame, judges):
    """
    Returns DataFrame with:
      Judge, MaxScore, NumContestantsAtMax
    """
    rows = []
    for j in judges:
        col = scores[j]
        valid = col.dropna()
        if valid.empty:
            continue
        max_score = valid.max()
        count_max = int((valid == max_score).sum())
        rows.append({
            "Judge": j,
            "MaxScore": float(max_score),
            "NumContestantsAtMax": count_max
        })
    return pd.DataFrame(rows)


# ============================================
# 7. Diagnostics: per-judge favorite contributions for one contestant
# ============================================

def favorite_contributions_for_candidate(scores: pd.DataFrame,
                                         df: pd.DataFrame,
                                         judges,
                                         candidate: str):
    """
    For a given candidate, show judge-by-judge:
      - judge
      - judge's max score
      - candidate's score (or NA)
      - number of contestants at that max
      - candidate's fractional favorite share (ties split)
      - candidate's equal favorite count (1 if at max, else 0)
    Returns a DataFrame.
    """
    row = df[df["Contestant"] == candidate]
    if row.empty:
        raise ValueError(f"Candidate '{candidate}' not found in data.")
    row = row.iloc[0]

    records = []
    for j in judges:
        col = scores[j]
        valid = col.dropna()
        if valid.empty:
            continue
        max_score = valid.max()
        top_idx = valid[valid == max_score].index
        num_at_max = len(top_idx)
        share_fractional = 1.0 / num_at_max
        cand_score = row[j]
        cand_at_max = (not pd.isna(cand_score)) and (cand_score == max_score)
        contrib_fractional = share_fractional if cand_at_max else 0.0
        contrib_equal = 1.0 if cand_at_max else 0.0

        records.append({
            "Judge": j,
            "JudgeMaxScore": float(max_score),
            "CandidateScore": float(cand_score) if not pd.isna(cand_score) else np.nan,
            "NumContestantsAtMax": num_at_max,
            "FractionalShare": contrib_fractional,
            "EqualShare": contrib_equal
        })

    return pd.DataFrame(records)


# ============================================
# 8. Example usage
# ============================================

if __name__ == "__main__":
    # Adjust path to your CSV file
    csv_path = "Chopin_Stage3_scores_NA.csv"
    candidate_to_inspect = "Eric Lu"   # you can change this to any contestant name

    contestants, judges, scores, df = load_scores(csv_path)

    # --- Pairwise / Condorcet ---
    wins, ties, comps = compute_pairwise(scores, contestants, judges)
    copeland_ranking, condorcet = copeland_scores(wins, comps, contestants)
    print("Copeland ranking (pairwise):")
    for name, score in copeland_ranking:
        print(f"  {name}: {score}")
    print("Condorcet winner(s):", condorcet)

    # --- IRV (top-down) ---
    irv_winner, irv_elims = irv_fractional_top(scores, df, contestants, judges)
    print("\nIRV winner:", irv_winner)
    print("IRV elimination order:")
    for r in irv_elims:
        print(f"  Round {r['round']:2d}: eliminated {r['eliminated']} (votes={r['votes']:.3f})")

    # --- Bottom-elimination (reverse) ---
    bottom_winner, bottom_elims = bottom_elimination_fractional(scores, df, contestants, judges)
    print("\nBottom-elimination winner:", bottom_winner)
    print("Bottom-elimination order:")
    for r in bottom_elims:
        print(f"  Round {r['round']:2d}: eliminated {r['eliminated']} (bottom_score={r['bottom_score']:.3f})")

    # --- Favorite metrics ---
    fav_frac = favorites_fractional(scores, df, contestants, judges)
    fav_equal = favorites_equal(scores, df, contestants, judges)

    print("\nFavorites (fractional, ties share 1):")
    for name, val in sorted(fav_frac.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {val:.3f}")

    print("\nFavorites (equal, ties each get 1):")
    for name, val in sorted(fav_equal.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {val:.1f}")

    # --- Diagnostics: judges' top-score summary ---
    print("\nJudge top-score summary:")
    jt = judge_top_summary(scores, judges)
    print(jt.to_string(index=False))

    # --- Diagnostics: per-judge contributions for one candidate ---
    print(f"\nPer-judge favorite contributions for '{candidate_to_inspect}':")
    contrib_df = favorite_contributions_for_candidate(scores, df, judges, candidate_to_inspect)
    # Show with some rounding for readability
    contrib_df_round = contrib_df.copy()
    contrib_df_round["CandidateScore"] = contrib_df_round["CandidateScore"].round(2)
    contrib_df_round["FractionalShare"] = contrib_df_round["FractionalShare"].round(3)
    contrib_df_round["EqualShare"] = contrib_df_round["EqualShare"].round(3)
    print(contrib_df_round.to_string(index=False))

    # Also show the totals for that candidate
    total_frac = contrib_df["FractionalShare"].sum()
    total_equal = contrib_df["EqualShare"].sum()
    print(f"\nTotals for '{candidate_to_inspect}':")
    print(f"  Fractional favorite total: {total_frac:.3f}")
    print(f"  Equal-count favorite total: {total_equal:.1f}")
