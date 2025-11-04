# pp_entry_builder_nba.py
# ------------------------------------------------------------
# PrizePicks entry builder (NBA only) — Balanced + Standard
# ------------------------------------------------------------
# INPUT CSV schema (props.csv):
# player,stat,line,side,p,edge,group
# e.g.:
# "Stephen Curry",PTS,27.5,Under,0.58,0.08,"GSW pace-down"
# "Anthony Davis",REB,12.5,Over,0.61,0.11,"LAL frontcourt"
# "D'Angelo Russell",AST,5.5,Under,0.57,0.07,"LAL usage"
#
# Notes:
# - p = model probability of the listed 'side' hitting (0..1)
# - edge = probability edge vs 0.5 (optional but used for filtering)
# - group = loose correlation tag (pace-up, opponent, usage, etc.)
# ------------------------------------------------------------

import itertools
import math
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

# =========================
# CONFIG (Balanced + Standard)
# =========================
CSV_PATH = "prop_bet.csv"

# Probability thresholds (balanced)
ANCHOR_MIN_P = 0.60
FILLER_MIN_P = 0.56
GLOBAL_MIN_EDGE = 0.06   # ignore thin/no-edge plays

# Portfolio / bankroll (standard)
BANKROLL = 131              # set your real bankroll here
DEPLOY_PCT = 0.20           # ~20% per day
SPLIT_3FLEX = 0.45          # 45% of deployed to 3-Flex
SPLIT_5FLEX = 0.55          # 55% of deployed to 5-Flex
UNITS_PER_ENTRY = 3.0       # stake per entry

# Entry construction
MAX_ENTRIES_3 = 8              # max 3-Flex entries to output
MAX_ENTRIES_5 = 8              # max 5-Flex entries to output
MAX_LEGS_PER_PLAYER = 1        # avoid multiple legs on same player in an entry
REQUIRE_CORR_FOR_FILLERS = True  # prefer fillers that share group with at least one anchor

# Exposure caps
MAX_PLAYER_EXPOSURE_PCT = 0.2  # cap a single player's total stake vs deployed amount

# Correlation tweak (simple boost if legs share 'group')
CORR_BOOST = 1.03   # small +3% multiplicative boost to entry win prob when ≥2 legs share group

# Output styling
ANSI_COLOR = True
WARN_NEAR_CAP = 0.90   # warn if player exposure >= 90% of cap

# =========================
# ANSI helpers
# =========================
def color(txt, c):
    if not ANSI_COLOR:
        return txt
    codes = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "end": "\033[0m",
    }
    return f"{codes.get(c,'')}{txt}{codes['end']}"

def bar(pct, width=24):
    filled = int(round(width * pct))
    empty = width - filled
    b = "█" * filled + "░" * empty
    # color by pct
    col = "green" if pct < 0.5 else ("yellow" if pct < 1.0 else "red")
    return color(b, col)

# =========================
# Helpers
# =========================
def load_props(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["player", "stat", "line", "side", "p", "edge", "group"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"props.csv missing columns: {missing}")
    # Basic cleaning
    df["player"] = df["player"].astype(str).str.strip()
    df["stat"] = df["stat"].astype(str).str.strip().str.upper()
    df["side"] = df["side"].astype(str).str.title().str.strip()  # Over/Under
    df["group"] = df["group"].fillna("").astype(str).str.strip()
    df["p"] = df["p"].astype(float).clip(0, 1)
    df["edge"] = df["edge"].astype(float)
    # Filter global edge + sanity
    df = df[(df["p"] >= 0.50) & (df["edge"] >= GLOBAL_MIN_EDGE)].copy()
    return df.reset_index(drop=True)

def split_anchor_filler(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    anchors = df[df["p"] >= ANCHOR_MIN_P].copy()
    fillers = df[(df["p"] >= FILLER_MIN_P) & (df["p"] < ANCHOR_MIN_P)].copy()
    return anchors, fillers

def entry_ok(legs: List[dict]) -> bool:
    # no duplicate player and no duplicate (player, stat)
    players = [l["player"] for l in legs]
    if len(players) != len(set(players)):
        return False
    seen = set()
    for l in legs:
        k = (l["player"], l["stat"])
        if k in seen:
            return False
        seen.add(k)
    return True

def entry_win_prob(legs: List[dict], win_needed: int) -> float:
    """
    Approximate probability entry hits win condition (>= win_needed legs).
    Assumes independence, then applies a small correlation boost for shared groups.
    """
    ps = [l["p"] for l in legs]
    n = len(ps)
    prob = 0.0
    # sum over all subsets with k >= win_needed successes
    for k in range(win_needed, n + 1):
        for combo in itertools.combinations(range(n), k):
            p_succ = 1.0
            p_fail = 1.0
            chosen = set(combo)
            for i in range(n):
                if i in chosen:
                    p_succ *= ps[i]
                else:
                    p_fail *= (1 - ps[i])
            prob += p_succ * p_fail

    # correlation boost if 2+ legs share same non-empty group
    groups = [l["group"] for l in legs if l["group"]]
    if groups:
        counts = Counter(groups)
        if any(c >= 2 for c in counts.values()):
            prob *= CORR_BOOST

    return min(max(prob, 0.0), 1.0)

def choose_fillers(anchors: List[dict], fillers_df: pd.DataFrame, need: int) -> List[List[dict]]:
    """
    Return a list of filler sets (each list of dicts) that pair well with given anchors.
    Strategy (balanced):
      - Prefer fillers sharing at least one 'group' with any anchor (if REQUIRE_CORR_FOR_FILLERS)
      - Otherwise fall back to top-p fillers
    """
    if need <= 0:
        return [[]]

    anchor_groups = set(a["group"] for a in anchors if a["group"])
    f = fillers_df.copy()

    if REQUIRE_CORR_FOR_FILLERS and anchor_groups:
        f["is_corr"] = f["group"].apply(lambda g: g in anchor_groups if g else False)
        preferred = f[f["is_corr"]].sort_values(["p", "edge"], ascending=False)
        backup = f[~f["is_corr"]].sort_values(["p", "edge"], ascending=False)
        ranked = pd.concat([preferred, backup], ignore_index=True)
    else:
        ranked = f.sort_values(["p", "edge"], ascending=False)

    # generate combos of size 'need', top-200 candidates to keep it fast
    ranked = ranked.head(200)
    legs = ranked.to_dict("records")
    combos = []
    for combo in itertools.combinations(legs, need):
        combo = list(combo)
        # ensure no duplicate player in fillers themselves
        if len({c["player"] for c in combo}) != len(combo):
            continue
        combos.append(combo)
        if len(combos) >= 300:
            break
    return combos

def build_entries_3flex(anchors_df: pd.DataFrame, fillers_df: pd.DataFrame) -> List[dict]:
    entries = []
    anchors = anchors_df.sort_values(["p", "edge"], ascending=False).to_dict("records")
    # 2 anchors + 1 filler
    for a2 in itertools.combinations(anchors, 2):
        if not entry_ok(list(a2)):
            continue
        filler_sets = choose_fillers(list(a2), fillers_df, need=1)
        for fs in filler_sets:
            legs = list(a2) + fs
            if not entry_ok(legs):
                continue
            win_prob = entry_win_prob(legs, win_needed=2)
            entries.append({"legs": legs, "win_prob": win_prob, "type": "3FLEX"})
            if len(entries) >= 1000:
                break
        if len(entries) >= 1000:
            break
    # rank by win prob
    entries.sort(key=lambda e: (-e["win_prob"], -sum(l["p"] for l in e["legs"])))
    return entries

def build_entries_5flex(anchors_df: pd.DataFrame, fillers_df: pd.DataFrame) -> List[dict]:
    entries = []
    anchors = anchors_df.sort_values(["p", "edge"], ascending=False).to_dict("records")
    # 3 anchors + 2 fillers
    for a3 in itertools.combinations(anchors, 3):
        if not entry_ok(list(a3)):
            continue
        filler_sets = choose_fillers(list(a3), fillers_df, need=2)
        for fs in filler_sets:
            legs = list(a3) + fs
            if not entry_ok(legs):
                continue
            win_prob = entry_win_prob(legs, win_needed=3)
            entries.append({"legs": legs, "win_prob": win_prob, "type": "5FLEX"})
            if len(entries) >= 1000:
                break
        if len(entries) >= 1000:
            break
    entries.sort(key=lambda e: (-e["win_prob"], -sum(l["p"] for l in e["legs"])))
    return entries

# ---------- Exposure management ----------
def cap_exposure(entries: List[dict], deployed_amount: float, units_per_entry: float,
                 max_player_exposure_pct: float) -> Tuple[List[dict], List[dict]]:
    """
    Enforce per-player exposure cap across the list (greedy keep).
    Returns: (kept_entries, rejected_entries_due_to_cap)
    """
    cap = deployed_amount * max_player_exposure_pct
    spend_by_player: Dict[str, float] = defaultdict(float)
    kept, rejected = [], []
    for e in entries:
        players = {l["player"] for l in e["legs"]}
        if any(spend_by_player[p] + units_per_entry > cap for p in players):
            rejected.append(e)
            continue
        kept.append(e)
        for p in players:
            spend_by_player[p] += units_per_entry
    return kept, rejected

def exposure_table(entries: List[dict], stake: float, deployed_amount: float) -> List[Tuple[str, float, float]]:
    """Return list of (player, dollar_exposure, pct_of_cap)."""
    cap = deployed_amount * MAX_PLAYER_EXPOSURE_PCT
    spend: Dict[str, float] = defaultdict(float)
    for e in entries:
        players = {l["player"] for l in e["legs"]}
        for p in players:
            spend[p] += stake
    out = []
    for p, amt in sorted(spend.items(), key=lambda kv: -kv[1]):
        pct_of_cap = amt / cap if cap > 0 else 0.0
        out.append((p, amt, pct_of_cap))
    return out

def format_legs(legs: List[dict]) -> str:
    return " | ".join([f"{l['player']} {l['stat']} {l['side']} {l['line']} (p={l['p']:.2f}, {l['group'] or '—'})"
                       for l in legs])

def print_entries(label: str, entries: List[dict], max_entries: int, stake: float):
    print(color(f"\n=== {label} (top {max_entries}) ===", "cyan"))
    for i, e in enumerate(entries[:max_entries], 1):
        legs_txt = format_legs(e["legs"])
        print(f"{i:>2}. win_prob={e['win_prob']:.3f}  stake=${stake:,.2f}")
        print(f"    {legs_txt}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    df = load_props(CSV_PATH)
    anchors_df, fillers_df = split_anchor_filler(df)

    if anchors_df.empty:
        raise SystemExit("No anchors (p ≥ {:.2f}) available. Loosen thresholds or add more props."
                         .format(ANCHOR_MIN_P))
    if fillers_df.empty:
        print(color("Warning: No fillers in {:.2f}–{:.2f} range; will attempt anchors-only combos where possible."
                    .format(FILLER_MIN_P, ANCHOR_MIN_P), "yellow"))

    # Build candidate entries
    entries3 = build_entries_3flex(anchors_df, fillers_df)
    entries5 = build_entries_5flex(anchors_df, fillers_df)

    # Budget
    deploy_amt = BANKROLL * DEPLOY_PCT
    budget_3 = deploy_amt * SPLIT_3FLEX
    budget_5 = deploy_amt * SPLIT_5FLEX

    # Exposure-capped lists
    entries3_cap, entries3_rej = cap_exposure(entries3, deploy_amt, UNITS_PER_ENTRY, MAX_PLAYER_EXPOSURE_PCT)
    entries5_cap, entries5_rej = cap_exposure(entries5, deploy_amt, UNITS_PER_ENTRY, MAX_PLAYER_EXPOSURE_PCT)

    # How many we can afford
    n3 = min(MAX_ENTRIES_3, int(budget_3 // UNITS_PER_ENTRY), len(entries3_cap))
    n5 = min(MAX_ENTRIES_5, int(budget_5 // UNITS_PER_ENTRY), len(entries5_cap))

    # Slice to budget
    entries3_final = entries3_cap[:n3]
    entries5_final = entries5_cap[:n5]

    # ===== Header =====
    print(color(f"\n====== PrizePicks Entry Plan (NBA • BALANCED • STANDARD) ======", "bold"))
    print(f"Bankroll: ${BANKROLL:,.2f} | Deploy {DEPLOY_PCT*100:.0f}% → ${deploy_amt:,.2f}")
    print(f"3-Flex budget: ${budget_3:,.2f}  | 5-Flex budget: ${budget_5:,.2f}")
    print(f"Units per entry: ${UNITS_PER_ENTRY:.2f}  | Player exposure cap: {int(MAX_PLAYER_EXPOSURE_PCT*100)}% of deployed")

    # ===== What to bet (concise checklist) =====
    total_cards = len(entries3_final) + len(entries5_final)
    print(color("\nWHAT TO BET (checklist)", "bold"))
    if total_cards == 0:
        print(color("No entries fit budget/exposure caps. Reduce UNITS_PER_ENTRY or relax thresholds.", "red"))
    else:
        if entries3_final:
            print(color(f"\n3-PICK FLEX — place {len(entries3_final)} entries @ ${UNITS_PER_ENTRY:.2f} each:", "cyan"))
            for i, e in enumerate(entries3_final, 1):
                print(f"  {i}. win_prob={e['win_prob']:.3f} | {format_legs(e['legs'])}")
        if entries5_final:
            print(color(f"\n5-PICK FLEX — place {len(entries5_final)} entries @ ${UNITS_PER_ENTRY:.2f} each:", "cyan"))
            for i, e in enumerate(entries5_final, 1):
                print(f"  {i}. win_prob={e['win_prob']:.3f} | {format_legs(e['legs'])}")

    # ===== Verbose lists (ranked) =====
    print_entries("3-PICK FLEX (2 anchors + 1 correlated filler)", entries3_cap, n3, UNITS_PER_ENTRY)
    print_entries("5-PICK FLEX (3 anchors + 2 correlated fillers)", entries5_cap, n5, UNITS_PER_ENTRY)

    # ===== Exposure summary =====
    print(color("\n--- Player Exposure (after selected entries) ---", "bold"))
    exp_rows = exposure_table(entries3_final + entries5_final, UNITS_PER_ENTRY, deploy_amt)
    if not exp_rows:
        print("No exposure (no entries selected).")
    else:
        cap_amt = deploy_amt * MAX_PLAYER_EXPOSURE_PCT
        for player, dollars, pct_of_cap in exp_rows:
            flag = ""
            if pct_of_cap >= 1.0:
                flag = color("  [OVER CAP — remove lowest-ranked entry with this player]", "red")
            elif pct_of_cap >= WARN_NEAR_CAP:
                flag = color("  [near cap]", "yellow")
            print(f"{player:<22} ${dollars:>5.2f} / ${cap_amt:.2f} cap  "
                  f"{bar(min(pct_of_cap, 1.2))}  ({pct_of_cap*100:5.1f}%)" + flag)

    # ===== Rejections due to exposure =====
    rejected_total = len(entries3_rej) + len(entries5_rej)
    if rejected_total > 0:
        print(color(f"\nFiltered out {rejected_total} entries due to exposure cap:", "yellow"))
        if entries3_rej:
            print(color(f"  3-FLEX rejected: {len(entries3_rej)} (examples below)", "yellow"))
            for e in entries3_rej[:3]:
                print(f"    win_prob={e['win_prob']:.3f} | {format_legs(e['legs'])}")
        if entries5_rej:
            print(color(f"  5-FLEX rejected: {len(entries5_rej)} (examples below)", "yellow"))
            for e in entries5_rej[:3]:
                print(f"    win_prob={e['win_prob']:.3f} | {format_legs(e['legs'])}")

    # ===== Footer guidance =====
    if total_cards > 0:
        print(color("\nHow to interpret:", "bold"))
        print("• Place the entries listed under WHAT TO BET with the shown stake per entry.")
        print("• A player may appear in multiple entries; that’s fine as long as exposure stays under the cap.")
        print("• If any player exceeds the cap (marked in red), drop the lowest-ranked entry containing that player.")
        print("• To see more entries, lower thresholds (ANCHOR_MIN_P/FILLER_MIN_P) or GLOBAL_MIN_EDGE,")
        print("  but expect lower average EV as you expand the slate.")
    else:
        print(color("\nTip:", "bold") + " Add more high-probability props to your CSV or reduce thresholds to generate entries.")
