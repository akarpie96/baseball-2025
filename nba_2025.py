# nba_props_multiseason_with_opp_and_sugg.py
import math
import time
from typing import Tuple, List, Dict, Optional
from datetime import datetime
from dateutil import parser as dateparser

import pandas as pd
from nba_api.stats.static import players as players_static
from nba_api.stats.static import teams as teams_static
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats

# ----------------------------
# Probability helpers
# ----------------------------
def fit_theta(mean_val: float, var_val: float) -> float:
    """Negative Binomial shape parameter; large -> ~Poisson behavior."""
    if mean_val <= 0 or var_val <= mean_val:
        return 1e9
    return (mean_val ** 2) / (var_val - mean_val)

def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    s = 0.0
    for x in range(0, k + 1):
        s += math.exp(-lam + x * math.log(lam) - math.lgamma(x + 1))
    return min(max(s, 0.0), 1.0)

def nb_cdf(k: int, r: float, p: float) -> float:
    if k < 0:
        return 0.0
    s = 0.0
    for x in range(0, k + 1):
        logC = math.lgamma(x + r) - math.lgamma(r) - math.lgamma(x + 1)
        s += math.exp(logC + x * math.log(1 - p) + r * math.log(p))
    return min(max(s, 0.0), 1.0)

def over_under_prob(line: float, mean_count: float, theta: float) -> Tuple[float, float]:
    """
    Probability of strictly Over/Under a non-integer line for a count stat.
    For integer lines, floor(line) is Under-inclusive.
    """
    k_floor = math.floor(line)
    if not math.isfinite(theta) or theta <= 0 or theta > 1e8:
        cdf = poisson_cdf(k_floor, mean_count)
        return (1.0 - cdf, cdf)
    r = float(theta)
    p = r / (r + mean_count)
    cdf = nb_cdf(k_floor, r, p)
    return (1.0 - cdf, cdf)

# ----------------------------
# Date & format helpers
# ----------------------------
def _min_to_float(mstr) -> float:
    try:
        if isinstance(mstr, (int, float)):
            return float(mstr)
        s = str(mstr)
        if ":" in s:
            mm, ss = s.split(":")
            return float(mm) + float(ss) / 60.0
        return float(s)
    except Exception:
        return 0.0

def _parse_game_dates(series: pd.Series) -> pd.Series:
    """
    Robust parser for mixed NBA date formats like 'Apr 30, 2025' and 'October 10, 2024'.
    Tries pandas mixed parsing; falls back to element-wise parsing with dateutil.
    """
    try:
        return pd.to_datetime(series, format="mixed", errors="raise")
    except Exception:
        pass

    def _safe_parse(x):
        if pd.isna(x):
            return pd.NaT
        s = str(x)
        for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        try:
            return dateparser.parse(s, fuzzy=True)
        except Exception:
            return pd.NaT

    return series.apply(_safe_parse)

# ----------------------------
# Data fetchers
# ----------------------------
def find_player_id(full_name: str) -> int:
    matches = players_static.find_players_by_full_name(full_name)
    if not matches:
        raise ValueError(f"No NBA player found for '{full_name}'.")
    for m in matches:
        if m.get("full_name", "").lower() == full_name.lower():
            return m["id"]
    return matches[0]["id"]

def find_team_id_by_abbr(abbr: str) -> Optional[int]:
    abbr = abbr.upper().strip()
    teams = teams_static.get_teams()
    for t in teams:
        if t.get("abbreviation") == abbr:
            return t["id"]
    return None

def fetch_player_gamelogs(player_id: int, season: str, season_type: str) -> pd.DataFrame:
    """
    season: '2024-25'
    season_type: 'Regular Season' | 'Playoffs' | 'Pre Season' | 'All-Star'
    """
    for attempt in range(3):
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star=season_type,
                timeout=30
            )
            df = gl.get_data_frames()[0]
            if df.empty:
                return pd.DataFrame()
            keep = ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST"]
            df = df[keep].copy()
            df["GAME_DATE"] = _parse_game_dates(df["GAME_DATE"])
            df["MIN"] = df["MIN"].apply(_min_to_float)
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            df = df.sort_values("GAME_DATE").reset_index(drop=True)
            return df
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame()

def fetch_multi_season_logs(player_id: int, seasons: List[str], season_types: List[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        for stype in season_types:
            df = fetch_player_gamelogs(player_id, s, stype)
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("GAME_DATE").reset_index(drop=True)
    return out

# ----------------------------
# Opponent difficulty (robust)
# ----------------------------
def fetch_opponent_allowed(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    """
    Pull team opponent-allowed stats for PTS/REB/AST.
    Tries multiple column-name variants to be robust across API changes.
    Returns columns: TEAM_ID, TEAM_NAME, OPP_PTS, OPP_REB, OPP_AST
    """
    for attempt in range(3):
        try:
            # Prefer explicit Opponent measure set
            dash = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                per_mode_detailed=per_mode,
                measure_type_detailed_defense="Opponent",
                timeout=30
            )
            df = dash.get_data_frames()[0].copy()
            cols = {c.upper(): c for c in df.columns}

            # Candidate names seen across NBA endpoints/eras
            opp_pts_candidates = ["OPP_PTS", "PTS_ALLOWED", "OPP_PTS_PER_GAME"]
            opp_reb_candidates = ["OPP_REB", "REB_ALLOWED", "OPP_REB_PER_GAME"]
            opp_ast_candidates = ["OPP_AST", "AST_ALLOWED", "OPP_AST_PER_GAME"]

            def _pick(cands):
                for k in cands:
                    if k in cols:
                        return cols[k]
                return None

            opp_pts_col = _pick(opp_pts_candidates)
            opp_reb_col = _pick(opp_reb_candidates)
            opp_ast_col = _pick(opp_ast_candidates)

            # Fallback: Base measure provides OPP_* in many seasons
            if not all([opp_pts_col, opp_reb_col, opp_ast_col]):
                dash2 = leaguedashteamstats.LeagueDashTeamStats(
                    season=season,
                    per_mode_detailed=per_mode,
                    measure_type_detailed_defense="Base",
                    timeout=30
                )
                df2 = dash2.get_data_frames()[0].copy()
                cols2 = {c.upper(): c for c in df2.columns}
                opp_pts_col = opp_pts_col or cols2.get("OPP_PTS")
                opp_reb_col = opp_reb_col or cols2.get("OPP_REB")
                opp_ast_col = opp_ast_col or cols2.get("OPP_AST")
                if all([opp_pts_col, opp_reb_col, opp_ast_col]):
                    df = df2
                    cols = cols2

            if not all([opp_pts_col, opp_reb_col, opp_ast_col]):
                return pd.DataFrame()

            need = ["TEAM_ID", "TEAM_NAME", opp_pts_col, opp_reb_col, opp_ast_col]
            out = df[need].copy()
            out.rename(
                columns={
                    opp_pts_col: "OPP_PTS",
                    opp_reb_col: "OPP_REB",
                    opp_ast_col: "OPP_AST",
                },
                inplace=True,
            )
            return out
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame()

def compute_opp_multipliers(season: str, opponent_team_id: int) -> Dict[str, float]:
    """
    Returns matchup multipliers: opp_allowed / league_mean for PTS/REB/AST.
    If anything goes wrong, returns 1.0s so the model still runs.
    """
    try:
        opp_df = fetch_opponent_allowed(season)
        if opp_df.empty or opponent_team_id is None:
            return {"PTS": 1.0, "REB": 1.0, "AST": 1.0}

        league_means = opp_df[["OPP_PTS", "OPP_REB", "OPP_AST"]].mean(numeric_only=True)
        row = opp_df.loc[opp_df["TEAM_ID"] == opponent_team_id]
        if row.empty:
            return {"PTS": 1.0, "REB": 1.0, "AST": 1.0}

        opp_vals = row.iloc[0]
        m_pts = float(opp_vals["OPP_PTS"]) / float(league_means["OPP_PTS"]) if league_means["OPP_PTS"] else 1.0
        m_reb = float(opp_vals["OPP_REB"]) / float(league_means["OPP_REB"]) if league_means["OPP_REB"] else 1.0
        m_ast = float(opp_vals["OPP_AST"]) / float(league_means["OPP_AST"]) if league_means["OPP_AST"] else 1.0
        return {"PTS": m_pts, "REB": m_reb, "AST": m_ast}
    except Exception:
        return {"PTS": 1.0, "REB": 1.0, "AST": 1.0}

# ----------------------------
# Weighting & projection
# ----------------------------
def add_recency_and_context_weights(df: pd.DataFrame,
                                    playoff_downweight: float,
                                    half_life_games: float) -> pd.DataFrame:
    """
    Adds:
      - RECENCY_WT: exponential decay by game age (most recent largest)
      - CONTEXT_WT: playoff downweight (if SEASON_TYPE == 'Playoffs')
      - WEIGHT: RECENCY_WT * CONTEXT_WT
    """
    if df.empty:
        return df
    df = df.copy().sort_values("GAME_DATE").reset_index(drop=True)
    n = len(df)
    df["AGE_IDX"] = list(reversed(range(n)))  # 0 = most recent
    ln2 = math.log(2.0)
    df["RECENCY_WT"] = (-(df["AGE_IDX"] / max(half_life_games, 1e-6)) * ln2).apply(math.exp)
    df["CONTEXT_WT"] = df["SEASON_TYPE"].apply(lambda x: playoff_downweight if x == "Playoffs" else 1.0)
    df["WEIGHT"] = df["RECENCY_WT"] * df["CONTEXT_WT"]
    return df

def weighted_projection(last_df: pd.DataFrame, stat_col: str, minutes_col: str = "MIN") -> Tuple[float, float, float, float]:
    """
    Weighted per-minute rate * weighted expected minutes.
    Returns: (mean, theta, rate_recent_weighted, minutes_exp_weighted)
    """
    if last_df.empty:
        return 0.0, 1e9, 0.0, 0.0
    mins = last_df[minutes_col].clip(lower=1e-6)
    w = last_df["WEIGHT"].clip(lower=0.0)

    rate_w = ((last_df[stat_col] / mins) * w).sum() / max(w.sum(), 1e-9)
    mins_w = (mins * w).sum() / max(w.sum(), 1e-9)
    mean_val = max(0.0, rate_w * mins_w)

    if len(last_df) >= 2:
        var_val = last_df[stat_col].var(ddof=1)
        mean_for_theta = last_df[stat_col].mean()
    else:
        var_val = max(mean_val, 1.0)
        mean_for_theta = mean_val

    theta = fit_theta(mean_for_theta, var_val)
    return mean_val, theta, rate_w, mins_w

# ----------------------------
# Prop line suggestion engine
# ----------------------------
def suggest_lines(mean: float,
                  theta: float,
                  base_step: float = 0.5,
                  span: float = 6.0,
                  min_edge: float = 0.05) -> List[Dict]:
    """
    Scan candidate lines around the mean (±span) at 'base_step' and return
    suggestions where max(P_over, P_under) - 0.5 >= min_edge.
    """
    if mean <= 0:
        return []
    center = round(mean * 2) / 2.0
    lo = center - span
    hi = center + span
    candidates = []
    x = lo
    while x <= hi + 1e-9:
        p_over, p_under = over_under_prob(x, mean, theta)
        best_side = "Over" if p_over >= p_under else "Under"
        best_p = max(p_over, p_under)
        edge = best_p - 0.5
        if edge >= min_edge:
            candidates.append({
                "line": round(x, 1),
                "side": best_side,
                "p": round(best_p, 3),
                "edge": round(edge, 3)
            })
        x += base_step
    candidates.sort(key=lambda d: (-d["edge"], -d["p"], d["line"]))
    return candidates[:10]

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # ===== CONFIG =====
    PLAYER_NAME = "Jimmy Butler"

    # Seasons/Types to include in the window
    SEASONS = ["2025-26", "2024-25"]
    SEASON_TYPES = ["Regular Season"]

    # Recency & weighting
    LAST_N = 82
    HALFLIFE_GAMES = 10
    PLAYOFF_DOWNWEIGHT = 0.70

    # Opponent scaling
    OPP_TEAM_ABBR = "PHX"         # e.g., "LAL", "BOS", "DEN"; set to None to skip
    OPP_MULT_SEASON = "2025-26"   # season to use for opponent-allowed stats

    # Prop lines to evaluate
    POINTS_LINE = 20.5
    REBOUNDS_LINE = 1.5
    ASSISTS_LINE = 2.5
    PRA_LINE = 21.5

    # Suggestions
    ENABLE_SUGGESTIONS = True
    SUGG_MIN_EDGE = 0.06   # ≥ 6% edge
    SUGG_SPAN = 6.0
    SUGG_STEP = 0.5

    # ===== Fetch & combine =====
    pid = find_player_id(PLAYER_NAME)
    logs_all = fetch_multi_season_logs(pid, SEASONS, SEASON_TYPES)
    if logs_all.empty:
        raise SystemExit("No logs returned. Check seasons/season types or try another player.")

    # Take last-N across combined logs
    logs_all = logs_all.sort_values("GAME_DATE").reset_index(drop=True)
    lastN = logs_all.tail(LAST_N).copy()

    # Add weights (recency + playoffs context)
    lastN = add_recency_and_context_weights(lastN, PLAYOFF_DOWNWEIGHT, HALFLIFE_GAMES)

    # ===== Base projections (pre-opponent) =====
    mean_pts, theta_pts, rate_pts, exp_min_pts = weighted_projection(lastN, "PTS")
    mean_reb, theta_reb, rate_reb, exp_min_reb = weighted_projection(lastN, "REB")
    mean_ast, theta_ast, rate_ast, exp_min_ast = weighted_projection(lastN, "AST")

    # PRA (preserves covariance within window)
    lastN["PRA"] = lastN["PTS"] + lastN["REB"] + lastN["AST"]
    mean_pra, theta_pra, rate_pra, exp_min_pra = weighted_projection(lastN, "PRA")

    # ===== Opponent difficulty multipliers =====
    if OPP_TEAM_ABBR:
        opp_id = find_team_id_by_abbr(OPP_TEAM_ABBR)
    else:
        opp_id = None
    mult = compute_opp_multipliers(OPP_MULT_SEASON, opp_id) if opp_id else {"PTS": 1.0, "REB": 1.0, "AST": 1.0}

    # Scale means by matchup multipliers
    mean_pts_adj = mean_pts * mult["PTS"]
    mean_reb_adj = mean_reb * mult["REB"]
    mean_ast_adj = mean_ast * mult["AST"]
    mean_pra_adj = mean_pts_adj + mean_reb_adj + mean_ast_adj

    # ===== Probabilities at your lines =====
    p_over_pts, p_under_pts = over_under_prob(POINTS_LINE, mean_pts_adj, theta_pts)
    p_over_reb, p_under_reb = over_under_prob(REBOUNDS_LINE, mean_reb_adj, theta_reb)
    p_over_ast, p_under_ast = over_under_prob(ASSISTS_LINE, mean_ast_adj, theta_ast)
    p_over_pra, p_under_pra = over_under_prob(PRA_LINE, mean_pra_adj, theta_pra)

    # ===== Output =====
    print(f"\n=== NBA Props Model (Multi-season, Recency-Weighted) ===")
    print(f"Player: {PLAYER_NAME}")
    print(f"Seasons: {SEASONS} | Season Types: {SEASON_TYPES}")
    print(f"Window: Last {LAST_N} games | Half-life: {HALFLIFE_GAMES} | Playoffs downweight: {PLAYOFF_DOWNWEIGHT}")
    print(f"Opponent scaling season: {OPP_MULT_SEASON} | Opponent: {OPP_TEAM_ABBR or 'None'}")
    print(f"Opponent multipliers  →  PTS:{mult['PTS']:.3f}  REB:{mult['REB']:.3f}  AST:{mult['AST']:.3f}")

    print("\n-- Points --")
    print(f"Weighted rate (PTS/min): {rate_pts:.3f} | Weighted exp minutes: {exp_min_pts:.1f}")
    print(f"Projected mean (adj): {mean_pts_adj:.2f} | Theta: {theta_pts:.1f}")
    print(f"P(Over {POINTS_LINE}): {p_over_pts:.3f} | P(Under): {p_under_pts:.3f}")

    print("\n-- Rebounds --")
    print(f"Weighted rate (REB/min): {rate_reb:.3f} | Weighted exp minutes: {exp_min_reb:.1f}")
    print(f"Projected mean (adj): {mean_reb_adj:.2f} | Theta: {theta_reb:.1f}")
    print(f"P(Over {REBOUNDS_LINE}): {p_over_reb:.3f} | P(Under): {p_under_reb:.3f}")

    print("\n-- Assists --")
    print(f"Weighted rate (AST/min): {rate_ast:.3f} | Weighted exp minutes: {exp_min_ast:.1f}")
    print(f"Projected mean (adj): {mean_ast_adj:.2f} | Theta: {theta_ast:.1f}")
    print(f"P(Over {ASSISTS_LINE}): {p_over_ast:.3f} | P(Under): {p_under_ast:.3f}")

    print("\n-- PRA --")
    print(f"(Component means → P:{mean_pts_adj:.2f} + R:{mean_reb_adj:.2f} + A:{mean_ast_adj:.2f})")
    print(f"Projected mean PRA (adj): {mean_pra_adj:.2f} | Theta: {theta_pra:.1f}")
    print(f"P(Over {PRA_LINE}): {p_over_pra:.3f} | P(Under): {p_under_pra:.3f}")

    print("\n--- Recent games (most recent last) ---")
    cols = ["GAME_DATE", "SEASON", "SEASON_TYPE", "MATCHUP", "MIN", "PTS", "REB", "AST", "WEIGHT"]
    print(lastN[cols].to_string(index=False, formatters={"WEIGHT": lambda x: f"{x:.3f}"}))


        # ===== CSV-ready output for pp_entry_builder_nba.py =====
    def _csv_row(stat: str, line: float, p_over: float, p_under: float) -> str:
        # decide side, p, edge
        if p_over >= p_under:
            side = "Over"
            p = p_over
        else:
            side = "Under"
            p = p_under
        edge = p - 0.5
        group = f"{OPP_TEAM_ABBR} matchup" if OPP_TEAM_ABBR else ""
        # CSV schema: player,stat,line,side,p,edge,group
        return f'"{PLAYER_NAME}",{stat},{line},{side},{p:.3f},{edge:.3f},"{group}"'

    print("\n=== Copy this into your CSV (props.csv / prop_bet.csv) ===")
    print("player,stat,line,side,p,edge,group")
    print(_csv_row("PTS", POINTS_LINE, p_over_pts, p_under_pts))
    print(_csv_row("REB", REBOUNDS_LINE, p_over_reb, p_under_reb))
    print(_csv_row("AST", ASSISTS_LINE, p_over_ast, p_under_ast))
    print(_csv_row("PRA", PRA_LINE, p_over_pra, p_under_pra))
    print("=== End CSV block ===\n")

    # ===== Suggestions
    if ENABLE_SUGGESTIONS:
        print("\n=== Suggested Prop Lines (edge ≥ {:.0f}%) ===".format(SUGG_MIN_EDGE * 100))
        sugg_pts = suggest_lines(mean_pts_adj, theta_pts, base_step=SUGG_STEP, span=SUGG_SPAN, min_edge=SUGG_MIN_EDGE)
        sugg_reb = suggest_lines(mean_reb_adj, theta_reb, base_step=SUGG_STEP, span=SUGG_SPAN, min_edge=SUGG_MIN_EDGE)
        sugg_ast = suggest_lines(mean_ast_adj, theta_ast, base_step=SUGG_STEP, span=SUGG_SPAN, min_edge=SUGG_MIN_EDGE)
        sugg_pra = suggest_lines(mean_pra_adj, theta_pra, base_step=SUGG_STEP, span=SUGG_SPAN, min_edge=SUGG_MIN_EDGE)

        def _print_suggs(label, items):
            if not items:
                print(f"{label}: (no strong edges)")
                return
            print(f"{label}:")
            for it in items:
                print(f"  {it['side']} {it['line']:<5}  p={it['p']:.3f}  edge={it['edge']:.3f}")

        _print_suggs("Points", sugg_pts)
        _print_suggs("Rebounds", sugg_reb)
        _print_suggs("Assists", sugg_ast)
        _print_suggs("PRA", sugg_pra)
