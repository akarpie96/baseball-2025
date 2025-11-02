import math
import requests
import pandas as pd
from datetime import datetime, timezone

BASE = "https://statsapi.mlb.com/api/v1"

# =========================
# Player & team lookups
# =========================
def search_player(name: str):
    """Fuzzy player lookup -> (player_id, full_name, primary_number, current_team_id, throws)."""
    r = requests.get(f"{BASE}/people/search", params={"names": name, "sportId": 1}, timeout=30)
    r.raise_for_status()
    people = (r.json() or {}).get("people", [])
    if not people:
        r = requests.get(f"{BASE}/search/people", params={"q": name}, timeout=30)
        r.raise_for_status()
        people = (r.json() or {}).get("people", [])
        if not people:
            raise ValueError(f"No player found for: {name}")
    p = people[0]
    pid = p["id"]
    full = p["fullName"]
    number = p.get("primaryNumber")
    current_team_id = (p.get("currentTeam") or {}).get("id")
    throws = p.get("pitchHand", {}).get("code")  # 'R'/'L'
    return pid, full, number, current_team_id, throws

def search_player_by_id(pid: int):
    r = requests.get(f"{BASE}/people/{pid}", timeout=30)
    r.raise_for_status()
    people = (r.json() or {}).get("people", [])
    if not people:
        return None, None, None, None, None
    p = people[0]
    full = p.get("fullName")
    number = p.get("primaryNumber")
    team_id = (p.get("currentTeam") or {}).get("id")
    throws = p.get("pitchHand", {}).get("code")
    return pid, full, number, team_id, throws

# =========================
# Schedule / next game
# =========================
def next_game_for_team(team_id: int):
    """Find next scheduled game (>= now)."""
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    r = requests.get(
        f"{BASE}/schedule",
        params={"teamId": team_id, "sportId": 1, "startDate": now_iso, "endDate": now_iso},
        timeout=30,
    )
    r.raise_for_status()
    js = r.json()
    dates = js.get("dates", [])
    if not dates:
        r = requests.get(f"{BASE}/schedule", params={"teamId": team_id, "sportId": 1}, timeout=30)
        r.raise_for_status()
        js = r.json()
        dates = js.get("dates", [])
    for d in dates:
        for g in d.get("games", []):
            state = (g.get("status") or {}).get("abstractGameState")
            if state in {"Preview", "Live"}:
                home = g["teams"]["home"]["team"]["id"]
                away = g["teams"]["away"]["team"]["id"]
                opp = away if home == team_id else home
                venue = g.get("venue", {})
                return {
                    "gamePk": g["gamePk"],
                    "date": g.get("gameDate"),
                    "opponent_team_id": opp,
                    "opponent_team_name": (g["teams"]["away"]["team"]["name"] if home == team_id else g["teams"]["home"]["team"]["name"]),
                    "venue_id": venue.get("id"),
                    "venue_name": venue.get("name"),
                    "home_away": "home" if home == team_id else "away",
                }
    return None

# =========================
# Pitcher logs (regular & postseason)
# =========================
def pitcher_game_logs(player_id: int, season: int):
    url = f"{BASE}/people/{player_id}/stats"
    params = {"stats": "gameLog", "group": "pitching", "season": season}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    blocks = js.get("stats") or []
    if not blocks:
        return pd.DataFrame()
    splits = blocks[0].get("splits") or []
    if not splits:
        return pd.DataFrame()
    rows = []
    for s in splits:
        stat = s.get("stat", {}) or {}
        ip_raw = stat.get("inningsPitched")
        ip = _ip_to_decimal(ip_raw) if ip_raw else 0.0
        rows.append({
            "date": s.get("date"),
            "team": (s.get("team") or {}).get("name"),
            "opponent": (s.get("opponent") or {}).get("name"),
            "strikeOuts": _to_float(stat.get("strikeOuts")),
            "battersFaced": _to_float(stat.get("battersFaced")),
            "inningsPitched": ip,
            "hitsAllowed": _to_float(stat.get("hits")),
            "runs": _to_float(stat.get("runs")),
            "earnedRuns": _to_float(stat.get("earnedRuns")),
            "pitchesThrown": _to_float(stat.get("numberOfPitches")),
            "postseason": False,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df

def pitcher_postseason_logs(player_id: int, season: int):
    url = f"{BASE}/people/{player_id}/stats"
    params = {"stats": "gameLog", "group": "pitching", "season": season, "gameType": "P"}  # playoffs
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    blocks = js.get("stats") or []
    if not blocks:
        return pd.DataFrame()
    splits = blocks[0].get("splits") or []
    if not splits:
        return pd.DataFrame()
    rows = []
    for s in splits:
        stat = s.get("stat", {}) or {}
        ip_raw = stat.get("inningsPitched")
        ip = _ip_to_decimal(ip_raw) if ip_raw else 0.0
        rows.append({
            "date": s.get("date"),
            "team": (s.get("team") or {}).get("name"),
            "opponent": (s.get("opponent") or {}).get("name"),
            "strikeOuts": _to_float(stat.get("strikeOuts")),
            "battersFaced": _to_float(stat.get("battersFaced")),
            "inningsPitched": ip,
            "hitsAllowed": _to_float(stat.get("hits")),
            "runs": _to_float(stat.get("runs")),
            "earnedRuns": _to_float(stat.get("earnedRuns")),
            "pitchesThrown": _to_float(stat.get("numberOfPitches")),
            "postseason": True,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df

# =========================
# Opponent / league summaries
# =========================
def team_batting_summary(team_id: int, season: int):
    """Team batting (for K% and contact proxy via H/PA)."""
    url = f"{BASE}/teams/stats"
    params = {"group": "hitting", "season": season, "sportId": 1, "teamId": team_id}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    splits = (js.get("stats") or [{}])[0].get("splits", [])
    if not splits:
        return {}
    s = splits[0].get("stat", {}) or {}
    return {
        "strikeOuts": _to_float(s.get("strikeOuts")),
        "plateAppearances": _to_float(s.get("plateAppearances")),
        "hits": _to_float(s.get("hits")),
    }

def league_batting_summary(season: int):
    """League aggregates to normalize opponent tendencies."""
    url = f"{BASE}/teams/stats"
    params = {"group": "hitting", "season": season, "sportId": 1}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    splits = (js.get("stats") or [{}])[0].get("splits", [])
    tot_K, tot_PA, tot_H = 0.0, 0.0, 0.0
    for sp in splits:
        st = sp.get("stat", {}) or {}
        tot_K += _to_float(st.get("strikeOuts"))
        tot_PA += _to_float(st.get("plateAppearances"))
        tot_H += _to_float(st.get("hits"))
    return {"strikeOuts": tot_K, "plateAppearances": tot_PA, "hits": tot_H}

# =========================
# Projection: Ks + Outs + Hits Allowed
# =========================
def project_pitcher_strikeouts(player_id: int, season: int, last_n: int, opp_team_id: int = None, playoff_downweight: float = 0.75):
    """Projected strikeouts (lambda) with postseason downweight and opponent K-rate multiplier."""
    reg = pitcher_game_logs(player_id, season)
    post = pitcher_postseason_logs(player_id, season)
    if reg.empty and post.empty:
        raise RuntimeError("No pitching game logs found (regular or postseason).")

    logs = pd.concat([reg, post], ignore_index=True) if not post.empty else reg.copy()
    logs["date"] = pd.to_datetime(logs["date"])
    logs = logs.sort_values("date").reset_index(drop=True)
    logs["weight"] = 1.0
    if not post.empty:
        logs.loc[logs["postseason"] == True, "weight"] = playoff_downweight

    # Last-N window for modeling
    recent = logs.tail(last_n) if len(logs) >= last_n else logs.copy()

    # Weighted helpers
    def wsum(series, weights): return float((series.fillna(0) * weights.fillna(0)).sum())
    def safe_div(a, b): return (a / b) if (b and b > 0) else 0.0

    # Ks per IP (recent + season)
    k_ip_recent = safe_div(wsum(recent["strikeOuts"], recent["weight"]), wsum(recent["inningsPitched"], recent["weight"]))
    k_ip_season = safe_div(wsum(logs["strikeOuts"], logs["weight"]), wsum(logs["inningsPitched"], logs["weight"]))
    k_ip = 0.65 * k_ip_recent + 0.35 * k_ip_season

    # Expected IP (favor recent if enough starts)
    exp_ip_recent = safe_div(wsum(recent["inningsPitched"], recent["weight"]), recent["weight"].sum() if recent["weight"].sum() else 0.0)
    exp_ip_season = safe_div(wsum(logs["inningsPitched"], logs["weight"]), logs["weight"].sum() if logs["weight"].sum() else 0.0)
    exp_ip = exp_ip_recent if len(recent) >= 3 else exp_ip_season

    # Opponent K-rate multiplier vs league
    if opp_team_id is None:
        _, _, _, team_id, _ = search_player_by_id(player_id)
        nxt = next_game_for_team(team_id) if team_id else None
        if nxt:
            opp_team_id = nxt["opponent_team_id"]
    opp_mult = 1.0
    if opp_team_id:
        opp = team_batting_summary(opp_team_id, season)
        lge = league_batting_summary(season)
        opp_k_pa = _ratio(opp.get("strikeOuts"), opp.get("plateAppearances"))
        lge_k_pa = _ratio(lge.get("strikeOuts"), lge.get("plateAppearances"))
        if lge_k_pa > 0:
            opp_mult = opp_k_pa / lge_k_pa

    lam = max(0.0, k_ip * exp_ip * opp_mult)

    # Over-dispersion from recent Ks (unweighted)
    ks = recent["strikeOuts"].astype(float).values
    mean_k = float(ks.mean()) if len(ks) else lam
    var_k = float(ks.var(ddof=1)) if len(ks) >= 2 else max(lam, 1.0)
    theta = _fit_theta(mean_k, var_k)
    var_model = mean_k + (mean_k ** 2) / theta if theta and theta > 0 else mean_k

    # last-N hits/opponents for reporting
    lastN = logs.tail(last_n)
    avg_hits_lastN = float(lastN["hitsAllowed"].mean()) if not lastN.empty else 0.0
    total_hits_lastN = float(lastN["hitsAllowed"].sum()) if not lastN.empty else 0.0
    opp_list_lastN = [str(x) for x in lastN["opponent"].fillna("").tolist()]

    return {
        "lambda": lam,
        "exp_ip": exp_ip,
        "k_per_ip_recent": k_ip_recent,
        "k_per_ip_season": k_ip_season,
        "opp_multiplier": opp_mult,
        "mean_recent": mean_k,
        "var_recent": var_k,
        "theta": theta,
        "var_model": var_model,
        "logs": logs,
        "recent_window": recent,
        "playoff_downweight": playoff_downweight,
        "lastN_avg_hits": avg_hits_lastN,
        "lastN_total_hits": total_hits_lastN,
        "lastN_opponents": opp_list_lastN,
    }

def project_pitcher_outs(player_id: int, season: int, last_n: int, playoff_downweight: float = 0.75):
    """Projected pitching outs using same IP expectation; NB variance from recent outs."""
    reg = pitcher_game_logs(player_id, season)
    post = pitcher_postseason_logs(player_id, season)
    if reg.empty and post.empty:
        raise RuntimeError("No pitching game logs found (regular or postseason).")

    logs = pd.concat([reg, post], ignore_index=True) if not post.empty else reg.copy()
    logs["date"] = pd.to_datetime(logs["date"])
    logs = logs.sort_values("date").reset_index(drop=True)
    logs["weight"] = 1.0
    if not post.empty:
        logs.loc[logs["postseason"] == True, "weight"] = playoff_downweight

    recent = logs.tail(last_n) if len(logs) >= last_n else logs.copy()

    def wsum(series, weights): return float((series.fillna(0) * weights.fillna(0)).sum())
    def safe_div(a, b): return (a / b) if (b and b > 0) else 0.0

    exp_ip_recent = safe_div(wsum(recent["inningsPitched"], recent["weight"]), recent["weight"].sum() if recent["weight"].sum() else 0.0)
    exp_ip_season = safe_div(wsum(logs["inningsPitched"], logs["weight"]), logs["weight"].sum() if logs["weight"].sum() else 0.0)
    exp_ip = exp_ip_recent if len(recent) >= 3 else exp_ip_season

    exp_outs = max(0.0, 3.0 * exp_ip)

    outs_recent = (recent["inningsPitched"] * 3.0).round().astype(int).values
    mean_o = float(outs_recent.mean()) if len(outs_recent) else exp_outs
    var_o = float(outs_recent.var(ddof=1)) if len(outs_recent) >= 2 else max(exp_outs, 1.0)
    theta = _fit_theta(mean_o, var_o)

    return {
        "expected_outs": exp_outs,
        "theta": theta,
        "mean_recent": mean_o,
        "var_recent": var_o,
        "recent_outs_array": outs_recent,
        "exp_ip": exp_ip,
        "logs": logs,
        "recent_window": recent,
        "playoff_downweight": playoff_downweight,
    }

def project_pitcher_hits_allowed(player_id: int, season: int, last_n: int, opp_team_id: int = None, playoff_downweight: float = 0.75):
    """
    Project Hits Allowed:
      mean_hits = (H/IP weighted recent+season) * expected_IP * opponent_contact_multiplier
    opponent_contact_multiplier = (opp H/PA) / (league H/PA)
    NB variance fit from recent hits-allowed counts (unweighted).
    """
    reg = pitcher_game_logs(player_id, season)
    post = pitcher_postseason_logs(player_id, season)
    if reg.empty and post.empty:
        raise RuntimeError("No pitching game logs found (regular or postseason).")

    logs = pd.concat([reg, post], ignore_index=True) if not post.empty else reg.copy()
    logs["date"] = pd.to_datetime(logs["date"])
    logs = logs.sort_values("date").reset_index(drop=True)
    logs["weight"] = 1.0
    if not post.empty:
        logs.loc[logs["postseason"] == True, "weight"] = playoff_downweight

    recent = logs.tail(last_n) if len(logs) >= last_n else logs.copy()

    def wsum(series, weights): return float((series.fillna(0) * weights.fillna(0)).sum())
    def safe_div(a, b): return (a / b) if (b and b > 0) else 0.0

    # H/IP (recent + season)
    hip_recent = safe_div(wsum(recent["hitsAllowed"], recent["weight"]), wsum(recent["inningsPitched"], recent["weight"]))
    hip_season = safe_div(wsum(logs["hitsAllowed"], logs["weight"]), wsum(logs["inningsPitched"], logs["weight"]))
    hip = 0.65 * hip_recent + 0.35 * hip_season

    # Expected IP
    exp_ip_recent = safe_div(wsum(recent["inningsPitched"], recent["weight"]), recent["weight"].sum() if recent["weight"].sum() else 0.0)
    exp_ip_season = safe_div(wsum(logs["inningsPitched"], logs["weight"]), logs["weight"].sum() if logs["weight"].sum() else 0.0)
    exp_ip = exp_ip_recent if len(recent) >= 3 else exp_ip_season

    # Opponent contact multiplier via H/PA
    if opp_team_id is None:
        _, _, _, team_id, _ = search_player_by_id(player_id)
        nxt = next_game_for_team(team_id) if team_id else None
        if nxt:
            opp_team_id = nxt["opponent_team_id"]

    contact_mult = 1.0
    if opp_team_id:
        opp = team_batting_summary(opp_team_id, season)
        lge = league_batting_summary(season)
        opp_h_pa = _ratio(opp.get("hits"), opp.get("plateAppearances"))
        lge_h_pa = _ratio(lge.get("hits"), lge.get("plateAppearances"))
        if lge_h_pa > 0:
            contact_mult = opp_h_pa / lge_h_pa

    mean_hits = max(0.0, hip * exp_ip * contact_mult)

    # Fit dispersion from recent hits-allowed (unweighted integers)
    hits_recent = recent["hitsAllowed"].round().astype(int).values
    mean_h = float(hits_recent.mean()) if len(hits_recent) else mean_hits
    var_h = float(hits_recent.var(ddof=1)) if len(hits_recent) >= 2 else max(mean_hits, 1.0)
    theta = _fit_theta(mean_h, var_h)

    return {
        "expected_hits": mean_hits,
        "theta": theta,
        "mean_recent": mean_h,
        "var_recent": var_h,
        "contact_multiplier": contact_mult,
        "hits_per_ip_recent": hip_recent,
        "hits_per_ip_season": hip_season,
        "exp_ip": exp_ip,
        "recent_window_len": len(recent),
    }

# =========================
# Probability helpers
# =========================
def over_under_prob(line: float, lam: float, theta: float):
    """Ks O/U probability via NB (fallback Poisson)."""
    if theta is None or theta <= 0 or not math.isfinite(theta) or theta > 1e8:
        return _poisson_over_under(line, lam)
    r = float(theta)
    p = r / (r + lam)
    k_floor = math.floor(line)
    cdf_k = _nb_cdf(k_floor, r, p)
    return 1.0 - cdf_k, cdf_k

def over_under_prob_count(line: float, mean_count: float, theta: float):
    """Generic count O/U probability (for Outs or Hits)."""
    if theta is None or theta <= 0 or not math.isfinite(theta) or theta > 1e8:
        return _poisson_over_under(line, mean_count)
    r = float(theta)
    p = r / (r + mean_count)
    k_floor = math.floor(line)
    cdf_k = _nb_cdf(k_floor, r, p)
    return 1.0 - cdf_k, cdf_k

# =========================
# Math helpers
# =========================
def _ip_to_decimal(ip_str: str):
    try:
        if ip_str is None:
            return 0.0
        parts = str(ip_str).split(".")
        whole = int(parts[0]); frac = int(parts[1]) if len(parts) > 1 else 0
        return whole + (frac / 3.0)
    except Exception:
        return 0.0

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def _ratio(a, b):
    a = _to_float(a); b = _to_float(b)
    return (a / b) if b and b > 0 else 0.0

def _fit_theta(mean_val: float, var_val: float):
    if mean_val <= 0 or var_val <= mean_val:
        return 1e9  # ~Poisson
    return (mean_val ** 2) / (var_val - mean_val)

def _nb_pmf(k: int, r: float, p: float):
    lgamma = math.lgamma
    logC = lgamma(k + r) - lgamma(r) - lgamma(k + 1)
    logpmf = logC + k * math.log(1 - p) + r * math.log(p)
    return math.exp(logpmf)

def _nb_cdf(k: int, r: float, p: float):
    if k < 0: return 0.0
    s = 0.0
    for x in range(0, k + 1):
        s += _nb_pmf(x, r, p)
    return min(max(s, 0.0), 1.0)

def _poisson_pmf(k: int, lam: float):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def _poisson_cdf(k: int, lam: float):
    if k < 0: return 0.0
    s = 0.0
    for x in range(0, k + 1):
        s += _poisson_pmf(x, lam)
    return min(max(s, 0.0), 1.0)

def _poisson_over_under(line: float, lam: float):
    k_floor = math.floor(line)
    cdf = _poisson_cdf(k_floor, lam)
    return 1.0 - cdf, cdf

# =========================
# ========== CONFIG / RUN ==========
# =========================
if __name__ == "__main__":
    # ---- Last-N used for modeling and summaries ----
    LAST_N = 10

    # ---- Your prop inputs ----
    PITCHER_NAME = "Max Scherzer"
    SEASON = 2025
    K_LINE = 6.5
    OUTS_LINE = 8.5
    HITS_LINE = 3.5
    PLAYOFF_DOWNWEIGHT = 0.75

    pid, full, num, team_id, throws = search_player(PITCHER_NAME)
    print(f"Found: {full} (ID {pid}), TeamID={team_id}, Throws={throws}")

    nxt = next_game_for_team(team_id) if team_id else None
    if nxt:
        opp_id = nxt["opponent_team_id"]
        opp_name = nxt.get("opponent_team_name")
        print(f"Next game vs {opp_name} (teamId {opp_id}) at {nxt.get('venue_name')} on {nxt.get('date')}")
    else:
        opp_id = None
        opp_name = None
        print("Next game not found; proceeding without opponent adjustment.")

    # ===== Ks projection =====
    proj_k = project_pitcher_strikeouts(
        pid, season=SEASON, last_n=LAST_N, opp_team_id=opp_id, playoff_downweight=PLAYOFF_DOWNWEIGHT
    )
    lam_k = proj_k["lambda"]; theta_k = proj_k["theta"]
    p_over_k, p_under_k = over_under_prob(K_LINE, lam_k, theta_k)

    # ===== Outs projection =====
    proj_outs = project_pitcher_outs(
        pid, season=SEASON, last_n=LAST_N, playoff_downweight=PLAYOFF_DOWNWEIGHT
    )
    mu_outs = proj_outs["expected_outs"]; theta_outs = proj_outs["theta"]
    p_over_outs, p_under_outs = over_under_prob_count(OUTS_LINE, mu_outs, theta_outs)

    # ===== Hits Allowed projection =====
    proj_hits = project_pitcher_hits_allowed(
        pid, season=SEASON, last_n=LAST_N, opp_team_id=opp_id, playoff_downweight=PLAYOFF_DOWNWEIGHT
    )
    mu_hits = proj_hits["expected_hits"]; theta_hits = proj_hits["theta"]
    p_over_hits, p_under_hits = over_under_prob_count(HITS_LINE, mu_hits, theta_hits)

    # ===== OUTPUT =====
    print("\n--- Projection Summary (Strikeouts) ---")
    print(f"Last N starts (N):           {LAST_N}")
    print(f"Playoff downweight:          {proj_k['playoff_downweight']:.2f}")
    print(f"Expected IP (next):          {proj_k['exp_ip']:.2f}")
    print(f"K/IP recent (weighted):      {proj_k['k_per_ip_recent']:.3f}")
    print(f"K/IP season (weighted):      {proj_k['k_per_ip_season']:.3f}")
    print(f"Opponent multiplier (Ks):    {proj_k['opp_multiplier']:.3f}")
    print(f"Projected Ks (λ):            {lam_k:.2f}")
    print(f"Fitted theta (NB shape):     {theta_k:.2f}")
    print(f"P(Over {K_LINE} Ks):         {p_over_k:.3f}")
    print(f"P(Under {K_LINE} Ks):        {p_under_k:.3f}")

    print(f"\n--- Projection Summary (Pitching Outs) ---")
    print(f"Expected IP (next):          {proj_outs['exp_ip']:.2f}")
    print(f"Expected Outs (μ):           {mu_outs:.2f}")
    print(f"Fitted theta (NB shape):     {theta_outs:.2f}")
    print(f"P(Over {OUTS_LINE} Outs):    {p_over_outs:.3f}")
    print(f"P(Under {OUTS_LINE} Outs):   {p_under_outs:.3f}")

    print(f"\n--- Projection Summary (Hits Allowed) ---")
    print(f"Expected IP (next):          {proj_hits['exp_ip']:.2f}")
    print(f"H/IP recent (weighted):      {proj_hits['hits_per_ip_recent']:.3f}")
    print(f"H/IP season (weighted):      {proj_hits['hits_per_ip_season']:.3f}")
    print(f"Opponent contact mult:       {proj_hits['contact_multiplier']:.3f}")
    print(f"Expected Hits (μ):           {mu_hits:.2f}")
    print(f"Fitted theta (NB shape):     {theta_hits:.2f}")
    print(f"P(Over {HITS_LINE} Hits):    {p_over_hits:.3f}")
    print(f"P(Under {HITS_LINE} Hits):   {p_under_hits:.3f}")

    print(f"\n--- Recent Performance (Last {LAST_N} Starts) ---")
    print(f"Avg Hits Allowed:            {proj_k['lastN_avg_hits']:.2f}")
    print(f"Total Hits Allowed:          {proj_k['lastN_total_hits']:.0f}")
    print("Opponents Faced:             " + ", ".join([o for o in proj_k['lastN_opponents'] if o]))
