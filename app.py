
# ============================================
# STREAMLIT APP: US Elections Explorer
# - Years: 2016 / 2018 / 2020 / 2022 / 2024
# - State map colors: Pres margin OR Avg House margin
# - District hover includes House results + 2026 ratings + optional FEC + optional ACS demographics
# - Ratings Universe view (Cook/Sabato/Inside from 3x 270toWin URLs ONLY)
# - WAR merge-in from your uploaded CSV (per district/year)
# - ACS (American Community Survey) 5-year *Data Profile* via Census Data API:
#   * District-level via “congressional district”
#   * State-level via “state”
#   * Fields (excluding relig/union): race, gender, income, education, age, veteran
#
# FIXES in this version (added):
# 4) Ratings parsing: canonicalize "Toss-up" correctly.
#    Your old normalize used .title() which turns "toss-up" into "Toss-Up",
#    breaking section detection ("Toss-up") and toss-up counting.
#    Now we map ALL rating strings to the exact canonical labels:
#    Likely Dem / Leans Dem / Tilt Dem / Toss-up / Tilt Rep / Leans Rep / Likely Rep
# ============================================

import re, json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="US Elections Explorer", layout="wide")

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ----------------------------
# URLS (2026 ratings) - ONLY these 3
# ----------------------------
URL_COOK_270 = "https://www.270towin.com/2026-house-election/index_show_table.php?map_title=cook-political-report-2026-house-ratings"
URL_SABATO_270 = "https://www.270towin.com/2026-house-election/table/crystal-ball-2026-house-forecast"
URL_INSIDE_270 = "https://www.270towin.com/2026-house-election/table/inside-elections-2026-house-ratings"

# ----------------------------
# DISTRICT SHAPES (Census cartographic boundary)
# 2016/2018/2020 -> CD116 (2010-cycle maps)
# 2022/2024 -> CD118 (post-2020 redistricting)
# ----------------------------
CD_ZIPS = {
    2016: "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_cd116_500k.zip",
    2018: "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_cd116_500k.zip",
    2020: "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_cd116_500k.zip",
    2022: "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_cd118_500k.zip",
    2024: "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_cd118_500k.zip",
}

STATE_FIPS = {
    "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11","FL":"12","GA":"13",
    "HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25",
    "MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36",
    "NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48",
    "UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","PR":"72",
}
FIPS_TO_STATE = {v: k for k, v in STATE_FIPS.items()}

# ----------------------------
# HELPERS
# ----------------------------
DIST_RE = re.compile(r"\b([A-Z]{2}-(?:AL|\d{1,2}))\b", re.I)

# Canonical rating labels we will *always* use
RATING_KEYS = [
    "Likely Dem", "Leans Dem", "Tilt Dem",
    "Toss-up",
    "Tilt Rep", "Leans Rep", "Likely Rep"
]
RATING_SCORE = {
    "Likely Dem": -3,
    "Leans Dem": -2,
    "Tilt Dem": -1,
    "Toss-up": 0,
    "Tilt Rep": 1,
    "Leans Rep": 2,
    "Likely Rep": 3,
}

# --- FIX: canonicalize rating strings (prevents Toss-Up vs Toss-up bugs) ---
_CANON_RATINGS = {
    "likely dem": "Likely Dem",
    "leans dem":  "Leans Dem",
    "tilt dem":   "Tilt Dem",
    "toss-up":    "Toss-up",
    "tilt rep":   "Tilt Rep",
    "leans rep":  "Leans Rep",
    "likely rep": "Likely Rep",
}

def normalize_rating_label(x) -> str:
    """
    Canonicalize pundit ratings so parsing + counting uses the same spellings.
    Critical fix: NEVER allow "Toss-Up" to appear (Title() creates that).
    """
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""

    # normalize whitespace/dashes first
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")  # en/em dash -> hyphen

    s_low = s.lower()
    s_low = s_low.replace("tossup", "toss-up").replace("toss up", "toss-up")
    s_low = re.sub(r"\s+", " ", s_low).strip()

    # Map by prefix so "Toss-up (10)" and "Toss-up/..." still canonicalize.
    for k, v in _CANON_RATINGS.items():
        if s_low.startswith(k):
            return v

    # fallback: try to coerce to one of our known labels
    out = s_low.title()
    out = out.replace("Toss Up", "Toss-up").replace("Toss-Up", "Toss-up")
    return out

def is_tossup(x):
    return normalize_rating_label(x) == "Toss-up"

def rating_side(label: str) -> str:
    lab = normalize_rating_label(label)
    if lab == "Toss-up":
        return "Toss-up"
    if "Dem" in lab:
        return "Dem"
    if "Rep" in lab:
        return "Rep"
    return ""

def rating_score(label: str):
    lab = normalize_rating_label(label)
    return RATING_SCORE.get(lab, np.nan)

def consensus_label_from_avgscore(s):
    if pd.isna(s):
        return ""
    if s <= -2.5: return "Likely Dem"
    if s <= -1.5: return "Leans Dem"
    if s <= -0.5: return "Tilt Dem"
    if s < 0.5:   return "Toss-up"
    if s < 1.5:   return "Tilt Rep"
    if s < 2.5:   return "Leans Rep"
    return "Likely Rep"

def safe_plot_col(series):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s.apply(lambda x: None if (x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x)) else float(x))

def fmt_int(x):
    try:
        if pd.isna(x):
            return ""
        return f"{int(float(x)):,}"
    except Exception:
        return ""

def fmt_float(x, nd=2):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""

def _pct_to_ratio(x):
    """
    Convert a Census 'percent-ish' value to a 0..1 ratio robustly.
    Handles:
      - already ratio (0.0..1.0)
      - percent (0..100)
      - weirdly scaled (0..10000) from accidental *100 or API quirks
    """
    try:
        if pd.isna(x):
            return np.nan
        v = float(x)
        if not np.isfinite(v):
            return np.nan
        av = abs(v)
        while av > 100.0:
            v /= 100.0
            av = abs(v)
        if av > 1.0:
            v /= 100.0
        return v
    except Exception:
        return np.nan

def fmt_pct(x):
    try:
        if pd.isna(x):
            return ""
        r = _pct_to_ratio(x)
        if pd.isna(r):
            return ""
        return f"{r:.2%}"
    except Exception:
        return ""

def fmt_money(x):
    try:
        if pd.isna(x):
            return ""
        return f"${float(x):,.0f}"
    except Exception:
        return ""

def norm_dist_id(st, dist):
    st = (st or "").strip().upper()
    if pd.isna(dist):
        return f"{st}-AL"
    try:
        d = int(float(dist))
    except Exception:
        d = None
    if d is None or d == 0:
        return f"{st}-AL"
    return f"{st}-{d}"

def cand_join(names):
    names = [n for n in names if n and str(n).strip()]
    names = [str(n).strip() for n in names]
    names = [n for n in names if n.upper() != "NAN"]
    names = sorted(set(names))
    return " / ".join(names[:3]) if names else ""

def party_simple_from_fec(party_str: str):
    p = (party_str or "").strip().lower()
    if "democrat" in p: return "DEMOCRAT"
    if "republican" in p: return "REPUBLICAN"
    return ""

def district_code_to_id(code: str):
    s = (code or "").strip().upper()
    m = re.match(r"^([A-Z]{2})-(\d{1,2}|00|AL)$", s)
    if not m:
        return ""
    st, d = m.group(1), m.group(2)
    if d in ("AL", "00"):
        return f"{st}-AL"
    try:
        return f"{st}-{int(d)}"
    except Exception:
        return ""

def normalize_geo_to_district_id(geo: str) -> str:
    s = (geo or "").strip().upper()
    m = re.match(r"^([A-Z]{2})-(AL|\d{1,2}|00|0)$", s)
    if not m:
        return ""
    st, d = m.group(1), m.group(2)
    if d in ("AL", "00", "0"):
        return f"{st}-AL"
    try:
        return f"{st}-{int(d)}"
    except Exception:
        return ""

def mode_count(vals):
    vals = [v for v in vals if v]
    if not vals:
        return 0
    vc = pd.Series(vals).value_counts()
    return int(vc.iloc[0])

def series_or_blank(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([""] * len(df))

# ----------------------------
# HTML FETCH
# ----------------------------
@st.cache_data(show_spinner=False, ttl=6*60*60)
def fetch_html(url, timeout=30):
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code in (401, 403):
            return ""
        r.raise_for_status()
        return r.text
    except Exception:
        return ""

# ----------------------------
# RATINGS PARSERS (270toWin)
# ----------------------------
@st.cache_data(show_spinner=False, ttl=6*60*60)
def parse_270toWin_table_like(url):
    html = fetch_html(url)
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    tokens = [t.strip() for t in soup.get_text("\n").split("\n")]
    tokens = [t for t in tokens if t]
    current = None
    out = {}

    for t in tokens:
        t_norm = normalize_rating_label(t)
        # detect section headers reliably (including Toss-up)
        if t_norm in RATING_KEYS:
            current = t_norm
            continue
        if not current:
            continue

        m = DIST_RE.search(t.upper())
        if m:
            out[m.group(1).upper()] = current

    return out

@st.cache_data(show_spinner=False, ttl=6*60*60)
def get_2026_ratings_maps():
    cook_map = parse_270toWin_table_like(URL_COOK_270)
    sabato_map = parse_270toWin_table_like(URL_SABATO_270)
    inside_map = parse_270toWin_table_like(URL_INSIDE_270)
    return cook_map, sabato_map, inside_map

def build_ratings_union_table(cook_map: dict, sabato_map: dict, inside_map: dict) -> pd.DataFrame:
    districts = sorted(set(list(cook_map.keys()) + list(sabato_map.keys()) + list(inside_map.keys())))
    if not districts:
        return pd.DataFrame()

    df = pd.DataFrame({"district_id": districts})
    df["state_po"] = df["district_id"].astype(str).str.split("-", n=1).str[0].str.upper()

    df["Cook_2026"] = df["district_id"].map(cook_map).fillna("")
    df["Sabato_2026"] = df["district_id"].map(sabato_map).fillna("")
    df["Inside_2026"] = df["district_id"].map(inside_map).fillna("")

    for src in ["Cook_2026", "Sabato_2026", "Inside_2026"]:
        df[src] = df[src].apply(normalize_rating_label)
        df[src + "_side"] = df[src].apply(rating_side)
        df[src + "_score"] = df[src].apply(rating_score)

    df["mentioned_by_count"] = (
        (df["Cook_2026"].astype(str).str.len() > 0).astype(int)
        + (df["Sabato_2026"].astype(str).str.len() > 0).astype(int)
        + (df["Inside_2026"].astype(str).str.len() > 0).astype(int)
    )

    df["exact_label_agree_max"] = df.apply(
        lambda r: mode_count([r["Cook_2026"], r["Sabato_2026"], r["Inside_2026"]]), axis=1
    )
    df["side_agree_max"] = df.apply(
        lambda r: mode_count([r["Cook_2026_side"], r["Sabato_2026_side"], r["Inside_2026_side"]]), axis=1
    )

    df["avg_score"] = df[["Cook_2026_score", "Sabato_2026_score", "Inside_2026_score"]].mean(axis=1, skipna=True)
    df["consensus_by_avgscore"] = df["avg_score"].apply(consensus_label_from_avgscore)
    df["consensus_side"] = df["consensus_by_avgscore"].apply(rating_side)

    def any_competitive(row):
        labs = [
            normalize_rating_label(row["Cook_2026"]),
            normalize_rating_label(row["Sabato_2026"]),
            normalize_rating_label(row["Inside_2026"]),
        ]
        return int(any(l in ("Toss-up", "Tilt Dem", "Tilt Rep") for l in labs if l))

    df["any_tossup_or_tilt"] = df.apply(any_competitive, axis=1)
    return df

# ----------------------------
# HOUSE LOADER (robust)
# ----------------------------
def load_house_wrapped_quotes_csv(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        header_line = f.readline().strip().lstrip("\ufeff")
        header = [h.strip() for h in header_line.split(",")]
        cand_idx = header.index("candidate") if "candidate" in header else None
        rows = []
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith('"') and ln.endswith('"'):
                ln = ln[1:-1]
            parts = ln.split(",")
            if cand_idx is not None and len(parts) > len(header):
                extra = len(parts) - len(header)
                candidate_merged = ",".join(parts[cand_idx : cand_idx + extra + 1])
                parts = parts[:cand_idx] + [candidate_merged] + parts[cand_idx + extra + 1 :]
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            if len(parts) > len(header):
                parts = parts[: len(header)]
            rows.append(parts)
        df = pd.DataFrame(rows, columns=header)
        df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
        return df

@st.cache_data(show_spinner=True)
def load_inputs(pres_path, house_path):
    pres_df = pd.read_csv(pres_path, low_memory=False)
    pres_df.columns = pres_df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    house_path_p = Path(house_path)
    try:
        if house_path_p.suffix.lower() in [".tab", ".tsv"]:
            house_df_try = pd.read_csv(house_path, sep="\t", low_memory=False)
        else:
            house_df_try = pd.read_csv(house_path, low_memory=False)
        house_df_try.columns = house_df_try.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
        y = pd.to_numeric(house_df_try.get("year", pd.Series(dtype="object")), errors="coerce")
        if y.notna().sum() == 0:
            raise ValueError("House year didn't parse with normal read.")
        house_df = house_df_try
    except Exception:
        house_df = load_house_wrapped_quotes_csv(house_path)

    # Normalize president
    pres_df["year"] = pd.to_numeric(pres_df.get("year", pd.Series(dtype="object")), errors="coerce")
    pres_df["party_simplified"] = pres_df.get("party_simplified", "").astype(str).str.strip().str.upper()
    pres_df["state_po"] = pres_df.get("state_po", "").astype(str).str.strip().str.upper()
    pres_df["candidatevotes"] = pd.to_numeric(pres_df.get("candidatevotes", pd.Series(dtype="object")), errors="coerce")
    pres_cand_col = "candidate" if "candidate" in pres_df.columns else None
    if pres_cand_col:
        pres_df[pres_cand_col] = pres_df[pres_cand_col].fillna("").astype(str).str.strip()

    # Normalize house
    house_df["year"] = pd.to_numeric(house_df.get("year", pd.Series(dtype="object")), errors="coerce")
    for c in ["office", "stage", "party", "state_po", "candidate"]:
        if c in house_df.columns:
            house_df[c] = house_df[c].fillna("").astype(str).str.strip().str.upper()
    house_df["candidatevotes"] = pd.to_numeric(house_df.get("candidatevotes", pd.Series(dtype="object")), errors="coerce")

    return pres_df, pres_cand_col, house_df

# ----------------------------
# LOAD FEC SPENDING (EXCEL)
# ----------------------------
@st.cache_data(show_spinner=True)
def load_fec_spending(spend_xlsx_path: str):
    if not spend_xlsx_path:
        return pd.DataFrame(), pd.DataFrame()
    p = Path(spend_xlsx_path)
    if not p.exists():
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_excel(p, sheet_name="House_Candidate_Spending")
    df.columns = [str(c).strip() for c in df.columns]

    df["cycle_year"] = pd.to_numeric(df.get("cycle_year", np.nan), errors="coerce")
    df["state_po"] = df.get("state_abbrev", "").astype(str).str.strip().str.upper()
    df["district_id"] = df.get("district_code", "").astype(str).apply(district_code_to_id)
    df["party_simple"] = df.get("party", "").astype(str).apply(party_simple_from_fec)

    for c in ["receipts", "disbursements", "cash_on_hand", "debts"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # District totals across all parties
    dist_all = (
        df.groupby(["cycle_year", "state_po", "district_id"], dropna=False)[["receipts", "disbursements"]]
          .sum()
          .reset_index()
          .rename(columns={"receipts": "fec_receipts_all", "disbursements": "fec_disburse_all"})
    )

    # District totals for Dem/Rep only
    maj = df[df["party_simple"].isin(["DEMOCRAT", "REPUBLICAN"])].copy()
    if maj.empty:
        return pd.DataFrame(), pd.DataFrame()

    dist_party = (
        maj.groupby(["cycle_year", "state_po", "district_id", "party_simple"])[["receipts", "disbursements"]]
           .sum()
           .reset_index()
    )

    piv = dist_party.pivot_table(
        index=["cycle_year", "state_po", "district_id"],
        columns="party_simple",
        values=["receipts", "disbursements"],
        aggfunc="sum",
        fill_value=0.0
    )
    piv.columns = [f"fec_{a.lower()}_{b.lower()}" for a, b in piv.columns.to_flat_index()]
    piv = piv.reset_index()

    spend_dist = piv.merge(dist_all, on=["cycle_year", "state_po", "district_id"], how="left")

    # Ensure expected columns exist even if one party is missing in a district
    for c in ["fec_receipts_democrat", "fec_receipts_republican", "fec_disburse_democrat", "fec_disburse_republican"]:
        if c not in spend_dist.columns:
            spend_dist[c] = 0.0

    spend_dist["fec_receipts_maj_total"] = spend_dist["fec_receipts_democrat"] + spend_dist["fec_receipts_republican"]
    spend_dist["fec_receipts_margin"] = (
        (spend_dist["fec_receipts_republican"] - spend_dist["fec_receipts_democrat"])
        / spend_dist["fec_receipts_maj_total"].replace(0, np.nan)
    )

    spend_dist["fec_disburse_maj_total"] = spend_dist["fec_disburse_democrat"] + spend_dist["fec_disburse_republican"]
    spend_dist["fec_disburse_margin"] = (
        (spend_dist["fec_disburse_republican"] - spend_dist["fec_disburse_democrat"])
        / spend_dist["fec_disburse_maj_total"].replace(0, np.nan)
    )

    # State totals (sum across districts)
    spend_state = (
        spend_dist.groupby(["cycle_year", "state_po"], dropna=False)[
            [
                "fec_receipts_democrat", "fec_receipts_republican", "fec_receipts_all",
                "fec_disburse_democrat", "fec_disburse_republican", "fec_disburse_all",
            ]
        ]
        .sum()
        .reset_index()
    )

    spend_state["fec_receipts_maj_total"] = spend_state["fec_receipts_democrat"] + spend_state["fec_receipts_republican"]
    spend_state["fec_receipts_margin"] = (
        (spend_state["fec_receipts_republican"] - spend_state["fec_receipts_democrat"])
        / spend_state["fec_receipts_maj_total"].replace(0, np.nan)
    )

    spend_state["fec_disburse_maj_total"] = spend_state["fec_disburse_democrat"] + spend_state["fec_disburse_republican"]
    spend_state["fec_disburse_margin"] = (
        (spend_state["fec_disburse_republican"] - spend_state["fec_disburse_democrat"])
        / spend_state["fec_disburse_maj_total"].replace(0, np.nan)
    )

    return spend_dist, spend_state


# ----------------------------
# LOAD WAR (CSV)
# ----------------------------
@st.cache_data(show_spinner=True)
def load_war_by_district_year(war_csv_path: str) -> pd.DataFrame:
    if not war_csv_path:
        return pd.DataFrame()
    p = Path(war_csv_path)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    y = series_or_blank(df, "Year")
    chamber = series_or_blank(df, "Chamber")
    geo = series_or_blank(df, "Geography")
    dem = series_or_blank(df, "Democrat")
    rep = series_or_blank(df, "Republican")
    war = series_or_blank(df, "WAR")
    sortable = series_or_blank(df, "Sortable")

    out = pd.DataFrame({
        "year": pd.to_numeric(y, errors="coerce"),
        "chamber": chamber.fillna("").astype(str).str.strip().str.upper(),
        "district_id": geo.fillna("").astype(str).apply(normalize_geo_to_district_id),
        "war_str": war.fillna("").astype(str).str.strip(),
        "war_sortable": pd.to_numeric(sortable, errors="coerce"),
        "war_dem_candidate": dem.fillna("").astype(str).str.strip(),
        "war_rep_candidate": rep.fillna("").astype(str).str.strip(),
    })

    out = out[(out["district_id"].astype(str).str.len() > 0) & (out["year"].notna())].copy()
    out = out[out["chamber"].isin(["HOUSE", "US HOUSE", "U.S. HOUSE", "USHOUSE", "HOUSE OF REPRESENTATIVES", ""])].copy()

    out = out.sort_values(["district_id", "year", "war_sortable"], ascending=[True, True, False])
    out = out.drop_duplicates(subset=["district_id", "year"], keep="first").reset_index(drop=True)
    return out[["district_id", "year", "war_str", "war_sortable", "war_dem_candidate", "war_rep_candidate"]]

# ----------------------------
# ACS (Census Data API) - district + state
#   FIX: resolve variables by LABEL (DP tables shift across years)
# ----------------------------
CENSUS_API_BASE = "https://api.census.gov/data"

ACS_OUTPUT_COLS = [
    "acs_total_pop",
    "acs_pct_male",
    "acs_pct_female",
    "acs_median_age",
    "acs_pct_white_alone",
    "acs_pct_black_alone",
    "acs_pct_asian_alone",
    "acs_pct_hispanic",
    "acs_median_hh_income",
    "acs_pct_bachelors_or_higher",
    "acs_pct_veteran",
]

def _census_get(url: str, timeout=40):
    r = requests.get(url, headers=UA, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Census API HTTP {r.status_code}: {r.text[:500]}")
    return r.json()

@st.cache_data(show_spinner=False, ttl=24*60*60)
def load_acs_variables_meta(acs_year: int) -> dict:
    url = f"{CENSUS_API_BASE}/{acs_year}/acs/acs5/profile/variables.json"
    r = requests.get(url, headers=UA, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"variables.json HTTP {r.status_code}: {r.text[:300]}")
    j = r.json()
    return j.get("variables", {})

def _pick_var_code(vars_meta: dict, wants_percent: bool, include_tokens, exclude_tokens):
    include_tokens = [t.lower() for t in include_tokens if t]
    exclude_tokens = [t.lower() for t in (exclude_tokens or []) if t]

    candidates = []
    for code, meta in vars_meta.items():
        lab = str(meta.get("label", "")).lower()

        if "margin of error" in lab or lab.startswith("annotation"):
            continue
        if "annotation" in lab:
            continue

        if wants_percent:
            if not code.endswith("PE"):
                continue
        else:
            if not code.endswith("E") or code.endswith("PE"):
                continue

        ok = True
        for t in include_tokens:
            if t not in lab:
                ok = False
                break
        if not ok:
            continue
        for t in exclude_tokens:
            if t in lab:
                ok = False
                break
        if not ok:
            continue

        score = 0
        score += max(0, 200 - len(lab))
        if "alone" in lab:
            score += 20
        if "in combination" in lab:
            score -= 60
        candidates.append((score, code, lab))

    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]

@st.cache_data(show_spinner=True, ttl=24*60*60)
def resolve_acs_profile_varmap(acs_year: int):
    vars_meta = load_acs_variables_meta(acs_year)
    notes = []
    varmap = {}

    total_code = None
    for inc in [
        ["estimate", "total population"],
        ["estimate", "sex and age", "total population"],
        ["estimate", "population", "total population"],
    ]:
        total_code = _pick_var_code(vars_meta, wants_percent=False, include_tokens=inc, exclude_tokens=["percent"])
        if total_code:
            break
    varmap["acs_total_pop"] = total_code

    varmap["acs_pct_male"] = (
        _pick_var_code(vars_meta, True, ["percent", "male"], ["margin of error"])
        or _pick_var_code(vars_meta, True, ["percent", "sex and age", "male"], [])
    )
    varmap["acs_pct_female"] = (
        _pick_var_code(vars_meta, True, ["percent", "female"], ["margin of error"])
        or _pick_var_code(vars_meta, True, ["percent", "sex and age", "female"], [])
    )

    varmap["acs_median_age"] = (
        _pick_var_code(vars_meta, False, ["median age"], [])
        or _pick_var_code(vars_meta, False, ["estimate", "median age"], [])
    )

    varmap["acs_pct_white_alone"] = (
        _pick_var_code(vars_meta, True, ["percent", "white alone"], ["in combination", "not hispanic"])
        or _pick_var_code(vars_meta, True, ["percent", "race", "white alone"], ["in combination", "not hispanic"])
        or _pick_var_code(vars_meta, True, ["percent", "white"], ["in combination", "not hispanic"])
    )
    varmap["acs_pct_black_alone"] = (
        _pick_var_code(vars_meta, True, ["percent", "black or african american alone"], ["in combination", "not hispanic"])
        or _pick_var_code(vars_meta, True, ["percent", "black alone"], ["in combination", "not hispanic"])
        or _pick_var_code(vars_meta, True, ["percent", "black or african american"], ["in combination", "not hispanic"])
    )
    varmap["acs_pct_asian_alone"] = (
        _pick_var_code(vars_meta, True, ["percent", "asian alone"], ["in combination", "not hispanic"])
        or _pick_var_code(vars_meta, True, ["percent", "asian"], ["in combination", "not hispanic"])
    )

    varmap["acs_pct_hispanic"] = (
        _pick_var_code(vars_meta, True, ["percent", "hispanic or latino"], ["not hispanic"])
        or _pick_var_code(vars_meta, True, ["percent", "hispanic"], ["not hispanic"])
    )

    varmap["acs_median_hh_income"] = (
        _pick_var_code(vars_meta, False, ["median household income"], ["margin of error"])
        or _pick_var_code(vars_meta, False, ["estimate", "median household income"], [])
    )

    varmap["acs_pct_bachelors_or_higher"] = (
        _pick_var_code(vars_meta, True, ["percent", "bachelor", "higher"], [])
        or _pick_var_code(vars_meta, True, ["percent", "bachelor"], [])
        or _pick_var_code(vars_meta, True, ["percent", "education", "bachelor"], [])
    )

    varmap["acs_pct_veteran"] = (
        _pick_var_code(vars_meta, True, ["percent", "veteran"], [])
        or _pick_var_code(vars_meta, True, ["percent", "civilian population 18 years and over", "veteran"], [])
    )

    for k in ACS_OUTPUT_COLS:
        c = varmap.get(k)
        if c:
            notes.append(f"{k} -> {c}")
        else:
            notes.append(f"{k} -> (NOT FOUND)")

    return varmap, notes

def _append_key(url: str, api_key: str | None):
    if api_key and str(api_key).strip():
        join = "&" if "?" in url else "?"
        return url + f"{join}key={str(api_key).strip()}"
    return url

@st.cache_data(show_spinner=True, ttl=24*60*60)
def load_acs_profile_congressional_district(acs_year: int, api_key: str | None) -> tuple[pd.DataFrame, list[str]]:
    varmap, notes = resolve_acs_profile_varmap(acs_year)
    codes = [varmap.get(k) for k in ACS_OUTPUT_COLS if varmap.get(k)]
    if not codes:
        return pd.DataFrame(), notes

    get_expr = "NAME," + ",".join(codes)
    url = (
        f"{CENSUS_API_BASE}/{acs_year}/acs/acs5/profile"
        f"?get={get_expr}&for=congressional%20district:*&in=state:*"
    )
    url = _append_key(url, api_key)
    data = _census_get(url)

    if not data or len(data) < 2:
        return pd.DataFrame(), notes

    header = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=header)

    df["state"] = df["state"].astype(str).str.zfill(2)
    df["congressional district"] = df["congressional district"].astype(str).str.zfill(2)
    df["state_po"] = df["state"].map(FIPS_TO_STATE).fillna("")

    def mk_did(row):
        st = row["state_po"]
        cd = row["congressional district"]
        if cd == "00":
            return f"{st}-AL"
        try:
            return f"{st}-{int(cd)}"
        except Exception:
            return ""

    df["district_id"] = df.apply(mk_did, axis=1)
    df = df[df["district_id"].astype(str).str.len() > 0].copy()

    out = df[["district_id", "state_po"]].copy()
    for logical in ACS_OUTPUT_COLS:
        code = varmap.get(logical)
        if code and code in df.columns:
            out[logical] = pd.to_numeric(df[code], errors="coerce")
        else:
            out[logical] = np.nan

    return out, notes

@st.cache_data(show_spinner=True, ttl=24*60*60)
def load_acs_profile_state(acs_year: int, api_key: str | None) -> tuple[pd.DataFrame, list[str]]:
    varmap, notes = resolve_acs_profile_varmap(acs_year)
    codes = [varmap.get(k) for k in ACS_OUTPUT_COLS if varmap.get(k)]
    if not codes:
        return pd.DataFrame(), notes

    get_expr = "NAME," + ",".join(codes)
    url = f"{CENSUS_API_BASE}/{acs_year}/acs/acs5/profile?get={get_expr}&for=state:*"
    url = _append_key(url, api_key)
    data = _census_get(url)

    if not data or len(data) < 2:
        return pd.DataFrame(), notes

    header = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=header)

    df["state"] = df["state"].astype(str).str.zfill(2)
    df["state_po"] = df["state"].map(FIPS_TO_STATE).fillna("")

    out = df[["state_po"]].copy()
    for logical in ACS_OUTPUT_COLS:
        code = varmap.get(logical)
        if code and code in df.columns:
            out[logical] = pd.to_numeric(df[code], errors="coerce")
        else:
            out[logical] = np.nan

    out = out[out["state_po"].astype(str).str.len() == 2].copy()
    return out, notes

def load_acs_with_fallback(requested_year: int, api_key: str | None):
    tried = []
    errors = {}
    var_notes = []
    for y in [requested_year, requested_year - 1, requested_year - 2]:
        if y <= 2009:
            continue
        tried.append(y)
        try:
            cd, cd_notes = load_acs_profile_congressional_district(y, api_key)
            stt, st_notes = load_acs_profile_state(y, api_key)
            var_notes = cd_notes
            if not cd.empty and not stt.empty:
                return y, cd, stt, tried, errors, var_notes
            else:
                errors[y] = f"Empty response (cd_rows={len(cd)}, state_rows={len(stt)})"
        except Exception as e:
            errors[y] = repr(e)
            continue

    return requested_year, pd.DataFrame(), pd.DataFrame(), tried, errors, var_notes

# ----------------------------
# COMPUTATIONS
# ----------------------------
def compute_pres_state_results(pres_df, pres_cand_col, year):
    df = pres_df[(pres_df["year"] == year) & (pres_df["state_po"].notna()) & (pres_df["candidatevotes"].notna())].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "state_po","pres_margin",
            "pres_dem_candidate","pres_rep_candidate",
            "pres_dem_votes","pres_rep_votes","pres_total_votes_all",
            "pres_dem_pct_all","pres_rep_pct_all"
        ])

    tot_all = df.groupby("state_po")["candidatevotes"].sum().rename("pres_total_votes_all").reset_index()

    maj = df[df["party_simplified"].isin(["DEMOCRAT","REPUBLICAN"])].copy()
    pv = maj.groupby(["state_po","party_simplified"])["candidatevotes"].sum().unstack(fill_value=0)
    if "DEMOCRAT" not in pv.columns: pv["DEMOCRAT"] = 0
    if "REPUBLICAN" not in pv.columns: pv["REPUBLICAN"] = 0
    pv = pv.reset_index().rename(columns={"DEMOCRAT":"pres_dem_votes","REPUBLICAN":"pres_rep_votes"})

    if pres_cand_col:
        dem_names = maj[maj["party_simplified"]=="DEMOCRAT"].groupby("state_po")[pres_cand_col].apply(lambda s: cand_join(s.tolist())).rename("pres_dem_candidate").reset_index()
        rep_names = maj[maj["party_simplified"]=="REPUBLICAN"].groupby("state_po")[pres_cand_col].apply(lambda s: cand_join(s.tolist())).rename("pres_rep_candidate").reset_index()
    else:
        dem_names = pd.DataFrame({"state_po": pv["state_po"], "pres_dem_candidate": ""})
        rep_names = pd.DataFrame({"state_po": pv["state_po"], "pres_rep_candidate": ""})

    out = pv.merge(tot_all, on="state_po", how="left").merge(dem_names, on="state_po", how="left").merge(rep_names, on="state_po", how="left")
    out["pres_dem_pct_all"] = out["pres_dem_votes"] / out["pres_total_votes_all"].replace(0, np.nan)
    out["pres_rep_pct_all"] = out["pres_rep_votes"] / out["pres_total_votes_all"].replace(0, np.nan)
    major_total = (out["pres_dem_votes"] + out["pres_rep_votes"]).replace(0, np.nan)
    out["pres_margin"] = (out["pres_rep_votes"] - out["pres_dem_votes"]) / major_total

    for c in ["pres_dem_pct_all","pres_rep_pct_all","pres_margin"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return out[[
        "state_po","pres_margin",
        "pres_dem_candidate","pres_rep_candidate",
        "pres_dem_votes","pres_rep_votes","pres_total_votes_all",
        "pres_dem_pct_all","pres_rep_pct_all"
    ]]

def compute_house_district_results(house_df, year):
    df = house_df[(house_df["year"] == year) & (house_df["office"] == "US HOUSE") & (house_df["stage"] == "GEN") & (house_df["candidatevotes"].notna())].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "state_po","district","district_id",
            "dem_candidate","rep_candidate",
            "dem_votes","rep_votes","total_votes_all",
            "dem_pct_all","rep_pct_all","house_margin"
        ])

    totals_all = df.groupby(["state_po","district"], dropna=False)["candidatevotes"].sum().rename("total_votes_all").reset_index()
    dmaj = df[df["party"].isin(["DEMOCRAT","REPUBLICAN"])].copy()

    if dmaj.empty:
        out = totals_all.copy()
        out["district_id"] = out.apply(lambda r: norm_dist_id(r["state_po"], r["district"]), axis=1)
        for c in ["dem_candidate","rep_candidate","dem_votes","rep_votes","dem_pct_all","rep_pct_all","house_margin"]:
            out[c] = np.nan
        return out

    pv = dmaj.groupby(["state_po","district","party"], dropna=False)["candidatevotes"].sum().unstack(fill_value=0)
    if "DEMOCRAT" not in pv.columns: pv["DEMOCRAT"] = 0
    if "REPUBLICAN" not in pv.columns: pv["REPUBLICAN"] = 0
    pv = pv.reset_index().rename(columns={"DEMOCRAT":"dem_votes","REPUBLICAN":"rep_votes"})

    dem_names = dmaj[dmaj["party"]=="DEMOCRAT"].groupby(["state_po","district"], dropna=False)["candidate"].apply(lambda s: cand_join(s.tolist())).rename("dem_candidate").reset_index()
    rep_names = dmaj[dmaj["party"]=="REPUBLICAN"].groupby(["state_po","district"], dropna=False)["candidate"].apply(lambda s: cand_join(s.tolist())).rename("rep_candidate").reset_index()

    out = pv.merge(totals_all, on=["state_po","district"], how="left")
    out = out.merge(dem_names, on=["state_po","district"], how="left").merge(rep_names, on=["state_po","district"], how="left")
    out["district_id"] = out.apply(lambda r: norm_dist_id(r["state_po"], r["district"]), axis=1)

    out["dem_pct_all"] = out["dem_votes"] / out["total_votes_all"].replace(0, np.nan)
    out["rep_pct_all"] = out["rep_votes"] / out["total_votes_all"].replace(0, np.nan)
    major_total = (out["dem_votes"] + out["rep_votes"]).replace(0, np.nan)
    out["house_margin"] = (out["rep_votes"] - out["dem_votes"]) / major_total

    for c in ["dem_pct_all","rep_pct_all","house_margin"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return out[[
        "state_po","district","district_id",
        "dem_candidate","rep_candidate",
        "dem_votes","rep_votes","total_votes_all",
        "dem_pct_all","rep_pct_all","house_margin"
    ]]

def compute_house_state_avg(house_df, year):
    ddf = compute_house_district_results(house_df, year)
    if ddf.empty:
        return ddf, pd.DataFrame(columns=["state_po","avg_house_margin"])
    avg = ddf.groupby("state_po")["house_margin"].mean().rename("avg_house_margin").reset_index()
    avg["avg_house_margin"] = pd.to_numeric(avg["avg_house_margin"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return ddf, avg

def attach_ratings(ddf, cook_map, sabato_map, inside_map):
    if ddf.empty:
        return ddf
    d = ddf.copy()
    d["Cook_2026"] = d["district_id"].map(cook_map).fillna("").apply(normalize_rating_label)
    d["Sabato_2026"] = d["district_id"].map(sabato_map).fillna("").apply(normalize_rating_label)
    d["Inside_2026"] = d["district_id"].map(inside_map).fillna("").apply(normalize_rating_label)
    d["tossup_agree_count"] = (
        d["Cook_2026"].apply(is_tossup).astype(int)
        + d["Sabato_2026"].apply(is_tossup).astype(int)
        + d["Inside_2026"].apply(is_tossup).astype(int)
    )
    return d

# ----------------------------
# SHAPES (cached download + read)
# ----------------------------
def _download_cached(url, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    cache_path.write_bytes(r.content)
    return cache_path

@st.cache_data(show_spinner=True)
def load_state_cd_geojson(year, state_po, cache_dir="district_shapes_cache"):
    state_po = state_po.upper().strip()
    if state_po not in STATE_FIPS:
        raise ValueError(f"Unknown state_po: {state_po}")
    url = CD_ZIPS.get(year)
    if not url:
        raise ValueError(f"No Census district shapes configured for year={year}")

    cache_dir = Path(cache_dir)
    zip_path = cache_dir / f"cd_shapes_{year}.zip"
    _download_cached(url, zip_path)

    gdf = gpd.read_file(f"zip://{zip_path}")

    cols_upper = {str(c).upper(): c for c in gdf.columns}
    cd_candidates = []
    for u, orig in cols_upper.items():
        if u.startswith("CD") and u.endswith("FP"):
            cd_candidates.append(orig)
    if not cd_candidates:
        cd_candidates = [c for c in gdf.columns if re.match(r"^CD\d+FP$", str(c).strip(), flags=re.I)]

    if not cd_candidates:
        raise ValueError(f"Could not find district FP column. Columns: {list(gdf.columns)}")
    cd_col = cd_candidates[0]

    if "STATEFP" not in gdf.columns:
        raise ValueError(f"Could not find STATEFP. Columns: {list(gdf.columns)}")

    stfp = STATE_FIPS[state_po]
    gdf = gdf[gdf["STATEFP"].astype(str).str.zfill(2) == stfp].copy()
    if gdf.empty:
        raise ValueError("No geometries for this state in the Census shapefile.")

    gdf[cd_col] = gdf[cd_col].astype(str).str.zfill(2)

    def mk_id(fp):
        return f"{state_po}-AL" if fp == "00" else f"{state_po}-{int(fp)}"

    gdf["district_id"] = gdf[cd_col].map(mk_id)
    gdf = gdf[gdf.geometry.notna()].copy()
    try:
        gdf["geometry"] = gdf.geometry.buffer(0)
    except Exception:
        pass
    geojson = json.loads(gdf.to_json())
    return geojson, gdf

# ============================
# NEW: US-wide district shapes loader
# ============================

@st.cache_data(show_spinner=True)
def load_us_cd_shapes(year: int, cache_dir: str = "district_shapes_cache"):
    """
    Load the Congressional district shapes for the entire US for the given year.
    This reuses the Census cartographic boundary files defined in CD_ZIPS and
    mirrors the state-level loader but without filtering to a single state.

    Returns a GeoDataFrame with a `district_id` column and geometries as well as
    the corresponding GeoJSON dictionary used by Plotly.  The district_id is
    normalized to the form ``"ST-N"`` or ``"ST-AL"`` matching the rest of
    the application.  The function caches its result to avoid re-reading
    shapefiles on every run.
    """
    url = CD_ZIPS.get(year)
    if not url:
        raise ValueError(f"No Census district shapes configured for year={year}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"cd_shapes_all_{year}.zip"
    _download_cached(url, zip_path)

    # Read the entire shapefile (all states).  Using the zip:// URI allows
    # geopandas to read directly from the compressed archive.
    gdf = gpd.read_file(f"zip://{zip_path}")

    # Identify the district number column.  The Census files include a column
    # named CD<cycle>FP (e.g. CD118FP).  We search for the first column
    # matching that pattern.  Fall back to any column starting with CD and
    # ending with FP if the exact name cannot be determined.
    cols_upper = {str(c).upper(): c for c in gdf.columns}
    cd_candidates = []
    for u, orig in cols_upper.items():
        if u.startswith("CD") and u.endswith("FP"):
            cd_candidates.append(orig)
    if not cd_candidates:
        cd_candidates = [c for c in gdf.columns if re.match(r"^CD\d+FP$", str(c).strip(), flags=re.I)]
    if not cd_candidates:
        raise ValueError(f"Could not find district FP column in US shapes. Columns: {list(gdf.columns)}")
    cd_col = cd_candidates[0]

    if "STATEFP" not in gdf.columns:
        raise ValueError(f"Could not find STATEFP in US shapes. Columns: {list(gdf.columns)}")

    # Ensure district field is zero-padded two digits and create district_id.
    gdf[cd_col] = gdf[cd_col].astype(str).str.zfill(2)
    # Normalize district_id using state FIPS to state abbreviation mapping.
    def mk_id(row):
        stfp = str(row["STATEFP"]).zfill(2)
        state_po = FIPS_TO_STATE.get(stfp, "")
        if not state_po:
            return None
        fp = row[cd_col]
        return f"{state_po}-AL" if fp == "00" else f"{state_po}-{int(fp)}"

    gdf["district_id"] = gdf.apply(mk_id, axis=1)
    gdf = gdf[gdf["district_id"].notna()].copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    try:
        # Sometimes geometries can be invalid; buffer(0) is a common fix.
        gdf["geometry"] = gdf.geometry.buffer(0)
    except Exception:
        pass
    # Compute centroids (lat/lon) for each district for later distance calculations.
    try:
        gdf["centroid_lat"] = gdf.geometry.centroid.y
        gdf["centroid_lon"] = gdf.geometry.centroid.x
    except Exception:
        gdf["centroid_lat"] = np.nan
        gdf["centroid_lon"] = np.nan

    geojson = json.loads(gdf.to_json())
    return gdf, geojson

# ============================
# NEW: Travel map figure builder
# ============================

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance between two points on the earth
    specified in decimal degrees using the haversine formula.  Returns
    distance in kilometres.
    """
    try:
        # convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        # Radius of earth in kilometres (mean radius)
        R = 6371.0088
        return float(R * c)
    except Exception:
        return float("nan")

def make_travel_map_figure(
    year: int,
    overlay_type: str,
    transport_mode: str,
    time_minutes: int,
    lat: float,
    lon: float,
    selected_districts: list,
    dist_df: pd.DataFrame,
    us_shapes: gpd.GeoDataFrame,
    us_geojson: dict,
    enable_acs: bool,
):
    """
    Build a Plotly figure that displays Congressional districts across the US
    coloured by either population or house margin and overlays a user-specified
    starting point with a travel radius circle.  Districts within the travel
    radius are highlighted, and user-selected districts are emphasised with
    thicker borders.  Distances are computed using a simple haversine formula
    and travel time estimates assume fixed speeds per mode.

    Parameters
    ----------
    year : int
        The election year (used to look up district metrics).
    overlay_type : {"Population heatmap", "House margin"}
        Determines which variable is used to colour the districts.
    transport_mode : {"Driving", "Walking", "Cycling"}
        Mode of transport used to estimate reachable distance.
    time_minutes : int
        Number of minutes one can travel away from the starting point.
    lat, lon : float
        Latitude and longitude of the user-specified starting location.
    selected_districts : list of str
        District identifiers (e.g. "TX-7") that the user wishes to highlight.
    dist_df : pd.DataFrame
        The per-district results dataframe for the chosen year (from year_data[year]["dist_df"]).
    us_shapes : GeoDataFrame
        The full US district shapes with centroid_lat/centroid_lon columns.
    us_geojson : dict
        GeoJSON representation of the US district shapes.
    enable_acs : bool
        Whether ACS data is available; used when overlay_type is population.

    Returns
    -------
    plotly.graph_objects.Figure
        A figure ready to be rendered by Streamlit.
    """
    if us_shapes.empty:
        return go.Figure()
    # Copy shapes to avoid modifying cached dataframe
    gdf = us_shapes.copy()

    # Determine overlay values
    overlay_vals = pd.Series([np.nan] * len(gdf), index=gdf.index)
    if overlay_type == "Population heatmap" and enable_acs:
        # Map ACS total population to district shapes
        if "acs_total_pop" in dist_df.columns:
            pop_map = dist_df.set_index("district_id")["acs_total_pop"].to_dict()
            overlay_vals = gdf["district_id"].map(pop_map)
        else:
            overlay_vals = pd.Series([np.nan] * len(gdf), index=gdf.index)
    else:
        # Use house margin
        if "house_margin" in dist_df.columns:
            hm_map = dist_df.set_index("district_id")["house_margin"].to_dict()
            overlay_vals = gdf["district_id"].map(hm_map)
        else:
            overlay_vals = pd.Series([np.nan] * len(gdf), index=gdf.index)

    # Normalize overlay values for plotting
    plot_vals = safe_plot_col(overlay_vals)
    arr = pd.to_numeric(plot_vals, errors="coerce")
    # Colour scale selection: for population use sequential; for margin use diverging
    if overlay_type == "Population heatmap":
        colorscale = "YlOrRd"
        # Set domain based on percentiles to reduce outlier skew
        if np.isfinite(arr).any():
            lo, hi = np.nanpercentile(arr[np.isfinite(arr)], [5, 95])
            zmin, zmax = float(lo), float(hi)
            if zmin == zmax:
                zmin, zmax = (0.0, float(np.nanmax(arr))) if np.isfinite(arr).any() else (0.0, 1.0)
        else:
            zmin, zmax = 0.0, 1.0
    else:
        # House margin (Rep − Dem); negative blue, positive red
        colorscale = "RdBu_r"
        if np.isfinite(arr).any():
            max_abs = float(np.nanmax(np.abs(arr.values)))
            zmin, zmax = -max_abs, max_abs
            if not np.isfinite(zmin) or zmax == 0:
                zmin, zmax = -1.0, 1.0
        else:
            zmin, zmax = -1.0, 1.0

    # Compute reachable radius in kilometres
    speeds = {"Driving": 96.56064, "Walking": 4.82803, "Cycling": 24.14016}  # mph converted to km/h
    speed_kmh = speeds.get(transport_mode, 96.56064)
    radius_km = float(speed_kmh) * (time_minutes / 60.0)

    # Compute distances and travel times from the user-specified point
    gdf["distance_km"] = gdf.apply(
        lambda row: _haversine(lat, lon, row.get("centroid_lat", np.nan), row.get("centroid_lon", np.nan)), axis=1
    )
    gdf["travel_time_hr"] = gdf["distance_km"] / speed_kmh
    gdf["within_radius"] = gdf["distance_km"] <= radius_km
    # Determine border colours: selected districts -> thick blue, reachable -> green, others -> light grey
    def border_color(did, within):
        if did in selected_districts:
            return "#0000FF"  # blue
        if within:
            return "#009900"  # green
        return "#AAAAAA"
    def border_width(did, within):
        if did in selected_districts:
            return 2.5
        if within:
            return 1.5
        return 0.5

    # Prepare hover information: district_id, distance_km, travel_time_hr, overlay value
    hover_id = gdf["district_id"].fillna("").astype(str)
    hover_dist = gdf["distance_km"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    hover_time = gdf["travel_time_hr"].apply(lambda x: f"{(x*60):.0f}" if pd.notna(x) else "")  # minutes
    hover_overlay = overlay_vals.apply(
        lambda v: fmt_int(v) if overlay_type == "Population heatmap" else fmt_pct(v)
    )
    customdata = np.stack([hover_id, hover_dist, hover_time, hover_overlay], axis=1)

    # Build the choropleth trace
    choropleth = go.Choropleth(
        geojson=us_geojson,
        locations=gdf["district_id"],
        featureidkey="properties.district_id",
        z=plot_vals,
        zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        marker_line_color=[border_color(did, within) for did, within in zip(gdf["district_id"], gdf["within_radius"])],
        marker_line_width=[border_width(did, within) for did, within in zip(gdf["district_id"], gdf["within_radius"])],
        colorbar_title=("Population" if overlay_type == "Population heatmap" else "House margin (R-D)"),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Distance: %{customdata[1]} km<br>"
            "Travel time: %{customdata[2]} min<br>"
            + ("Population: %{customdata[3]}" if overlay_type == "Population heatmap" else "House margin: %{customdata[3]}")
            + "<extra></extra>"
        ),
        showscale=True,
    )

    # Compute circle coordinates (approximate) for the travel radius.
    if np.isfinite(radius_km) and radius_km > 0 and np.isfinite(lat) and np.isfinite(lon):
        num_points = 360
        angles = np.linspace(0, 2 * np.pi, num_points)
        # Convert km to degrees latitude/longitude (approximate):
        deg_lat = radius_km / 111.32
        # For longitude the scaling depends on latitude
        deg_lon = radius_km / (111.32 * np.cos(np.radians(lat)) if np.cos(np.radians(lat)) != 0 else 1.0)
        circle_lats = lat + deg_lat * np.sin(angles)
        circle_lons = lon + deg_lon * np.cos(angles)
        circle_trace = go.Scattergeo(
            lat=circle_lats,
            lon=circle_lons,
            mode="lines",
            line=dict(color="#009900", width=2),
            name="Travel radius",
            hoverinfo="skip",
            showlegend=False,
        )
    else:
        circle_trace = None

    # Marker for starting point
    marker_trace = go.Scattergeo(
        lat=[lat],
        lon=[lon],
        mode="markers",
        marker=dict(size=8, color="#000000", symbol="x"),
        name="Starting point",
        hovertext="Starting point",
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure()
    fig.add_trace(choropleth)
    if circle_trace is not None:
        fig.add_trace(circle_trace)
    fig.add_trace(marker_trace)
    fig.update_layout(
        title_text="US Congressional Districts — Travel Canvas",
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            subunitcolor="rgb(204, 204, 204)",
            countrycolor="rgb(204, 204, 204)",
            lakecolor="rgb(255, 255, 255)",
            showlakes=True,
            showsubunits=False,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=640,
    )
    return fig

# ============================
# NEW: Render travel canvas tab
# ============================
def render_travel_canvas(year: int, year_data: dict, enable_acs: bool):
    """
    Render the canvassing / travel map tab.  This function encapsulates
    interactive controls for selecting a starting location, travel mode and
    duration, choosing a colour overlay, manually highlighting districts,
    and displaying both a map and a summary table of reachable districts.

    Parameters
    ----------
    year : int
        The currently selected election year.
    year_data : dict
        Dictionary returned from build_year_data() for all years.  This
        function will access the per-district dataframe via year_data[year]["dist_df"].
    enable_acs : bool
        Whether ACS demographic data is enabled.  If true, population heatmap
        will be available; otherwise only the House margin overlay is offered.
    """
    # Provide a visual divider to separate this tool from other sections
    st.divider()
    travel_tab = st.tabs(["Travel Map / Canvas"])[0]
    with travel_tab:
        st.subheader("Travel Map & District Reachability")
        st.write(
            """
            Use this tool to drop a starting point anywhere in the continental United States and explore
            which congressional districts fall within a selected travel time.  Districts are coloured
            either by total population (via ACS data) or by the House margin (Rep − Dem) for the
            selected year.  You can highlight specific districts manually and choose the mode of
            transportation to adjust the travel radius.
            """
        )

        # Choose overlay variable: population requires ACS; otherwise default to House margin
        overlay_options = ["House margin"]
        if enable_acs:
            overlay_options.insert(0, "Population heatmap")
        overlay_type = st.radio(
            "Colour districts by", overlay_options, index=0
        )

        # Transportation mode selection
        transport_mode = st.radio(
            "Transportation mode", ["Driving", "Walking", "Cycling"], index=0
        )
        time_minutes = st.slider(
            "Travel time (minutes)", min_value=5, max_value=120, value=60, step=5
        )

        # Starting location input
        st.markdown("**Starting location (latitude & longitude)**")
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input(
                "Latitude", min_value=-90.0, max_value=90.0, value=39.5, step=0.1,
            )
        with col_lon:
            lon = st.number_input(
                "Longitude", min_value=-180.0, max_value=180.0, value=-98.35, step=0.1,
            )

        # List of all districts for multiselect
        all_districts = (
            year_data[year]["dist_df"]["district_id"].dropna().astype(str).unique().tolist()
        )
        all_districts = sorted(all_districts)
        selected_districts = st.multiselect(
            "Districts to highlight (optional)", options=all_districts, default=[]
        )

        # Load full US district shapes for the selected year
        try:
            us_gdf, us_geojson = load_us_cd_shapes(year)
        except Exception as e:
            st.error(f"Unable to load US district shapes for year {year}.")
            st.exception(e)
            us_gdf, us_geojson = gpd.GeoDataFrame(), {}

        # Build and display the travel map figure
        if not us_gdf.empty:
            fig_travel = make_travel_map_figure(
                year,
                overlay_type,
                transport_mode,
                time_minutes,
                lat,
                lon,
                selected_districts,
                year_data[year]["dist_df"],
                us_gdf,
                us_geojson,
                enable_acs,
            )
            st.plotly_chart(fig_travel, use_container_width=True)

            # Compute distances and travel times for each district to display in summary table
            speed_map = {"Driving": 96.56064, "Walking": 4.82803, "Cycling": 24.14016}
            speed_kmh = speed_map.get(transport_mode, 96.56064)
            radius_km = float(speed_kmh) * (time_minutes / 60.0)
            temp = us_gdf[["district_id", "centroid_lat", "centroid_lon"]].copy()
            temp["distance_km"] = temp.apply(
                lambda row: _haversine(lat, lon, row.get("centroid_lat", np.nan), row.get("centroid_lon", np.nan)),
                axis=1,
            )
            temp["travel_minutes"] = (temp["distance_km"] / speed_kmh) * 60.0
            merge_val = None
            if overlay_type == "Population heatmap" and enable_acs and "acs_total_pop" in year_data[year]["dist_df"].columns:
                merge_val = year_data[year]["dist_df"][["district_id", "acs_total_pop"]].copy().rename(columns={"acs_total_pop": "overlay"})
            elif overlay_type == "House margin" and "house_margin" in year_data[year]["dist_df"].columns:
                merge_val = year_data[year]["dist_df"][["district_id", "house_margin"]].copy().rename(columns={"house_margin": "overlay"})
            if merge_val is not None:
                temp = temp.merge(merge_val, on="district_id", how="left")
            temp["within_radius"] = temp["distance_km"] <= radius_km
            temp["selected"] = temp["district_id"].isin(selected_districts)
            disp = temp.copy()
            disp["Distance (km)"] = disp["distance_km"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "")
            disp["Travel time (min)"] = disp["travel_minutes"].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "")
            if overlay_type == "Population heatmap" and enable_acs:
                disp["Population"] = disp.get("overlay", np.nan).apply(fmt_int)
            elif overlay_type == "House margin":
                disp["House margin (R-D)"] = disp.get("overlay", np.nan).apply(fmt_pct)
            disp_view = disp[(disp["within_radius"] == True) | (disp["selected"] == True)].copy()
            disp_view = disp_view.sort_values(
                by=["selected", "within_radius", "travel_minutes"], ascending=[False, False, True]
            )
            cols_to_show = ["district_id", "Distance (km)", "Travel time (min)"]
            if overlay_type == "Population heatmap" and enable_acs:
                cols_to_show.append("Population")
            elif overlay_type == "House margin":
                cols_to_show.append("House margin (R-D)")
            if not disp_view.empty:
                st.markdown("**Reachable / Selected districts**")
                st.dataframe(
                    disp_view[cols_to_show].rename(columns={"district_id": "District"}).reset_index(drop=True),
                    use_container_width=True,
                    height=320,
                )
            else:
                st.info(
                    "No districts fall within the selected travel radius. Adjust the starting point, time or mode to explore more districts."
                )
        else:
            st.info("US-wide shapes could not be loaded; travel map is unavailable.")
    # end of legacy travel code

# After rendering all other sections, display the interactive travel canvas tab.
render_travel_canvas(year, year_data, enable_acs)

# ----------------------------
# BUILD ALL YEAR DATA ONCE
# ----------------------------
@st.cache_data(show_spinner=True)
def build_year_data(pres_path, house_path, spend_xlsx_path, war_csv_path,
                    enable_acs: bool, acs_requested_year: int, census_api_key: str | None):
    pres_df, pres_cand_col, house_df = load_inputs(pres_path, house_path)
    cook_map, sabato_map, inside_map = get_2026_ratings_maps()
    ratings_union = build_ratings_union_table(cook_map, sabato_map, inside_map)
    spend_dist, spend_state = load_fec_spending(spend_xlsx_path)
    war_dist_year = load_war_by_district_year(war_csv_path)

    acs_used_year, acs_cd, acs_state, acs_tried, acs_errors, acs_var_notes = (np.nan, pd.DataFrame(), pd.DataFrame(), [], {}, [])
    if enable_acs:
        acs_used_year, acs_cd, acs_state, acs_tried, acs_errors, acs_var_notes = load_acs_with_fallback(int(acs_requested_year), census_api_key)

    YEARS = [2016, 2018, 2020, 2022, 2024]
    year_data = {}

    for y in YEARS:
        pres_state = compute_pres_state_results(pres_df, pres_cand_col, y)
        dist_year, house_avg = compute_house_state_avg(house_df, y)
        dist_year = attach_ratings(dist_year, cook_map, sabato_map, inside_map)

        if not war_dist_year.empty:
            wy = war_dist_year[war_dist_year["year"] == y].copy()
            wy = wy.drop(columns=["year"], errors="ignore")
            dist_year = dist_year.merge(wy, on="district_id", how="left")

        if enable_acs and not acs_cd.empty:
            dist_year = dist_year.merge(
                acs_cd.drop(columns=["state_po"], errors="ignore"),
                on="district_id",
                how="left"
            )

        if not spend_dist.empty:
            sd = spend_dist[spend_dist["cycle_year"] == y].copy()
            sd = sd.drop(columns=["cycle_year", "state_po"], errors="ignore")
            dist_year = dist_year.merge(sd, on="district_id", how="left")

        sdf = pres_state.merge(house_avg, on="state_po", how="outer")

        if not spend_state.empty:
            ss = spend_state[spend_state["cycle_year"] == y].copy()
            ss = ss.drop(columns=["cycle_year"], errors="ignore")
            sdf = sdf.merge(ss, on="state_po", how="left")

        if enable_acs and not acs_state.empty:
            sdf = sdf.merge(acs_state, on="state_po", how="left")

        sdf = sdf.sort_values("state_po").reset_index(drop=True)

        if not sdf.empty:
            sdf["pres_dem_votes_str"] = sdf.get("pres_dem_votes", np.nan).map(fmt_int)
            sdf["pres_rep_votes_str"] = sdf.get("pres_rep_votes", np.nan).map(fmt_int)
            sdf["pres_total_votes_all_str"] = sdf.get("pres_total_votes_all", np.nan).map(fmt_int)
            sdf["pres_dem_pct_all_str"] = sdf.get("pres_dem_pct_all", np.nan).map(fmt_pct)
            sdf["pres_rep_pct_all_str"] = sdf.get("pres_rep_pct_all", np.nan).map(fmt_pct)
            sdf["pres_margin_str"] = sdf.get("pres_margin", np.nan).map(fmt_pct)
            sdf["avg_house_margin_str"] = sdf.get("avg_house_margin", np.nan).map(fmt_pct)

            for col in [
                "fec_disburse_democrat","fec_disburse_republican","fec_disburse_all","fec_disburse_margin",
                "fec_receipts_democrat","fec_receipts_republican","fec_receipts_all","fec_receipts_margin",
            ]:
                if col not in sdf.columns:
                    sdf[col] = np.nan

            sdf["fec_disburse_democrat_str"] = sdf["fec_disburse_democrat"].map(fmt_money)
            sdf["fec_disburse_republican_str"] = sdf["fec_disburse_republican"].map(fmt_money)
            sdf["fec_disburse_all_str"] = sdf["fec_disburse_all"].map(fmt_money)
            sdf["fec_disburse_margin_str"] = sdf["fec_disburse_margin"].map(fmt_pct)

            sdf["fec_receipts_democrat_str"] = sdf["fec_receipts_democrat"].map(fmt_money)
            sdf["fec_receipts_republican_str"] = sdf["fec_receipts_republican"].map(fmt_money)
            sdf["fec_receipts_all_str"] = sdf["fec_receipts_all"].map(fmt_money)
            sdf["fec_receipts_margin_str"] = sdf["fec_receipts_margin"].map(fmt_pct)

            if enable_acs:
                if "acs_median_hh_income" in sdf.columns:
                    sdf["acs_median_hh_income_str"] = sdf["acs_median_hh_income"].map(fmt_money)
                if "acs_median_age" in sdf.columns:
                    sdf["acs_median_age_str"] = sdf["acs_median_age"].map(lambda v: "" if pd.isna(v) else f"{float(v):.1f}")
                for pc in [
                    "acs_pct_bachelors_or_higher", "acs_pct_veteran",
                    "acs_pct_white_alone", "acs_pct_black_alone", "acs_pct_asian_alone", "acs_pct_hispanic",
                    "acs_pct_male", "acs_pct_female",
                ]:
                    if pc in sdf.columns:
                        sdf[pc + "_str"] = sdf[pc].map(fmt_pct)

            for c in [
                "pres_dem_candidate","pres_rep_candidate",
                "pres_dem_votes_str","pres_rep_votes_str","pres_total_votes_all_str",
                "pres_dem_pct_all_str","pres_rep_pct_all_str",
                "pres_margin_str","avg_house_margin_str",
                "fec_disburse_democrat_str","fec_disburse_republican_str","fec_disburse_all_str","fec_disburse_margin_str",
                "fec_receipts_democrat_str","fec_receipts_republican_str","fec_receipts_all_str","fec_receipts_margin_str",
                "acs_median_hh_income_str","acs_median_age_str",
                "acs_pct_bachelors_or_higher_str","acs_pct_veteran_str",
                "acs_pct_white_alone_str","acs_pct_black_alone_str","acs_pct_asian_alone_str","acs_pct_hispanic_str",
                "acs_pct_male_str","acs_pct_female_str",
            ]:
                if c in sdf.columns:
                    sdf[c] = sdf[c].fillna("").astype(str)

        year_data[y] = {"state_df": sdf, "dist_df": dist_year}

    pref_year = 2024 if not year_data[2024]["dist_df"].empty else (2022 if not year_data[2022]["dist_df"].empty else (2020 if not year_data[2020]["dist_df"].empty else 2016))
    dist_for_toss = year_data[pref_year]["dist_df"]

    if not dist_for_toss.empty:
        base_cols = [
            "district_id",
            "dem_candidate","rep_candidate",
            "dem_votes","rep_votes","total_votes_all",
            "dem_pct_all","rep_pct_all",
            "Cook_2026","Sabato_2026","Inside_2026",
            "tossup_agree_count","house_margin",
            "war_str","war_sortable","war_dem_candidate","war_rep_candidate",
        ]
        acs_cols = [c for c in ACS_OUTPUT_COLS if c in dist_for_toss.columns]
        fec_cols = [c for c in dist_for_toss.columns if c.startswith("fec_")]

        cols = [c for c in base_cols if c in dist_for_toss.columns] + acs_cols + [c for c in fec_cols if c not in base_cols]

        tossup_table = (
            dist_for_toss.loc[dist_for_toss.get("tossup_agree_count", 0) > 0, cols]
            .sort_values(["tossup_agree_count","district_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
    else:
        tossup_table = pd.DataFrame()

    return year_data, tossup_table, ratings_union, acs_used_year, acs_tried, acs_errors, acs_var_notes

# ----------------------------
# PLOTTERS
# ----------------------------
def make_state_map_figure(sdf, year, metric_col):
    if sdf.empty:
        return None

    if metric_col == "pres_margin":
        if "pres_margin" not in sdf.columns or sdf["pres_margin"].notna().sum() == 0:
            metric_col = "avg_house_margin"

    sdf = sdf.copy()
    sdf["_plot_val"] = safe_plot_col(sdf.get(metric_col, pd.Series([None]*len(sdf))))
    arr = pd.to_numeric(sdf["_plot_val"], errors="coerce")
    zmax = float(np.nanmax(np.abs(arr.values))) if np.isfinite(arr).any() else 0.5
    if not np.isfinite(zmax) or zmax == 0:
        zmax = 0.5

    title = f"{year} Presidential Margin by State (Rep - Dem)" if metric_col == "pres_margin" else f"{year} Avg House Margin by State (Rep - Dem)"
    subtitle = "blue = Dem, red = Rep"

    def col_or_blank(c):
        return sdf[c].fillna("").astype(str) if c in sdf.columns else pd.Series([""]*len(sdf))

    st_ = col_or_blank("state_po")
    dname = col_or_blank("pres_dem_candidate")
    rname = col_or_blank("pres_rep_candidate")
    dv = col_or_blank("pres_dem_votes_str")
    rv = col_or_blank("pres_rep_votes_str")
    dt = col_or_blank("pres_dem_pct_all_str")
    rt = col_or_blank("pres_rep_pct_all_str")
    pm = col_or_blank("pres_margin_str")
    hm = col_or_blank("avg_house_margin_str")

    fig = go.Figure(
        data=[
            go.Choropleth(
                locations=sdf["state_po"],
                locationmode="USA-states",
                z=sdf["_plot_val"],
                zmin=-zmax, zmax=zmax,
                colorscale="RdBu_r",
                colorbar_title=("Pres margin" if metric_col=="pres_margin" else "Avg House margin"),
                customdata=np.stack([st_, dname, dv, dt, rname, rv, rt, pm, hm], axis=1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Pres (D): %{customdata[1]} — %{customdata[2]} (%{customdata[3]})<br>"
                    "Pres (R): %{customdata[4]} — %{customdata[5]} (%{customdata[6]})<br>"
                    "Pres margin (R-D): %{customdata[7]}<br>"
                    "Avg House margin (R-D): %{customdata[8]}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title_text=f"{title} — {subtitle}",
        geo=dict(scope="usa", projection_type="albers usa"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=520
    )
    return fig

def make_district_map_figure(state_po, year, sub, spend_measure: str, include_acs: bool):
    geojson, gdf = load_state_cd_geojson(year, state_po)

    if spend_measure == "Disbursements":
        dem_sp = "fec_disburse_democrat"
        rep_sp = "fec_disburse_republican"
        all_sp = "fec_disburse_all"
        mar_sp = "fec_disburse_margin"
    else:
        dem_sp = "fec_receipts_democrat"
        rep_sp = "fec_receipts_republican"
        all_sp = "fec_receipts_all"
        mar_sp = "fec_receipts_margin"

    sub = sub.copy()
    for c in [dem_sp, rep_sp, all_sp, mar_sp]:
        if c not in sub.columns:
            sub[c] = np.nan

    sub["fec_dem_sp_str"] = sub[dem_sp].map(fmt_money)
    sub["fec_rep_sp_str"] = sub[rep_sp].map(fmt_money)
    sub["fec_all_sp_str"] = sub[all_sp].map(fmt_money)
    sub["fec_sp_margin_str"] = sub[mar_sp].map(fmt_pct)

    if include_acs:
        if "acs_median_hh_income" in sub.columns:
            sub["acs_median_hh_income_str"] = sub["acs_median_hh_income"].map(fmt_money)
        if "acs_median_age" in sub.columns:
            sub["acs_median_age_str"] = sub["acs_median_age"].map(lambda v: "" if pd.isna(v) else f"{float(v):.1f}")
        for pc in [
            "acs_pct_bachelors_or_higher","acs_pct_veteran",
            "acs_pct_white_alone","acs_pct_black_alone","acs_pct_asian_alone","acs_pct_hispanic",
            "acs_pct_male","acs_pct_female",
        ]:
            if pc in sub.columns:
                sub[pc + "_str"] = sub[pc].map(fmt_pct)
        if "acs_total_pop" in sub.columns:
            sub["acs_total_pop_str"] = sub["acs_total_pop"].map(fmt_int)

    pick_cols = [
        "district_id",
        "house_margin",
        "dem_candidate","rep_candidate",
        "dem_votes_str","rep_votes_str","total_votes_str",
        "dem_pct_all_str","rep_pct_all_str",
        "Cook_2026","Sabato_2026","Inside_2026","tossup_agree_count",
        "war_str","war_sortable",
        "fec_dem_sp_str","fec_rep_sp_str","fec_all_sp_str","fec_sp_margin_str",
    ]
    if include_acs:
        pick_cols += [
            "acs_total_pop_str","acs_median_age_str","acs_median_hh_income_str",
            "acs_pct_bachelors_or_higher_str","acs_pct_veteran_str",
            "acs_pct_white_alone_str","acs_pct_black_alone_str","acs_pct_asian_alone_str","acs_pct_hispanic_str",
            "acs_pct_male_str","acs_pct_female_str"
        ]
    pick_cols = [c for c in pick_cols if c in sub.columns]

    m = gdf[["district_id"]].merge(sub[pick_cols], on="district_id", how="left")

    m["house_margin_plot"] = safe_plot_col(m.get("house_margin", np.nan))
    arr = pd.to_numeric(m["house_margin_plot"], errors="coerce")
    zmax = float(np.nanmax(np.abs(arr.values))) if np.isfinite(arr).any() else 0.5
    if not np.isfinite(zmax) or zmax == 0:
        zmax = 0.5

    hover_data = {
        "district_id": True,
        "house_margin_plot":":.2%",
        "dem_candidate": True,
        "dem_votes_str": True,
        "dem_pct_all_str": True,
        "rep_candidate": True,
        "rep_votes_str": True,
        "rep_pct_all_str": True,
        "total_votes_str": True,
        "Cook_2026": True,
        "Sabato_2026": True,
        "Inside_2026": True,
        "tossup_agree_count": True,
        "war_str": True,
        "war_sortable": True,
        "fec_dem_sp_str": True,
        "fec_rep_sp_str": True,
        "fec_all_sp_str": True,
        "fec_sp_margin_str": True,
    }
    if include_acs:
        for k in [
            "acs_total_pop_str","acs_median_age_str","acs_median_hh_income_str",
            "acs_pct_bachelors_or_higher_str","acs_pct_veteran_str",
            "acs_pct_white_alone_str","acs_pct_black_alone_str","acs_pct_asian_alone_str","acs_pct_hispanic_str",
            "acs_pct_male_str","acs_pct_female_str",
        ]:
            hover_data[k] = True

    fig2 = px.choropleth(
        m,
        geojson=geojson,
        locations="district_id",
        featureidkey="properties.district_id",
        color="house_margin_plot",
        color_continuous_scale="RdBu_r",
        range_color=(-zmax, zmax),
        title=f"{state_po} — {year} House margin by district + 2026 ratings + WAR + FEC {spend_measure}" + (" + ACS" if include_acs else ""),
        hover_data=hover_data,
        scope="usa",
    )
    fig2.update_geos(fitbounds="locations", visible=False)
    fig2.update_layout(margin=dict(l=0, r=0, t=60, b=0), height=520)
    return fig2

# ----------------------------
# SIDEBAR: FILE PATHS + CONTROLS
# ----------------------------
st.sidebar.header("Inputs")

default_pres = "1976-2024-president-extended.csv"
default_house = "1976-2024-house (1).tab"
default_spend = "fec_house_campaign_spending_2016_2018_2020_2022_2024.xlsx"
default_war = "data-Eq2Z0.csv"

pres_path = st.sidebar.text_input("Presidential CSV path", value=default_pres)
house_path = st.sidebar.text_input("House TAB/CSV path", value=default_house)
spend_path = st.sidebar.text_input("FEC spending XLSX path", value=default_spend)
war_path = st.sidebar.text_input("WAR CSV path", value=default_war)

st.sidebar.divider()

YEARS = [2016, 2018, 2020, 2022, 2024]
year = st.sidebar.radio("Year", YEARS, index=0)
metric_label = st.sidebar.radio("State map colors", ["Pres margin", "Avg House margin"], index=0)
metric_col = "pres_margin" if metric_label == "Pres margin" else "avg_house_margin"

spend_measure = st.sidebar.radio("Spending measure (FEC)", ["Disbursements", "Receipts"], index=0)

st.sidebar.divider()
st.sidebar.subheader("ACS (Census API)")

enable_acs = st.sidebar.checkbox("Enable ACS demographics (Census API)", value=True)
acs_requested_year = st.sidebar.number_input(
    "ACS 5-year Profile year (try 2024; auto-fallback)",
    min_value=2010, max_value=2030, value=2024, step=1,
    disabled=(not enable_acs),
)

census_api_key = st.sidebar.text_input(
    "Optional Census API key (recommended, but not required)",
    value="",
    type="password",
    disabled=(not enable_acs),
)

include_acs_in_hover = st.sidebar.checkbox(
    "Include ACS demographics in district hover/map",
    value=True,
    disabled=(not enable_acs),
)

try:
    year_data, tossup_table, ratings_union, acs_used_year, acs_tried, acs_errors, acs_var_notes = build_year_data(
        pres_path, house_path, spend_path, war_path,
        bool(enable_acs), int(acs_requested_year), census_api_key
    )
except Exception as e:
    st.error("Failed to load/parse your input files. Check the paths and file formats.")
    st.exception(e)
    st.stop()

if enable_acs:
    st.sidebar.caption(f"ACS: requested {acs_requested_year} • used {acs_used_year} • tried {acs_tried}")
    if acs_errors:
        with st.sidebar.expander("ACS errors (if any)"):
            for k, v in acs_errors.items():
                st.write(f"{k}: {v}")
    with st.sidebar.expander("ACS variables resolved (debug)"):
        for line in (acs_var_notes or []):
            st.write(line)

sdf = year_data[year]["state_df"]
if sdf.empty:
    st.error("No state-level data for the selected year.")
    st.stop()

states = sorted([s for s in sdf["state_po"].dropna().unique().tolist() if isinstance(s, str) and len(s)==2])
state_po = st.sidebar.selectbox("State", states, index=0)

# ----------------------------
# MAIN UI
# ----------------------------
st.title("US Elections Explorer (2016 / 2018 / 2020 / 2022 / 2024)")

left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.subheader("State map")
    fig = make_state_map_figure(sdf, year, metric_col)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader(f"{state_po} summary ({year})")

    row = sdf[sdf["state_po"] == state_po]
    if row.empty:
        st.info("No row for this state/year.")
    else:
        r0 = row.iloc[0]
        has_pres = bool(r0.get("pres_dem_votes_str","") or r0.get("pres_rep_votes_str",""))

        pres_block = ""
        if has_pres:
            pres_block = f"""
**Pres (D):** {r0.get("pres_dem_candidate","")} — {r0.get("pres_dem_votes_str","")} ({r0.get("pres_dem_pct_all_str","")})
**Pres (R):** {r0.get("pres_rep_candidate","")} — {r0.get("pres_rep_votes_str","")} ({r0.get("pres_rep_pct_all_str","")})
**Pres margin (Rep − Dem):** {r0.get("pres_margin_str","N/A")}
""".strip()

        if spend_measure == "Disbursements":
            fec_dem = r0.get("fec_disburse_democrat_str","")
            fec_rep = r0.get("fec_disburse_republican_str","")
            fec_all = r0.get("fec_disburse_all_str","")
            fec_mar = r0.get("fec_disburse_margin_str","")
        else:
            fec_dem = r0.get("fec_receipts_democrat_str","")
            fec_rep = r0.get("fec_receipts_republican_str","")
            fec_all = r0.get("fec_receipts_all_str","")
            fec_mar = r0.get("fec_receipts_margin_str","")

        fec_block = f"""
**FEC {spend_measure} (House candidates, state total):**
• Dem: {fec_dem} • Rep: {fec_rep} • Total (all parties): {fec_all} • Margin (Rep − Dem): {fec_mar}
""".strip()

        acs_block = ""
        if enable_acs and ("acs_median_hh_income_str" in sdf.columns or "acs_pct_bachelors_or_higher_str" in sdf.columns):
            acs_block = f"""
**ACS (5-year Data Profile, state):**
• Median HH income: {r0.get("acs_median_hh_income_str","")}
• Median age: {r0.get("acs_median_age_str","")}
• % Bachelor+ (25+): {r0.get("acs_pct_bachelors_or_higher_str","")}
• % Veteran: {r0.get("acs_pct_veteran_str","")}
""".strip()

        st.markdown(
            f"""
{pres_block}
**Avg House margin (Rep − Dem):** {r0.get("avg_house_margin_str","N/A")}

{fec_block}

{acs_block}
""".strip()
        )

st.divider()

st.subheader(f"{state_po} districts ({year})")
ddf = year_data[year]["dist_df"]
if ddf.empty:
    st.info("No district-level House results for this year.")
    st.stop()

sub = ddf[ddf["state_po"] == state_po].copy()
if sub.empty:
    st.info("No districts found for this state/year.")
    st.stop()

def sort_key(did):
    if str(did).endswith("-AL"):
        return (-1, 0)
    try:
        return (0, int(str(did).split("-")[1]))
    except Exception:
        return (0, 999)

sub["k"] = sub["district_id"].apply(sort_key)
sub = sub.sort_values("k").drop(columns=["k"]).reset_index(drop=True)

sub["dem_votes_str"] = sub.get("dem_votes", np.nan).map(fmt_int)
sub["rep_votes_str"] = sub.get("rep_votes", np.nan).map(fmt_int)
sub["total_votes_str"] = sub.get("total_votes_all", np.nan).map(fmt_int)
sub["dem_pct_all_str"] = sub.get("dem_pct_all", np.nan).map(fmt_pct)
sub["rep_pct_all_str"] = sub.get("rep_pct_all", np.nan).map(fmt_pct)

try:
    fig2 = make_district_map_figure(state_po, year, sub, spend_measure, (enable_acs and include_acs_in_hover))
    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.warning("District map unavailable (could not load Census district shapes or plot them).")
    st.exception(e)

if spend_measure == "Disbursements":
    dem_sp = "fec_disburse_democrat"
    rep_sp = "fec_disburse_republican"
    all_sp = "fec_disburse_all"
    mar_sp = "fec_disburse_margin"
else:
    dem_sp = "fec_receipts_democrat"
    rep_sp = "fec_receipts_republican"
    all_sp = "fec_receipts_all"
    mar_sp = "fec_receipts_margin"

for c in [dem_sp, rep_sp, all_sp, mar_sp]:
    if c not in sub.columns:
        sub[c] = np.nan

acs_cols = [c for c in ACS_OUTPUT_COLS if c in sub.columns] if enable_acs else []

show_cols = [
    "district_id",
    "dem_candidate","dem_votes","dem_pct_all",
    "rep_candidate","rep_votes","rep_pct_all",
    "total_votes_all",
    "house_margin",
    "Cook_2026","Sabato_2026","Inside_2026","tossup_agree_count",
    "war_str","war_sortable","war_dem_candidate","war_rep_candidate",
    dem_sp, rep_sp, all_sp, mar_sp,
] + acs_cols
show_cols = [c for c in show_cols if c in sub.columns]

show = sub[show_cols].copy()

for c in ["dem_votes","rep_votes","total_votes_all"]:
    if c in show.columns:
        show[c] = show[c].map(fmt_int)
for c in ["dem_pct_all","rep_pct_all","house_margin", mar_sp]:
    if c in show.columns:
        show[c] = show[c].map(fmt_pct)
for c in [dem_sp, rep_sp, all_sp]:
    if c in show.columns:
        show[c] = show[c].map(fmt_money)

if "war_sortable" in show.columns:
    show["war_sortable"] = show["war_sortable"].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")

if enable_acs:
    if "acs_total_pop" in show.columns:
        show["acs_total_pop"] = show["acs_total_pop"].map(fmt_int)
    if "acs_median_hh_income" in show.columns:
        show["acs_median_hh_income"] = show["acs_median_hh_income"].map(fmt_money)
    if "acs_median_age" in show.columns:
        show["acs_median_age"] = show["acs_median_age"].map(lambda v: "" if pd.isna(v) else f"{float(v):.1f}")
    for pc in [
        "acs_pct_male","acs_pct_female",
        "acs_pct_white_alone","acs_pct_black_alone","acs_pct_asian_alone","acs_pct_hispanic",
        "acs_pct_bachelors_or_higher","acs_pct_veteran"
    ]:
        if pc in show.columns:
            show[pc] = show[pc].map(fmt_pct)

rename_map = {
    dem_sp: f"FEC {spend_measure} (Dem)",
    rep_sp: f"FEC {spend_measure} (Rep)",
    all_sp: f"FEC {spend_measure} (Total all parties)",
    mar_sp: f"FEC {spend_measure} margin (Rep−Dem)",
    "war_str": "WAR (string)",
    "war_sortable": "WAR (sortable)",
    "war_dem_candidate": "WAR Dem name",
    "war_rep_candidate": "WAR Rep name",
    "acs_total_pop": "ACS total pop",
    "acs_pct_male": "ACS % male",
    "acs_pct_female": "ACS % female",
    "acs_median_age": "ACS median age",
    "acs_pct_white_alone": "ACS % White (alone)",
    "acs_pct_black_alone": "ACS % Black (alone)",
    "acs_pct_asian_alone": "ACS % Asian (alone)",
    "acs_pct_hispanic": "ACS % Hispanic",
    "acs_median_hh_income": "ACS median HH income",
    "acs_pct_bachelors_or_higher": "ACS % Bachelor+ (25+)",
    "acs_pct_veteran": "ACS % veteran (18+)",
}
show = show.rename(columns=rename_map)
st.dataframe(show, use_container_width=True, height=420)

st.subheader("Toss-ups (filtered to this state)")
if isinstance(tossup_table, pd.DataFrame) and not tossup_table.empty:
    st_toss = tossup_table[tossup_table["district_id"].str.startswith(state_po + "-", na=False)].copy()
    if st_toss.empty:
        st.info("No toss-up districts (by your 3 sources) found for this state in the scraped tables.")
    else:
        st_toss_disp = st_toss.copy()

        for c in ["dem_votes","rep_votes","total_votes_all"]:
            if c in st_toss_disp.columns:
                st_toss_disp[c] = st_toss_disp[c].map(fmt_int)
        for c in ["dem_pct_all","rep_pct_all","house_margin", "fec_disburse_margin", "fec_receipts_margin"]:
            if c in st_toss_disp.columns:
                st_toss_disp[c] = st_toss_disp[c].map(fmt_pct)
        for c in st_toss_disp.columns:
            if c.startswith("fec_") and ("disburse" in c or "receipts" in c) and not c.endswith("margin"):
                st_toss_disp[c] = st_toss_disp[c].map(fmt_money)

        if enable_acs:
            if "acs_total_pop" in st_toss_disp.columns:
                st_toss_disp["acs_total_pop"] = st_toss_disp["acs_total_pop"].map(fmt_int)
            if "acs_median_hh_income" in st_toss_disp.columns:
                st_toss_disp["acs_median_hh_income"] = st_toss_disp["acs_median_hh_income"].map(fmt_money)
            if "acs_median_age" in st_toss_disp.columns:
                st_toss_disp["acs_median_age"] = st_toss_disp["acs_median_age"].map(lambda v: "" if pd.isna(v) else f"{float(v):.1f}")
            for pc in [
                "acs_pct_male","acs_pct_female",
                "acs_pct_white_alone","acs_pct_black_alone","acs_pct_asian_alone","acs_pct_hispanic",
                "acs_pct_bachelors_or_higher","acs_pct_veteran"
            ]:
                if pc in st_toss_disp.columns:
                    st_toss_disp[pc] = st_toss_disp[pc].map(fmt_pct)

        st.dataframe(st_toss_disp, use_container_width=True, height=260)
else:
    st.info("No toss-up table available (ratings scrape returned no districts).")

# ----------------------------
# Ratings Universe view
# ----------------------------
st.divider()
st.subheader("Ratings universe (Cook / Sabato / Inside from 270toWin) — leans / tilts / toss-ups / likely")

if ratings_union is None or ratings_union.empty:
    st.info("No districts were parsed from the 270toWin rating tables.")
else:
    context = year_data[year]["dist_df"].copy()

    base_merge_cols = [
        "district_id",
        "dem_candidate","rep_candidate",
        "dem_votes","rep_votes","total_votes_all",
        "dem_pct_all","rep_pct_all",
        "house_margin",
        "tossup_agree_count",
        "war_str","war_sortable","war_dem_candidate","war_rep_candidate",
    ]
    fec_cols = [c for c in context.columns if c.startswith("fec_")]
    acs_cols2 = [c for c in ACS_OUTPUT_COLS if c in context.columns] if enable_acs else []

    merged = ratings_union.merge(
        context[[c for c in base_merge_cols if c in context.columns] + fec_cols + acs_cols2],
        on="district_id",
        how="left"
    )

    c1, c2, c3, c4 = st.columns([1.1, 1.0, 1.0, 1.3])
    with c1:
        state_only = st.checkbox("Only selected state", value=True)
    with c2:
        min_mention = st.selectbox("Mentioned by (>=)", [1,2,3], index=0)
    with c3:
        consensus_filter = st.selectbox("Consensus side", ["All", "Dem", "Rep", "Toss-up"], index=0)
    with c4:
        competitive_only = st.checkbox("Only Toss-up/Tilt (any source)", value=False)

    disagree_only = st.checkbox("Only show disagreements (side_agree_max < mentioned_by_count)", value=False)

    view = merged[merged["mentioned_by_count"] >= min_mention].copy()
    if state_only:
        view = view[view["district_id"].astype(str).str.startswith(state_po + "-", na=False)]
    if consensus_filter != "All":
        view = view[view["consensus_side"] == consensus_filter]
    if competitive_only:
        view = view[view["any_tossup_or_tilt"] == 1]
    if disagree_only:
        view = view[view["side_agree_max"] < view["mentioned_by_count"]]

    for c in ["dem_votes","rep_votes","total_votes_all"]:
        if c in view.columns:
            view[c] = view[c].map(fmt_int)
    for c in ["dem_pct_all","rep_pct_all","house_margin"]:
        if c in view.columns:
            view[c] = view[c].map(fmt_pct)
    for c in view.columns:
        if c.startswith("fec_") and ("disburse" in c or "receipts" in c) and not c.endswith("margin"):
            view[c] = view[c].map(fmt_money)
        if c.startswith("fec_") and c.endswith("margin"):
            view[c] = view[c].map(fmt_pct)

    if enable_acs:
        if "acs_total_pop" in view.columns:
            view["acs_total_pop"] = view["acs_total_pop"].map(fmt_int)
        if "acs_median_hh_income" in view.columns:
            view["acs_median_hh_income"] = view["acs_median_hh_income"].map(fmt_money)
        if "acs_median_age" in view.columns:
            view["acs_median_age"] = view["acs_median_age"].map(lambda v: "" if pd.isna(v) else f"{float(v):.1f}")
        for pc in [
            "acs_pct_male","acs_pct_female",
            "acs_pct_white_alone","acs_pct_black_alone","acs_pct_asian_alone","acs_pct_hispanic",
            "acs_pct_bachelors_or_higher","acs_pct_veteran"
        ]:
            if pc in view.columns:
                view[pc] = view[pc].map(fmt_pct)

    if "war_sortable" in view.columns:
        view["war_sortable"] = view["war_sortable"].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")

    view["_disagree"] = (view["side_agree_max"] < view["mentioned_by_count"]).astype(int)
    view = view.sort_values(
        by=["mentioned_by_count","_disagree","any_tossup_or_tilt","district_id"],
        ascending=[False, False, False, True]
    ).drop(columns=["_disagree"], errors="ignore")

    core_cols = [
        "district_id",
        "Cook_2026","Sabato_2026","Inside_2026",
        "mentioned_by_count","side_agree_max","exact_label_agree_max",
        "consensus_by_avgscore","avg_score",
        "dem_candidate","rep_candidate","house_margin",
        "war_str","war_sortable","war_dem_candidate","war_rep_candidate",
    ]

    if spend_measure == "Disbursements":
        fec_pick = ["fec_disburse_democrat","fec_disburse_republican","fec_disburse_all","fec_disburse_margin"]
    else:
        fec_pick = ["fec_receipts_democrat","fec_receipts_republican","fec_receipts_all","fec_receipts_margin"]

    acs_pick = [
        "acs_total_pop","acs_median_age","acs_median_hh_income",
        "acs_pct_bachelors_or_higher","acs_pct_veteran",
        "acs_pct_white_alone","acs_pct_black_alone","acs_pct_asian_alone","acs_pct_hispanic",
        "acs_pct_male","acs_pct_female",
    ] if enable_acs else []

    show_cols = [c for c in core_cols if c in view.columns] + [c for c in fec_pick if c in view.columns] + [c for c in acs_pick if c in view.columns]

    rename = {
        "mentioned_by_count": "Mentioned by (count)",
        "side_agree_max": "Side agree (max)",
        "exact_label_agree_max": "Exact label agree (max)",
        "consensus_by_avgscore": "Consensus (avg score)",
        "avg_score": "Avg score (Dem - / Rep +)",
        "war_str": "WAR (string)",
        "war_sortable": "WAR (sortable)",
        "war_dem_candidate": "WAR Dem name",
        "war_rep_candidate": "WAR Rep name",
        "fec_disburse_democrat": "FEC Disburse (Dem)",
        "fec_disburse_republican": "FEC Disburse (Rep)",
        "fec_disburse_all": "FEC Disburse (All parties)",
        "fec_disburse_margin": "FEC Disburse margin (Rep−Dem)",
        "fec_receipts_democrat": "FEC Receipts (Dem)",
        "fec_receipts_republican": "FEC Receipts (Rep)",
        "fec_receipts_all": "FEC Receipts (All parties)",
        "fec_receipts_margin": "FEC Receipts margin (Rep−Dem)",
        "acs_total_pop": "ACS total pop",
        "acs_pct_male": "ACS % male",
        "acs_pct_female": "ACS % female",
        "acs_median_age": "ACS median age",
        "acs_pct_white_alone": "ACS % White (alone)",
        "acs_pct_black_alone": "ACS % Black (alone)",
        "acs_pct_asian_alone": "ACS % Asian (alone)",
        "acs_pct_hispanic": "ACS % Hispanic",
        "acs_median_hh_income": "ACS median HH income",
        "acs_pct_bachelors_or_higher": "ACS % Bachelor+ (25+)",
        "acs_pct_veteran": "ACS % veteran (18+)",
    }

    st.dataframe(view[show_cols].rename(columns=rename), use_container_width=True, height=520)

# ----------------------------
# Notes / warnings
# ----------------------------
if spend_path and not Path(spend_path).exists():
    st.warning("FEC spending XLSX path not found. Add the file to the repo (same folder as app.py) or correct the path.")
if war_path and not Path(war_path).exists():
    st.warning("WAR CSV path not found. Add the file to the repo (same folder as app.py) or correct the path.")

if enable_acs:
    st.caption(
        "Notes: Presidential stats only exist for presidential years (2016/2020/2024); midterms show House + FEC spending. "
        "Ratings are scraped ONLY from the 3x 270toWin tables (Cook/Sabato/Inside). "
        "District shapes are cached locally. "
        f"ACS demographics come from the U.S. Census Bureau ACS 5-year *Data Profile* via the Census Data API "
        f"(requested {acs_requested_year}; used {acs_used_year}; tried {acs_tried}). "
        "IMPORTANT: DP05 variable *codes* change across years; this app resolves the correct variables by LABEL each run/year. "
        "Ratings fix: canonicalizes Toss-up so individual Toss-up ratings populate correctly."
    )
else:
    st.caption(
        "Notes: Presidential stats only exist for presidential years (2016/2020/2024); midterms show House + FEC spending. "
        "Ratings are scraped ONLY from the 3x 270toWin tables (Cook/Sabato/Inside). "
        "District shapes are cached locally. "
        "ACS is disabled."
    )

    # Legacy Travel Canvas implementation retained for backward compatibility.
    # This block is disabled because a newer travel canvas implementation is provided later.
    # If you wish to restore the legacy canvassing tool, set the condition below to True.
    if False:
        # -------------------------------------------------------------
        # Travel Canvas / Canvassing tool
        # -------------------------------------------------------------
        st.divider()
        # Introduce a new tab specifically for the canvassing/travel map.
        travel_tab = st.tabs(["Travel Map / Canvas"])[0]
        with travel_tab:
            st.subheader("Travel Map & District Reachability")
            st.write(
                """
                Use this tool to drop a starting point anywhere in the continental United States and explore
                which congressional districts fall within a selected travel time.  Districts are coloured
                either by total population (via ACS data) or by the House margin (Rep − Dem) for the
                selected year.  You can highlight specific districts manually and choose the mode of
                transportation to adjust the travel radius.
                """
            )

            # Overlay selection: population or house margin
            overlay_type = st.radio(
                "Colour districts by", ["Population heatmap", "House margin"], index=0
            )
            # Transportation mode and corresponding speed definitions
            transport_mode = st.radio(
                "Transportation mode", ["Driving", "Walking", "Cycling"], index=0
            )
            time_minutes = st.slider(
                "Travel time (minutes)", min_value=5, max_value=120, value=60, step=5
            )

            # Location input – default to approximate center of the contiguous US
            st.markdown("**Starting location (latitude & longitude)**")
            col_lat, col_lon = st.columns(2)
            with col_lat:
                lat = st.number_input(
                    "Latitude", min_value=-90.0, max_value=90.0, value=39.5, step=0.1,
                )
            with col_lon:
                lon = st.number_input(
                    "Longitude", min_value=-180.0, max_value=180.0, value=-98.35, step=0.1,
                )

            # Allow users to manually highlight districts
            all_districts = (
                year_data[year]["dist_df"]["district_id"].dropna().astype(str).unique().tolist()
            )
            all_districts = sorted(all_districts)
            selected_districts = st.multiselect(
                "Districts to highlight (optional)", options=all_districts, default=[]
            )

            # Load US-wide district shapes once for the selected year
            try:
                us_gdf, us_geojson = load_us_cd_shapes(year)
            except Exception as e:
                st.error(f"Unable to load US district shapes for year {year}.")
                st.exception(e)
                us_gdf, us_geojson = gpd.GeoDataFrame(), {}

            # Build and display the travel map figure
            if not us_gdf.empty:
                fig_travel = make_travel_map_figure(
                    year,
                    overlay_type,
                    transport_mode,
                    time_minutes,
                    lat,
                    lon,
                    selected_districts,
                    year_data[year]["dist_df"],
                    us_gdf,
                    us_geojson,
                    enable_acs,
                )
                st.plotly_chart(fig_travel, use_container_width=True)

                # Compute distances and travel times for each district and present a summary table
                # Use the same speed definitions as the figure function
                speed_map = {"Driving": 96.56064, "Walking": 4.82803, "Cycling": 24.14016}
                speed_kmh = speed_map.get(transport_mode, 96.56064)
                radius_km = float(speed_kmh) * (time_minutes / 60.0)
                # Prepare a dataframe with centroids and distances
                temp = us_gdf[["district_id", "centroid_lat", "centroid_lon"]].copy()
                temp["distance_km"] = temp.apply(
                    lambda row: _haversine(lat, lon, row.get("centroid_lat", np.nan), row.get("centroid_lon", np.nan)),
                    axis=1,
                )
                temp["travel_minutes"] = (temp["distance_km"] / speed_kmh) * 60.0
                # Merge overlay value for display
                merge_val = None
                if overlay_type == "Population heatmap" and enable_acs and "acs_total_pop" in year_data[year]["dist_df"].columns:
                    merge_val = year_data[year]["dist_df"][["district_id", "acs_total_pop"]].copy().rename(columns={"acs_total_pop": "overlay"})
                elif overlay_type == "House margin" and "house_margin" in year_data[year]["dist_df"].columns:
                    merge_val = year_data[year]["dist_df"][["district_id", "house_margin"]].copy().rename(columns={"house_margin": "overlay"})
                if merge_val is not None:
                    temp = temp.merge(merge_val, on="district_id", how="left")
                # Identify reachable and/or selected districts
                temp["within_radius"] = temp["distance_km"] <= radius_km
                temp["selected"] = temp["district_id"].isin(selected_districts)
                # Format columns for display
                disp = temp.copy()
                disp["Distance (km)"] = disp["distance_km"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "")
                disp["Travel time (min)"] = disp["travel_minutes"].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "")
                if overlay_type == "Population heatmap" and enable_acs:
                    disp["Population"] = disp["overlay"].apply(fmt_int)
                elif overlay_type == "House margin":
                    disp["House margin (R-D)"] = disp["overlay"].apply(fmt_pct)
                # Filter to reachable districts or selected districts for brevity
                disp_view = disp[(disp["within_radius"] == True) | (disp["selected"] == True)].copy()
                # Sort: selected first, then by travel time
                disp_view = disp_view.sort_values(
                    by=["selected", "within_radius", "travel_minutes"], ascending=[False, False, True]
                )
                # Columns to show
                cols_to_show = ["district_id", "Distance (km)", "Travel time (min)"]
                if overlay_type == "Population heatmap" and enable_acs:
                    cols_to_show.append("Population")
                elif overlay_type == "House margin":
                    cols_to_show.append("House margin (R-D)")
                if not disp_view.empty:
                    st.markdown("**Reachable / Selected districts**")
                    st.dataframe(
                        disp_view[cols_to_show].rename(columns={"district_id": "District"}).reset_index(drop=True),
                        use_container_width=True,
                        height=320,
                    )
                else:
                    st.info(
                        "No districts fall within the selected travel radius. Adjust the starting point, time or mode to explore more districts."
                    )
            else:
                st.info("US-wide shapes could not be loaded; travel map is unavailable.")
