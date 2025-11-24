import os
import re
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from base64 import b64encode

# -------------------- Paths --------------------
DATA_DIR = Path("data")
DIGITAL_CHANNELS_CSV = DATA_DIR / "digital_channels.csv"
DEFAULT_USAGE_MATRIX_CSV = DATA_DIR / "default_usage_matrix.csv"
DIGITAL_PAIRS_MATRIX_CSV = DATA_DIR / "digital_pairs_matrix.csv"

# -------------------- Page setup --------------------
st.set_page_config(page_title="Cross-Reach Calculator",
                   page_icon="mindshare_lithuania_logo.jpg",
                   layout="centered")

# logo is optional; don't crash if missing
if os.path.exists("mindshare_lithuania_logo.jpg"):
    with open("mindshare_lithuania_logo.jpg", "rb") as _f:
        _data_uri = f"data:image/jpeg;base64,{b64encode(_f.read()).decode()}"
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px; margin:0 0 8px 0;">
            <img src="{_data_uri}" style="height:40px; border-radius:8px;" />
            <h1 style="margin:0;">Cross-Reach Calculator</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("<h1>Cross-Reach Calculator</h1>", unsafe_allow_html=True)


def gray_notice(message: str):
    st.markdown(
        f"""
        <div style="
            background-color:#F2F2F2;
            padding:12px 16px;
            border-radius:8px;
            border:1px solid #e6e6e6;
            font-size:0.95rem;
            line-height:1.4;
        ">{message}</div>
        """,
        unsafe_allow_html=True,
    )

# -------------------- Attention adjustment indexes (internal) --------------------
ADJ = {
    "Cinema": 0.98, "Direct mail": 0.56, "Influencers": 0.20, "Magazines": 0.56,
    "Newspapers": 0.29, "News portals": 0.20, "OOH": 0.41, "Podcasts": 0.61,
    "POS (Instore)": 0.42, "Radio": 0.40, "Search": 0.61, "Social media": 0.20,
    "TV": 0.55, "VOD": 0.61,
}

# -------------------- Mode first --------------------
MODE_LABELS = [
    "Independence (Sainsbury)",
    "Overlap-aware (Sainsbury + monthly usage)",
    "Overlap-aware (Digital pairs matrix)",
]
mode = st.radio("Choose a mode", MODE_LABELS)

# -------------------- helpers --------------------
def pct_to_unit(x):
    """
    STRICT: interpret user-entered numbers as percent in [0, 100].
    0.3 -> 0.003 (0.3%), 30 -> 0.30 (30%).
    """
    if x is None or x == "":
        return None
    try:
        x = float(str(x).replace(",", "."))
    except Exception:
        return None
    if np.isnan(x):
        return None
    if x < 0 or x > 100:
        return None
    return x / 100.0

def unit_to_pct(x):
    return None if x is None else x * 100.0

def to_unit_df_from_pct(df_pct: pd.DataFrame) -> pd.DataFrame:
    """Vectorized, unambiguous conversion of a % matrix to 0–1 (float)."""
    return (df_pct.astype(float) / 100.0)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

# -------------------- Loaders: digital channels (categories + aliases) --------------------
def _validate_digital_channels_df(df: pd.DataFrame) -> List[str]:
    """Return a list of human-readable warnings for common data quality issues."""
    warnings: List[str] = []
    # missing/blank categories
    if df["category"].isna().any() or (df["category"].astype(str).str.strip() == "").any():
        warnings.append("Some rows have a missing/blank 'category'.")
    # duplicate alias collisions
    alias_map: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        canonical = row["name"]
        aliases = [a.strip() for a in str(row["aliases"]).split("|") if a.strip()]
        for a in aliases:
            n = _norm(a)
            alias_map.setdefault(n, []).append(canonical)
    dup_aliases = {a: c for a, c in alias_map.items() if len(set(c)) > 1}
    if dup_aliases:
        examples = ", ".join(list(dup_aliases.keys())[:5])
        warnings.append(f"Alias collisions detected (same alias mapped to multiple names), e.g., {examples}.")
    return warnings

@st.cache_data(ttl=60)
def load_digital_channels_table(path: Path = DIGITAL_CHANNELS_CSV) -> pd.DataFrame:
    """
    Expected columns: name, category, aliases (pipe-separated: a|b|c)
    """
    if not path.exists():
        alt_path = Path("/mnt/data") / path.name
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Missing digital channels file: {path} (also tried {alt_path}).")
    df = pd.read_csv(path)
    for col in ["name", "category", "aliases"]:
        if col not in df.columns:
            raise ValueError(f"{path} must contain column '{col}'.")
    df["name"] = df["name"].astype(str)
    df["category"] = df["category"].astype(str)
    df["aliases"] = df["aliases"].fillna("").astype(str)
    df["name_norm"] = df["name"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)

    warns = _validate_digital_channels_df(df)
    if warns:
        st.warning(" | ".join(warns))
    return df

@st.cache_data(ttl=60)
def build_digital_lookups() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      name_to_category: normalized name/alias -> category
      alias_to_canonical: normalized alias -> canonical name
    """
    df = load_digital_channels_table()
    name_to_category: Dict[str, str] = {}
    alias_to_canonical: Dict[str, str] = {}

    for _, row in df.iterrows():
        canonical = row["name"]
        cat = row["category"]
        name_to_category[_norm(canonical)] = cat

        aliases = [a.strip() for a in row["aliases"].split("|") if a.strip()]
        for a in aliases:
            n = _norm(a)
            alias_to_canonical[n] = canonical
            name_to_category[n] = cat
    return name_to_category, alias_to_canonical

# Lazy lookups — only load when first needed, and don't crash if file is missing
_name_to_cat: Dict[str, str] = {}
_alias_to_canon: Dict[str, str] = {}
_lookups_ready = False

def _ensure_lookups():
    """Load digital channel lookups once, with graceful fallbacks."""
    global _name_to_cat, _alias_to_canon, _lookups_ready
    if _lookups_ready:
        return
    try:
        _name_to_cat, _alias_to_canon = build_digital_lookups()
    except FileNotFoundError:
        _name_to_cat, _alias_to_canon = {}, {}
    except Exception as e:
        st.warning(f"Digital channels mapping could not be loaded: {e}")
        _name_to_cat, _alias_to_canon = {}, {}
    finally:
        _lookups_ready = True

def canonicalize_channel(ch: str) -> str:
    """Map an alias to its canonical channel name if known."""
    _ensure_lookups()
    n = _norm(ch)
    return _alias_to_canon.get(n, ch)

def digital_category_for(ch: str) -> Optional[str]:
    """Return the category for a channel/alias, if known."""
    _ensure_lookups()
    return _name_to_cat.get(_norm(ch))

def attention_factor_for_channel(ch: str) -> float:
    """Return attention index for a channel or its category; default to 1.0."""
    key = str(ch).strip()
    if key in ADJ:
        return ADJ[key]
    cat = digital_category_for(key)
    if cat and cat in ADJ:
        return ADJ[cat]
    return 1.0


# -------------------- Charts --------------------
def bar_chart(df, x_field, y_field, height=320, color="#000050"):
    auto_height = min(max(height, 36 * max(1, len(df))), 800)
    return (
        alt.Chart(df)
        .mark_bar(color=color)
        .encode(
            x=alt.X(f"{x_field}:N", sort=None, title="Channel"),
            y=alt.Y(f"{y_field}:Q", axis=alt.Axis(format="%", title="Media reach")),
            tooltip=[alt.Tooltip(f"{x_field}:N"), alt.Tooltip(f"{y_field}:Q", format=".1%")],
        )
        .properties(height=auto_height)
    )

# -------------------- Union math --------------------
def compute_union(chans, R_dict, U_matrix):
    """
    chans: list of str
    R_dict: dict channel -> marginal reach (0..1), already clipped to U(a,a)
    U_matrix: DataFrame (0..1) with U(a,a)=monthly users, U(a,b)=monthly users of intersection
    """
    U = U_matrix.copy().astype(float).fillna(0.0)

    # Effective fraction within monthly users for each channel
    r = {}
    for a in chans:
        ua = float(U.loc[a, a]) if (a in U.index and a in U.columns) else 0.0
        r[a] = 0.0 if ua <= 0 else min(1.0, float(R_dict[a]) / ua)

    # Pairwise effective overlap P2 (absolute fraction)
    P2 = pd.DataFrame(index=chans, columns=chans, dtype=float)
    for a in chans:
        P2.loc[a, a] = float(R_dict[a])

    for i, a in enumerate(chans):
        for j, b in enumerate(chans):
            if j <= i:
                continue
            uab = float(U.loc[a, b])
            pab = uab * r[a] * r[b]
            pab = min(pab, float(R_dict[a]), float(R_dict[b]))
            P2.loc[a, b] = P2.loc[b, a] = float(pab)

    sum_R = float(sum(R_dict[c] for c in chans))
    sum_pairs = float(sum(P2.loc[chans[i], chans[j]] for i in range(len(chans)) for j in range(i + 1, len(chans))))
    lower_pair = max(max(R_dict[c] for c in chans), sum_R - sum_pairs)
    upper_simple = min(1.0, sum_R)

    def est_triple(a, b, c):
        R1, R2, R3 = float(R_dict[a]), float(R_dict[b]), float(R_dict[c])
        P_ab, P_ac, P_bc = float(P2.loc[a, b]), float(P2.loc[a, c]), float(P2.loc[b, c])
        denom = max(1e-12, R1 * R2 * R3)
        t = (P_ab * P_ac * P_bc) / denom
        lower = max(0.0, (P_ab + P_ac + P_bc) - R1 - R2 - R3)
        upper = min(P_ab, P_ac, P_bc)
        return float(np.clip(t, lower, upper))

    triple_sum = 0.0
    if len(chans) >= 3:
        for i in range(len(chans)):
            for j in range(i + 1, len(chans)):
                for k in range(j + 1, len(chans)):
                    triple_sum += est_triple(chans[i], chans[j], chans[k])

    est_union = float(np.clip(sum_R - sum_pairs + triple_sum, lower_pair, upper_simple))
    return est_union, lower_pair, upper_simple, P2

# -------------------- CSV loaders (cached) --------------------
def _read_table_any_decimal(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, index_col=0)
    except Exception:
        return pd.read_csv(path, index_col=0, sep=";", decimal=",")

@st.cache_data(ttl=60)
def load_default_usage_matrix(path: Path = DEFAULT_USAGE_MATRIX_CSV) -> pd.DataFrame:
    if not path.exists():
        alt_path = Path("/mnt/data") / path.name
        if alt_path.exists():
            path = alt_path
        else:
            keys = list(ADJ.keys())
            if not keys:
                raise FileNotFoundError(f"Missing usage matrix file: {path} (also tried {alt_path}).")
            st.warning(
                "default_usage_matrix.csv not found; using a neutral fallback (diagonal 100%, no off-diagonal overlaps). "
                "Please provide a proper matrix in data/default_usage_matrix.csv."
            )
            df_fallback = pd.DataFrame(0.0, index=keys, columns=keys)
            np.fill_diagonal(df_fallback.values, 1.0)
            return df_fallback

    df = _read_table_any_decimal(path)
    df.index = df.index.map(lambda s: str(s).strip())
    df.columns = df.columns.map(lambda s: str(s).strip())
    if set(df.index) != set(df.columns):
        raise ValueError("default_usage_matrix.csv must be square with identical row/column labels.")
    df = df.astype(float)
    maxv = float(np.nanmax(df.to_numpy()))
    if maxv > 1.5:  # probably entered as %
        df = df / 100.0
    return df.clip(lower=0, upper=1)

def get_usage_submatrix(chosen: List[str]) -> pd.DataFrame:
    base = load_default_usage_matrix()
    missing = [c for c in chosen if c not in base.index]
    if missing:
        for m in missing:
            base.loc[m, :] = 0.0
            base.loc[:, m] = 0.0
        base = base.sort_index().sort_index(axis=1)
    return base.loc[chosen, chosen].copy()

@st.cache_data(ttl=60)
def load_digital_pairs_matrix(path: Path = DIGITAL_PAIRS_MATRIX_CSV) -> pd.DataFrame:
    if not path.exists():
        alt_path = Path("/mnt/data") / path.name
        if not alt_path.exists():
            raise FileNotFoundError(f"{path} not found (also tried {alt_path}).")
        path = alt_path

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df.index.name is None and df.shape[1] > 0:
            if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                df = df.set_index(df.columns[0])

        df = df.loc[:, [c for c in df.columns if not str(c).strip().lower().startswith("unnamed")]]

        df.index = [str(i).strip() for i in df.index]
        df.columns = [str(c).strip() for c in df.columns]

        df = df.applymap(
            lambda x: pd.to_numeric(
                str(x).replace("\u00a0", " ").replace("%", "").strip().replace(",", "."),
                errors="coerce",
            )
        )

        with np.errstate(all="ignore"):
            maxv = float(np.nanmax(df.to_numpy())) if df.size else 0.0
        if maxv <= 1.5:
            df = df * 100.0

        return df.clip(lower=0, upper=100)

    try:
        df = pd.read_csv(
            path,
            engine="python",
            sep=None,
            decimal=".",
            encoding="utf-8-sig",
        )
        df = _clean(df)
        if len(df) >= 2 and df.shape[1] >= 2:
            return df
    except Exception:
        pass

    try:
        df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig", index_col=0)
        return _clean(df)
    except Exception:
        pass

    df = pd.read_csv(path, sep=",", decimal=".", encoding="utf-8-sig")
    return _clean(df)


# =====================================================================
# Mode 1
# =====================================================================
if mode == MODE_LABELS[0]:
    st.subheader("Independence: Sainsbury formula")
    st.caption("Cross reach = 1 − ∏(1 − Rᵢ). Assumes channels are independent.")

    rows = st.sidebar.slider("Rows (channels)", 3, 30, 5, key="rows_ind")
    seed = [
        {"Channel": "TV", "Reach %": 65.0},
        {"Channel": "Social media", "Reach %": 35.0},
        {"Channel": "Magazines", "Reach %": 15.0},
    ]
    if len(seed) < rows:
        seed += [{"Channel": "", "Reach %": 0.0} for _ in range(rows - len(seed))]
    df = pd.DataFrame(seed[:rows]).astype({"Reach %": float})

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Channel": st.column_config.TextColumn("Channel", width="medium"),
            "Reach %": st.column_config.NumberColumn("Reach %", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
        },
        key="table_ind",
    )

    channels = edited["Channel"].fillna("").astype(str).tolist()
    r_input = (edited["Reach %"].fillna(0.0).astype(float) / 100.0).clip(0, 1).tolist()

    cross = 1 - np.prod([1 - x for x in r_input])
    st.metric("Overall Cross-Media Reach", f"{cross:.1%}")

    chart_df = pd.DataFrame({"Channel": channels, "Media reach": r_input})
    chart_df = chart_df[chart_df["Channel"].str.strip() != ""]
    st.altair_chart(bar_chart(chart_df, "Channel", "Media reach", height=320), use_container_width=True)

    with st.expander("Details"):
        details = edited.copy()
        details["Reach (0–1)"] = r_input
        details["Reach %"] = [x * 100 for x in r_input]
        st.dataframe(details, use_container_width=True)

# =====================================================================
# Mode 2
# =====================================================================
elif mode == MODE_LABELS[1]:
    st.subheader("Overlap-aware: Sainsbury + monthly usage matrix")
    st.write(
        "Enter **media reach** (%) for each selected channel. "
        "Results are shown for **Regular** and **Attentive** (after applying attention indexes). "
        "The **monthly usage** matrix U(A) and U(A∩B) is editable at the end."
    )

    catalog = list(ADJ.keys())
    chosen = st.multiselect("Channels to include", catalog, default=[c for c in ["TV", "Social media", "Radio"] if c in catalog])
    if len(chosen) < 2:
        st.info("Select at least two channels to calculate.")
        st.stop()

    try:
        U0_units = get_usage_submatrix(chosen)  # 0..1
    except Exception as e:
        st.error(f"Failed to load usage matrix: {e}")
        st.stop()

    # Build reach editor (let the editor manage its own state by key)
    marg_df = pd.DataFrame({"Channel": chosen, "Reach %": [0.0] * len(chosen)}).astype({"Reach %": float})
    editor_key = "media_reach_editor__" + "|".join(chosen)
    marg_edit = st.data_editor(
        marg_df,
        column_config={
            "Channel": st.column_config.TextColumn(disabled=True),
            "Reach %": st.column_config.NumberColumn(
                "Reach %",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.1f",
            ),
        },
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
        key=editor_key,
    )

    # Robustly read reach from editor output (guard NaNs)
    vals_series = pd.to_numeric(marg_edit["Reach %"], errors="coerce")
    if vals_series.isna().any():
        gray_notice('Fill all <strong>“Reach %”</strong> cells to calculate results.')
        st.stop()
    vals = vals_series.astype(float).tolist()
    R_raw = {ch: pct_to_unit(v) for ch, v in zip(chosen, vals)}
    if any(R_raw[ch] is None for ch in chosen):
        gray_notice('Enter a <strong>media reach (%)</strong> for each selected channel to calculate results.')
        st.stop()
    if any(not (0 <= R_raw[ch] <= 1) for ch in chosen):
        st.error("Media reach values must be between 0 and 100%.")
        st.stop()

    # Editable % matrix in session state
    key_mat = "usage_matrix_df"
    if (key_mat not in st.session_state
        or list(st.session_state[key_mat].index) != chosen
        or list(st.session_state[key_mat].columns) != chosen):
        st.session_state[key_mat] = (U0_units * 100.0).round(1)

    U_df_pct = st.session_state[key_mat].copy()
    U = to_unit_df_from_pct(U_df_pct)

    # Symmetrize and enforce constraints
    for i, a in enumerate(chosen):
        for j, b in enumerate(chosen):
            if j <= i:
                continue
            ab = U.loc[a, b]
            ba = U.loc[b, a]
            both = [x for x in (ab, ba) if pd.notna(x)]
            v = float(np.mean(both)) if both else 0.0
            U.loc[a, b] = U.loc[b, a] = v

    for a in chosen:
        ua = float(U.loc[a, a])
        if not (0.0 <= ua <= 1.0):
            st.error(f"Monthly usage U({a}) must be between 0% and 100%. Got {ua*100:.1f}%.")
            st.stop()

    for a in chosen:
        for b in chosen:
            if a == b:
                continue
            U.loc[a, b] = float(min(U.loc[a, b], U.loc[a, a], U.loc[b, b]))

    # Clip R to U(a,a) and compute
    clipped = []
    for a in chosen:
        ua = float(U.loc[a, a])
        if R_raw[a] - ua > 1e-9:
            clipped.append((a, R_raw[a], ua))
    R_regular = {a: min(R_raw[a], float(U.loc[a, a])) for a in chosen}
    R_attentive = {ch: min(R_regular[ch] * ADJ.get(ch, 1.0), float(U.loc[ch, ch])) for ch in chosen}

    if clipped:
        msg = "Some inputs exceeded monthly usage and were clipped:<br>" + "<br>".join(
            f"• <strong>{a}</strong>: {r*100:.1f}% → {u*100:.1f}%" for a, r, u in clipped
        )
        gray_notice(msg)
        try:
            st.toast("One or more reaches were clipped to monthly usage.", icon="⚠️")
        except Exception:
            pass

    chans = chosen[:]
    est_reg, lb_reg, ub_reg, P2_reg = compute_union(chans, R_regular, U)
    est_att, lb_att, ub_att, P2_att = compute_union(chans, R_attentive, U)

    st.markdown("**Cross-media reach (Regular)**")
    reg_left, reg_right = st.columns([1, 3])
    with reg_left:
        st.metric("Reach", f"{est_reg:.1%}")
        st.caption(f"Bounds: LB≈{lb_reg:.1%}, UB≈{ub_reg:.1%}")
    with reg_right:
        reach_df_reg = pd.DataFrame({"Channel": chans, "Media reach": [R_regular[c] for c in chans]})
        st.altair_chart(bar_chart(reach_df_reg, "Channel", "Media reach", height=280, color="#000050"),
                        use_container_width=True)

    st.divider()

    st.markdown("**Cross-media reach (Attentive)**")
    att_left, att_right = st.columns([1, 3])
    with att_left:
        st.metric("Reach", f"{est_att:.1%}")
        st.caption(f"Bounds: LB≈{lb_att:.1%}, UB≈{ub_att:.1%}")
    with att_right:  # fixed (use att_right, not reg_right)
        reach_df_att = pd.DataFrame({"Channel": chans, "Media reach": [R_attentive[c] for c in chans]})
        st.altair_chart(bar_chart(reach_df_att, "Channel", "Media reach", height=280, color="#FEC8FF"),
                        use_container_width=True)

    with st.expander("Math & inputs ▸ Monthly usage matrix U (edit if needed)"):
        st.markdown("Diagonal = U(A), off-diagonals = U(A∩B). Values are % of population.")
        edited = st.data_editor(
            U_df_pct,
            use_container_width=True,
            num_rows="fixed",
            key="usage_matrix_editor_bottom",
        )
        st.session_state[key_mat] = edited

        diag = pd.DataFrame({
            "Channel": chans,
            "Adjustment": [ADJ.get(c, 1.0) for c in chans],
            "Reach % (input)": vals,
            "Reach % (Regular)": [R_regular[c] * 100 for c in chans],
            "Reach % (Attentive)": [R_attentive[c] * 100 for c in chans],
            "U(A) % (monthly users)": [float(U.loc[c, c]) * 100 for c in chans],
        })
        st.markdown("**Diagnostics (per channel)**")
        st.dataframe(diag, use_container_width=True, hide_index=True)

        st.markdown("**Derived effective overlap P(A∩B) used for the union (%, after conversion & clipping)**")
        tabs = st.tabs(["Regular", "Attentive"])
        with tabs[0]:
            st.dataframe(P2_reg.applymap(lambda v: None if v is None else round(float(v) * 100, 2)))
        with tabs[1]:
            st.dataframe(P2_att.applymap(lambda v: None if v is None else round(float(v) * 100, 2)))

# =====================================================================
# Mode 3: Digital pairs matrix (Attention idx only in Math part)
# =====================================================================
elif mode == MODE_LABELS[2]:
    st.subheader("Overlap-aware: Digital pairs matrix")
    st.write(
        "Pick digital channels below and enter **media reach** (%) for each. "
        "We compute **Regular** (as entered) and **Attentive** (after applying an **Attention idx**). "
        "Defaults come from the category mapping; edit Attention only in the Math section below."
    )

    try:
        digital_pct = load_digital_pairs_matrix()  # % 0..100
    except FileNotFoundError as e:
        st.error(f"{DIGITAL_PAIRS_MATRIX_CSV} not found. {e}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load digital pairs matrix: {e}")
        st.stop()

    digital_catalog = list(digital_pct.index)
    default_selection = [c for c in ["Facebook", "Instagram", "Youtube", "delfi.lt"] if c in digital_catalog]
    chosen = st.multiselect("Digital channels to include", digital_catalog, default=default_selection)
    if len(chosen) < 2:
        gray_notice("Select at least <strong>two</strong> digital channels.")
        st.stop()

    # state only for attention overrides; reach comes directly from editor
    if "attention_overrides" not in st.session_state:
        st.session_state["attention_overrides"] = {}

    # reach editor (no mirroring)
    marg_df = pd.DataFrame({
        "Channel": chosen,
        "Category": [digital_category_for(ch) or "—" for ch in chosen],
        "Reach %": [0.0] * len(chosen),
    }).astype({"Reach %": float})

    editor_key = "media_reach_editor_digital__" + "|".join(chosen)
    marg_edit = st.data_editor(
        marg_df,
        column_config={
            "Channel": st.column_config.TextColumn(disabled=True),
            "Category": st.column_config.TextColumn(disabled=True),
            "Reach %": st.column_config.NumberColumn(
                "Reach %",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.1f",
            ),
        },
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
        key=editor_key,
    )

    # Robustly read reach values (guard NaNs)
    vals_series = pd.to_numeric(marg_edit["Reach %"], errors="coerce")
    if vals_series.isna().any():
        gray_notice('Fill all <strong>“Reach %”</strong> cells.')
        st.stop()
    vals = vals_series.astype(float).tolist()
    R_raw = {ch: pct_to_unit(v) for ch, v in zip(chosen, vals)}
    if any(R_raw[ch] is None for ch in chosen):
        gray_notice('Enter a <strong>media reach (%)</strong> for each selected channel.')
        st.stop()
    if any(not (0 <= R_raw[ch] <= 1) for ch in chosen):
        st.error("Media reach values must be between 0 and 100%.")
        st.stop()

    U_df_pct = digital_pct.loc[chosen, chosen].copy()
    U_df_pct_display = U_df_pct.round(1)
    U = to_unit_df_from_pct(U_df_pct).astype(float)

    # Validate diagonals
    for a in chosen:
        ua = float(U.loc[a, a])
        if not (0.0 <= ua <= 1.0):
            st.error(f"Monthly usage U({a}) must be between 0% and 100%. Got {ua*100:.1f}%.")
            st.stop()

    # Symmetrize off-diagonals
    for i, a in enumerate(chosen):
        for j, b in enumerate(chosen):
            if j <= i:
                continue
            mean_val = np.nanmean([U.loc[a, b], U.loc[b, a]])
            v = 0.0 if np.isnan(mean_val) else float(mean_val)
            U.loc[a, b] = U.loc[b, a] = v

    # Enforce feasibility
    for a in chosen:
        for b in chosen:
            if a == b:
                continue
            U.loc[a, b] = float(min(U.loc[a, b], U.loc[a, a], U.loc[b, b]))

    # Attention idx — defaults & overrides
    att_idx: Dict[str, float] = {}
    for ch in chosen:
        canon = canonicalize_channel(ch)
        att_idx[canon] = float(st.session_state["attention_overrides"].get(canon, attention_factor_for_channel(canon)))

    # Clip to usage and compute
    clipped = []
    for a in chosen:
        ua = float(U.loc[a, a])
        if R_raw[a] - ua > 1e-9:
            clipped.append((a, R_raw[a], ua))
    R_regular = {a: min(R_raw[a], float(U.loc[a, a])) for a in chosen}
    R_attentive = {ch: float(np.clip(R_regular[ch] * att_idx[canonicalize_channel(ch)], 0.0, float(U.loc[ch, ch]))) for ch in chosen}

    if clipped:
        msg = "Some inputs exceeded monthly usage and were clipped:<br>" + "<br>".join(
            f"• <strong>{a}</strong>: {r*100:.1f}% → {u*100:.1f}%" for a, r, u in clipped
        )
        gray_notice(msg)
        try:
            st.toast("One or more reaches were clipped to monthly usage.", icon="⚠️")
        except Exception:
            pass

    chans = chosen[:]
    est_reg, lb_reg, ub_reg, P2_reg = compute_union(chans, R_regular, U)
    est_att, lb_att, ub_att, P2_att = compute_union(chans, R_attentive, U)

    st.markdown("**Cross-media reach (Regular)**")
    reg_left, reg_right = st.columns([1, 3])
    with reg_left:
        st.metric("Reach", f"{est_reg:.1%}")
        st.caption(f"Bounds: LB≈{lb_reg:.1%}, UB≈{ub_reg:.1%}")
    with reg_right:
        reach_df_reg = pd.DataFrame({"Channel": chans, "Media reach": [R_regular[c] for c in chans]})
        st.altair_chart(bar_chart(reach_df_reg, "Channel", "Media reach", height=280, color="#000050"),
                        use_container_width=True)

    st.divider()

    st.markdown("**Cross-media reach (Attentive)**")
    att_left, att_right = st.columns([1, 3])
    with att_left:
        st.metric("Reach", f"{est_att:.1%}")
        st.caption(f"Bounds: LB≈{lb_att:.1%}, UB≈{ub_att:.1%}")
    with att_right:
        reach_df_att = pd.DataFrame({"Channel": chans, "Media reach": [R_attentive[c] for c in chans]})
        st.altair_chart(bar_chart(reach_df_att, "Channel", "Media reach", height=280, color="#FEC8FF"),
                        use_container_width=True)

    with st.expander("Math & inputs ▸ Digital U matrix (read-only source) & Attention idx"):
        st.dataframe(U_df_pct_display, use_container_width=True)

        # Editable Attention here only
        att_df = pd.DataFrame({
            "Channel": chans,
            "Category": [digital_category_for(c) or "—" for c in chans],
            "Attention idx": [att_idx[canonicalize_channel(c)] for c in chans],
        }).astype({"Attention idx": float})
        att_edit = st.data_editor(
            att_df,
            column_config={
                "Channel": st.column_config.TextColumn(disabled=True),
                "Category": st.column_config.TextColumn(disabled=True),
                "Attention idx": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
            },
            num_rows="fixed",
            hide_index=True,
            key="attention_editor_bottom",
            use_container_width=True,
        )
        # Persist edits so next rerun uses them
        for i, ch in enumerate(chans):
            new_val = att_edit.loc[i, "Attention idx"]
            canon = canonicalize_channel(ch)
            if not pd.isna(new_val):
                st.session_state["attention_overrides"][canon] = float(new_val)

        diag = pd.DataFrame({
            "Channel": chans,
            "Category": [digital_category_for(c) or "—" for c in chans],
            "Attention idx (used)": [st.session_state["attention_overrides"].get(canonicalize_channel(c), attention_factor_for_channel(c)) for c in chans],
            "Reach % (Regular)": [R_regular[c] * 100 for c in chans],
            "Reach % (Attentive)": [R_attentive[c] * 100 for c in chans],
            "U(A) % (monthly users)": [float(U.loc[c, c]) * 100 for c in chans],
        })
        st.markdown("**Diagnostics (per channel)**")
        st.dataframe(diag, use_container_width=True, hide_index=True)

        st.markdown("**Derived effective overlap P(A∩B) used for the union (%, after conversion & clipping)**")
        tabs = st.tabs(["Regular", "Attentive"])
        with tabs[0]:
            st.dataframe(P2_reg.applymap(lambda v: None if pd.isna(v) else round(float(v) * 100, 2)))
        with tabs[1]:
            st.dataframe(P2_att.applymap(lambda v: None if pd.isna(v) else round(float(v) * 100, 2)))
