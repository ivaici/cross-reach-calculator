import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from base64 import b64encode  # ← ensures logo always loads

# -------------------- Page setup --------------------
# Make sure `mindshare_lithuania_logo.jpg` is present in the same directory as this script
st.set_page_config(page_title="Cross-Reach Calculator", page_icon="mindshare_lithuania_logo.jpg", layout="centered")

# --- Header with logo aligned to title (base64 so it always loads) ---
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

# Small helper to render a gray notice box (instead of st.info)
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
    "Overlap-aware (Digital pairs matrix)",  # NEW
]
mode = st.radio("Choose a mode", MODE_LABELS)

# -------------------- helpers --------------------
def pct_to_unit(x):
    """Accepts '0.84', '0,84', 84 -> returns 0.84 (proportion 0..1)."""
    if x is None or x == "":
        return None
    try:
        x = float(str(x).replace(",", "."))
    except Exception:
        return None
    return x / 100.0 if x > 1 else x

def unit_to_pct(x):
    return None if x is None else x * 100.0

def apply_attention(channel, value01):
    """Multiply reach (0..1) by the channel's adjustment index (default 1.0)."""
    factor = ADJ.get(channel, 1.0)
    return float(np.clip(value01 * factor, 0.0, 1.0))

def bar_chart(df, x_field, y_field, height=320, color="#9A3EFF"):
    """Unified bar chart with configurable color and auto height."""
    auto_height = max(height, 36 * max(1, len(df)))  # grow for long lists
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

def compute_union(chans, R_dict, U_matrix):
    r = {a: (0.0 if U_matrix.loc[a, a] in (None, 0) else min(1.0, R_dict[a] / U_matrix.loc[a, a])) for a in chans}
    P2 = pd.DataFrame(index=chans, columns=chans, dtype=float)
    for a in chans:
        P2.loc[a, a] = R_dict[a]
    for i, a in enumerate(chans):
        for j, b in enumerate(chans):
            if j <= i:
                continue
            uab = U_matrix.loc[a, b]
            pab = float(uab * r[a] * r[b])
            P2.loc[a, b] = P2.loc[b, a] = min(pab, R_dict[a], R_dict[b])
    sum_R = sum(R_dict[c] for c in chans)
    sum_pairs = sum(P2.loc[chans[i], chans[j]] for i in range(len(chans)) for j in range(i + 1, len(chans)))
    lower_pair = max(max(R_dict[c] for c in chans), sum_R - sum_pairs)
    upper_simple = min(1.0, sum_R)
    def est_triple(a, b, c):
        R1, R2, R3 = R_dict[a], R_dict[b], R_dict[c]
        P_ab, P_ac, P_bc = P2.loc[a, b], P2.loc[a, c], P2.loc[b, c]
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

# -------------------- Digital categories & matrix loader (for Mode 3) --------------------
DIGITAL_CATEGORY = {
    "BNS": "News portals", "ELTA": "News portals", "15min.lt": "News portals", "delfi.lt": "News portals",
    "diena.lt": "News portals", "lnk.lt": "News portals", "lrytas.lt": "News portals", "lrt.lt": "News portals",
    "tv3.lt": "News portals", "vz.lt": "News portals", "zmones.lt": "News portals",
    "gmail.com": "Other sites", "google.lt": "Search",
    "BeReal": "Social media", "Discord": "Social media", "Facebook": "Social media", "Instagram": "Social media",
    "Linkedin": "Social media", "Pinterest": "Social media", "Reddit": "Social media", "Snapchat": "Social media",
    "Telegram": "Social media", "Threads": "Social media", "Tik ToK": "Social media", "Tinder": "Social media",
    "VK": "Social media", "X": "Social media",
    "BasketNews": "Podcasts", "Bazaras? Bazaras!": "Podcasts", "Čia tik tarp mūsų": "Podcasts",
    "Kitokie pasikalbėjimai": "Podcasts", "Klajumo kanalas": "Podcasts", "Nepatogūs klausimai": "Podcasts",
    "Nesiaukite": "Podcasts", "Penktas kėlinys": "Podcasts", "PIN (Pradėk iš naujo)": "Podcasts",
    "Pralaužk vieną šaltą": "Podcasts", "Proto Industrija": "Podcasts", "Proto pemza": "Podcasts",
    "Savaitės rifas": "Podcasts", "Tapk geresniu": "Podcasts", "Vėl tie patys": "Podcasts",
    "15min Klausyk": "VOD", "3Play / TV3 Play": "VOD", "Amazon Prime": "VOD", "Apple TV": "VOD",
    "Delfi TV": "VOD", "Disney+": "VOD", "Go3": "VOD", "HBO": "VOD", "Youtube": "VOD", "Laisvės TV": "VOD",
    "lnk.lt / LNK Go": "VOD", "Lrytas TV": "VOD", "LRT Epika": "VOD", "LRT Mediateka": "VOD",
    "Netflix": "VOD", "Spotify": "VOD", "Telia Play": "VOD", "Twitch": "VOD", "Žmonės Cinema": "VOD",
}

def attention_factor_for_channel(ch: str) -> float:
    cat = DIGITAL_CATEGORY.get(ch)
    return ADJ.get(cat, 1.0)

@st.cache_data
def load_digital_pairs_matrix(path: str = "digital_pairs_matrix.csv") -> pd.DataFrame:
    import io, re
    txt = open(path, "r", encoding="utf-8").read()
    txt = re.sub(r"\s*;\s*", ",", txt)
    txt = re.sub(r"\s*,\s*", ",", txt)
    txt = re.sub(r"(?<=\d),(?=\d)", ".", txt)
    df = pd.read_csv(io.StringIO(txt), index_col=0)
    df.columns = [str(c).strip() for c in df.columns]
    df.index = [str(i).strip() for i in df.index]
    df = df.apply(pd.to_numeric, errors="coerce").clip(lower=0, upper=100)
    return df

# =====================================================================
# Mode 1: Independence (Sainsbury) — no attention option
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
        seed += [{"Channel": "", "Reach %": None} for _ in range(rows - len(seed))]
    df = pd.DataFrame(seed[:rows])

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
    r_input = (edited["Reach %"].fillna(0) / 100.0).clip(0, 1).tolist()

    cross = 1 - np.prod([1 - x for x in r_input])
    st.metric("Overall Cross-Media Reach", f"{cross:.1%}")

    chart_df = pd.DataFrame({"Channel": channels, "Media reach": r_input})
    st.altair_chart(bar_chart(chart_df, "Channel", "Media reach", height=320), use_container_width=True)

    with st.expander("Details"):
        details = edited.copy()
        details["Reach (0–1)"] = r_input
        details["Reach %"] = [x * 100 for x in r_input]
        st.dataframe(details, use_container_width=True)

# =====================================================================
# Mode 2: Overlap-aware (Sainsbury + monthly usage) — show Regular & Attentive
# =====================================================================
elif mode == MODE_LABELS[1]:
    st.subheader("Overlap-aware: Sainsbury + monthly usage matrix")
    st.write(
        "Enter **media reach** (%) for each selected channel. "
        "Results will be calculated two ways: **Regular** (as entered) and **Attentive** (after applying attention indexes). "
        "The **monthly usage** matrix U(A) and U(A∩B) is editable at the end."
    )

    # 14-channel catalog (matches your matrix)
    catalog = [
        "Cinema","Direct mail","Influencers","Magazines","Newspapers","News portals",
        "OOH","Podcasts","POS (Instore)","Radio","Search","Social media","TV","VOD",
    ]
    chosen = st.multiselect("Channels to include", catalog, default=["TV", "Social media", "Radio"])
    if len(chosen) < 2:
        st.info("Select at least two channels to calculate.")
        st.stop()

    # --- Default MONTHLY USAGE matrix (proportions 0..1) ---
    matrix_rows = [
        [0.25, 0.14, 0.14, 0.14, 0.14, 0.20, 0.22, 0.14, 0.19, 0.22, 0.22, 0.22, 0.23, 0.20],  # Cinema
        [0.16, 0.37, 0.24, 0.21, 0.23, 0.32, 0.38, 0.21, 0.36, 0.36, 0.35, 0.35, 0.38, 0.33],  # Direct mail
        [0.16, 0.24, 0.47, 0.20, 0.21, 0.42, 0.42, 0.27, 0.38, 0.43, 0.45, 0.47, 0.47, 0.44],  # Influencers
        [0.14, 0.18, 0.18, 0.34, 0.26, 0.27, 0.28, 0.17, 0.27, 0.29, 0.29, 0.29, 0.31, 0.29],  # Magazines
        [0.14, 0.20, 0.18, 0.26, 0.36, 0.29, 0.30, 0.17, 0.28, 0.31, 0.31, 0.30, 0.33, 0.29],  # Newspapers
        [0.22, 0.32, 0.42, 0.31, 0.33, 0.73, 0.62, 0.34, 0.57, 0.66, 0.71, 0.70, 0.71, 0.67],  # News portals
        [0.22, 0.33, 0.37, 0.28, 0.30, 0.55, 0.72, 0.31, 0.62, 0.61, 0.63, 0.61, 0.65, 0.59],  # OOH
        [0.16, 0.21, 0.27, 0.20, 0.19, 0.34, 0.36, 0.38, 0.32, 0.35, 0.37, 0.38, 0.37, 0.38],  # Podcasts
        [0.19, 0.32, 0.34, 0.27, 0.28, 0.50, 0.62, 0.29, 0.66, 0.55, 0.57, 0.56, 0.58, 0.54],  # POS (Instore)
        [0.22, 0.32, 0.38, 0.29, 0.31, 0.58, 0.61, 0.31, 0.55, 0.77, 0.67, 0.65, 0.70, 0.63],  # Radio
        [0.24, 0.35, 0.45, 0.33, 0.35, 0.71, 0.71, 0.37, 0.64, 0.75, 0.84, 0.81, 0.83, 0.77],  # Search
        [0.24, 0.35, 0.47, 0.33, 0.34, 0.70, 0.69, 0.38, 0.63, 0.74, 0.81, 0.84, 0.81, 0.78],  # Social media
        [0.23, 0.33, 0.41, 0.31, 0.33, 0.63, 0.65, 0.32, 0.58, 0.70, 0.74, 0.72, 0.85, 0.68],  # TV
        [0.23, 0.33, 0.44, 0.32, 0.33, 0.67, 0.67, 0.38, 0.61, 0.71, 0.77, 0.78, 0.77, 0.80],  # VOD
    ]
    default_usage_df = pd.DataFrame(matrix_rows, index=catalog, columns=catalog)

    # -------------------- Media reach inputs --------------------
    if "reach_values" not in st.session_state:
        st.session_state["reach_values"] = {}  # channel -> reach in percent (float or None)

    marg_df = pd.DataFrame({
        "Channel": chosen,
        "Reach %": [st.session_state["reach_values"].get(ch) for ch in chosen],
    })

    # key depends on current selection (forces remount when channels change)
    editor_key = "media_reach_editor__" + "|".join(chosen)

    marg_edit = st.data_editor(
        marg_df,
        column_config={
            "Channel": st.column_config.TextColumn(disabled=True),
            "Reach %": st.column_config.NumberColumn("Reach %", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
        },
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
        key=editor_key,
    )

    for i, ch in enumerate(chosen):
        st.session_state["reach_values"][ch] = marg_edit.loc[i, "Reach %"]

    R_raw = {ch: pct_to_unit(marg_edit.loc[i, "Reach %"]) for i, ch in enumerate(chosen)}
    if any(R_raw[ch] is None for ch in chosen):
        gray_notice('Enter a <strong>media reach (%)</strong> for each selected channel to calculate results.')
        st.stop()
    if any(not (0 <= R_raw[ch] <= 1) for ch in chosen):
        st.error("Media reach values must be between 0 and 100%.")
        st.stop()

    # --- Monthly usage matrix (editable, % values shown) ---
    key_mat = "usage_matrix_df"
    if (
        key_mat not in st.session_state
        or list(st.session_state[key_mat].index) != chosen
        or list(st.session_state[key_mat].columns) != chosen
    ):
        st.session_state[key_mat] = (default_usage_df.loc[chosen, chosen] * 100.0).round(1)

    U_df_pct = st.session_state[key_mat].copy()
    U = U_df_pct.copy()
    for a in chosen:
        for b in chosen:
            U.loc[a, b] = pct_to_unit(U.loc[a, b])

    # Symmetrize off-diagonals
    for i, a in enumerate(chosen):
        for j, b in enumerate(chosen):
            if j <= i:
                continue
            ab, ba = U.loc[a, b], U.loc[b, a]
            both = [x for x in (ab, ba) if x is not None]
            v = 0.5 * (ab + ba) if len(both) == 2 else (both[0] if len(both) == 1 else None)
            U.loc[a, b] = U.loc[b, a] = v

    # --- Validation (applies to Regular; Attentive will be ≤ Regular) ---
    for a in chosen:
        ua = U.loc[a, a]
        if ua is None or not (0 <= ua <= 1):
            st.error("Monthly usage U(A) (diagonal) must be 0–100%.")
            st.stop()
        if R_raw[a] - ua > 1e-9:
            st.error(f"Regular reach for {a} ({R_raw[a]:.2%}) cannot exceed monthly usage U({a}) ({ua:.2%}).")
            st.stop()
    for i, a in enumerate(chosen):
        for j, b in enumerate(chosen):
            if j <= i:
                continue
            uab = U.loc[a, b]
            if uab is None:
                st.error(f"Please fill monthly usage overlap U({a}∩{b}) in the matrix below.")
                st.stop()
            if uab < 0 or uab > min(U.loc[a, a], U.loc[b, b]) + 1e-9:
                st.error(f"U({a}∩{b}) must be ≤ min(U({a}), U({b})).")
                st.stop()

    # --- Build both reach dictionaries ---
    R_regular = R_raw
    R_attentive = {ch: apply_attention(ch, R_raw[ch]) for ch in chosen}

    # --- Compute both scenarios ---
    chans = chosen[:]
    est_reg, lb_reg, ub_reg, P2_reg = compute_union(chans, R_regular, U)
    est_att, lb_att, ub_att, P2_att = compute_union(chans, R_attentive, U)

    # --- Regular block: title above, short metric label next to chart ---
    st.markdown("**Cross-media reach (Regular)**")
    reg_left, reg_right = st.columns([1, 3])  # tweak ratios if you want
    with reg_left:
        st.metric("Reach", f"{est_reg:.1%}")   # short label (no truncation)
        st.caption(f"Bounds: LB≈{lb_reg:.1%}, UB≈{ub_reg:.1%}")
    with reg_right:
        reach_df_reg = pd.DataFrame({"Channel": chans, "Media reach": [R_regular[c] for c in chans]})
        st.altair_chart(
            bar_chart(reach_df_reg, "Channel", "Media reach", height=280, color="#9A3EFF"),
            use_container_width=True,
        )

    st.divider()

    # --- Attentive block: title above, short metric label next to chart ---
    st.markdown("**Cross-media reach (Attentive)**")
    att_left, att_right = st.columns([1, 3])
    with att_left:
        st.metric("Reach", f"{est_att:.1%}")   # short label (no truncation)
        st.caption(f"Bounds: LB≈{lb_att:.1%}, UB≈{ub_att:.1%}")
    with att_right:
        reach_df_att = pd.DataFrame({"Channel": chans, "Media reach": [R_attentive[c] for c in chans]})
        st.altair_chart(
            bar_chart(reach_df_att, "Channel", "Media reach", height=280, color="#FEC8FF"),
            use_container_width=True,
        )

    # --- Matrix editor (percent values) + diagnostics ---
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
            "Reach % (input)": [st.session_state["reach_values"].get(c) for c in chans],
            "Reach % (Regular)": [R_regular[c] * 100 for c in chans],
            "Reach % (Attentive)": [R_attentive[c] * 100 for c in chans],
            "U(A) % (monthly users)": [U.loc[c, c] * 100 for c in chans],
        })
        st.markdown("**Diagnostics (per channel)**")
        st.dataframe(diag, use_container_width=True, hide_index=True)

        st.markdown("**Derived effective overlap P(A∩B) used for the union (%, after conversion & clipping)**")
        tabs = st.tabs(["Regular", "Attentive"])
        with tabs[0]:
            st.dataframe(P2_reg.applymap(lambda v: None if v is None else round(v * 100, 2)))
        with tabs[1]:
            st.dataframe(P2_att.applymap(lambda v: None if v is None else round(v * 100, 2)))

# =====================================================================
# Mode 3: Overlap-aware (Digital pairs matrix) — category-based attention
# =====================================================================
elif mode == MODE_LABELS[2]:
    st.subheader("Overlap-aware: Digital pairs matrix")
    st.write(
        "Pick digital channels below and enter **media reach** (%) for each. "
        "We’ll compute cross-media reach two ways: **Regular** (as entered) and **Attentive** (after applying "
        "**category** attention factors). "
        "The matrix uses diagonal as **U(A)** and off-diagonals as **U(A∩B)**, all as % of population."
    )

    try:
        digital_pct = load_digital_pairs_matrix("digital_pairs_matrix.csv")
    except FileNotFoundError:
        st.error("`digital_pairs_matrix.csv` not found. Place it next to this script.")
        st.stop()

    digital_catalog = list(digital_pct.index)
    default_selection = [c for c in ["Facebook", "Instagram", "Youtube", "delfi.lt"] if c in digital_catalog]
    chosen = st.multiselect("Digital channels to include", digital_catalog, default=default_selection)
    if len(chosen) < 2:
        gray_notice("Select at least <strong>two</strong> digital channels.")
        st.stop()

    if "reach_values_digital" not in st.session_state:
        st.session_state["reach_values_digital"] = {}

    marg_df = pd.DataFrame({
        "Channel": chosen,
        "Category": [DIGITAL_CATEGORY.get(ch, "—") for ch in chosen],
        "Attention idx": [attention_factor_for_channel(ch) for ch in chosen],
        "Reach %": [st.session_state["reach_values_digital"].get(ch) for ch in chosen],
    })

    editor_key = "media_reach_editor_digital__" + "|".join(chosen)
    marg_edit = st.data_editor(
        marg_df,
        column_config={
            "Channel": st.column_config.TextColumn(disabled=True),
            "Category": st.column_config.TextColumn(disabled=True),
            "Attention idx": st.column_config.NumberColumn(disabled=True, format="%.2f"),
            "Reach %": st.column_config.NumberColumn("Reach %", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
        },
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
        key=editor_key,
    )

    for i, ch in enumerate(chosen):
        st.session_state["reach_values_digital"][ch] = marg_edit.loc[i, "Reach %"]

    R_raw = {ch: pct_to_unit(marg_edit.loc[i, "Reach %"]) for i, ch in enumerate(chosen)}
    if any(R_raw[ch] is None for ch in chosen):
        gray_notice('Enter a <strong>media reach (%)</strong> for each selected channel.')
        st.stop()
    if any(not (0 <= R_raw[ch] <= 1) for ch in chosen):
        st.error("Media reach values must be between 0 and 100%.")
        st.stop()

    U_df_pct = digital_pct.loc[chosen, chosen].copy()
    U = (U_df_pct / 100.0).astype(float)

    for a in chosen:
        ua = U.loc[a, a]
        if not (0 <= ua <= 1):
            st.error(f"Monthly usage U({a}) must be 0–100%.")
            st.stop()
        if R_raw[a] - ua > 1e-9:
            st.error(f"Regular reach for {a} ({R_raw[a]:.2%}) cannot exceed monthly usage U({a}) ({ua:.2%}).")
            st.stop()

    for i, a in enumerate(chosen):
        for j, b in enumerate(chosen):
            if j <= i:
                continue
            v = float(np.nanmean([U.loc[a, b], U.loc[b, a]]))
            U.loc[a, b] = U.loc[b, a] = v

    R_regular = R_raw
    R_attentive = {ch: float(np.clip(R_raw[ch] * attention_factor_for_channel(ch), 0.0, U.loc[ch, ch])) for ch in chosen}

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
        st.altair_chart(bar_chart(reach_df_reg, "Channel", "Media reach", height=280, color="#9A3EFF"), use_container_width=True)

    st.divider()

    st.markdown("**Cross-media reach (Attentive)**")
    att_left, att_right = st.columns([1, 3])
    with att_left:
        st.metric("Reach", f"{est_att:.1%}")
        st.caption(f"Bounds: LB≈{lb_att:.1%}, UB≈{ub_att:.1%}")
    with att_right:
        reach_df_att = pd.DataFrame({"Channel": chans, "Media reach": [R_attentive[c] for c in chans]})
        st.altair_chart(bar_chart(reach_df_att, "Channel", "Media reach", height=280, color="#FEC8FF"), use_container_width=True)

    with st.expander("Math & inputs ▸ Digital U matrix (read-only source)"):
        st.dataframe(U_df_pct, use_container_width=True)
        diag = pd.DataFrame({
            "Channel": chans,
            "Category": [DIGITAL_CATEGORY.get(c, "—") for c in chans],
            "Attention idx": [attention_factor_for_channel(c) for c in chans],
            "Reach % (input)": [R_regular[c]*100 for c in chans],
            "Reach % (Attentive)": [R_attentive[c]*100 for c in chans],
            "U(A) % (monthly users)": [U.loc[c, c]*100 for c in chans],
        })
        st.markdown("**Diagnostics (per channel)**")
        st.dataframe(diag, use_container_width=True, hide_index=True)

        st.markdown("**Derived effective overlap P(A∩B) used for the union (%, after conversion & clipping)**")
        tabs = st.tabs(["Regular", "Attentive"])
        with tabs[0]:
            st.dataframe(P2_reg.applymap(lambda v: None if pd.isna(v) else round(v * 100, 2)))
        with tabs[1]:
            st.dataframe(P2_att.applymap(lambda v: None if pd.isna(v) else round(v * 100, 2)))
