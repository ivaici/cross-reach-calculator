import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Cross-Reach Calculator", page_icon="📈", layout="centered")
st.title("📈 Cross-Reach Calculator")

# -------------------- Attention adjustment indexes (internal) --------------------
ADJ = {
    "Cinema": 0.98,
    "Direct mail": 0.56,
    "Influencers": 0.20,
    "Magazines": 0.56,
    "Newspapers": 0.29,
    "News portals": 0.20,
    "OOH": 0.41,
    "Podcasts": 0.61,
    "POS (Instore)": 0.42,
    "Radio": 0.40,
    "Search": 0.61,
    "Social media": 0.20,
    "TV": 0.55,
    "VOD": 0.61,
}

# -------------------- Global reach basis & mode --------------------
reach_basis = st.radio(
    "Choose a reach basis",
    ["Regular reach", "Attention-adjusted reach"],
    horizontal=True,
)
use_attentive = (reach_basis == "Attention-adjusted reach")

MODE_LABELS = [
    "Independence (Sainsbury)",
    "Overlap-aware (monthly usage + media reach)",
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

# =====================================================================
# Mode 1: Independence (Sainsbury)
# =====================================================================
if mode == MODE_LABELS[0]:
    st.subheader("Independence: Sainsbury formula")
    st.caption("Cross reach = 1 − ∏(1 − Rᵢ). If 'Attention-adjusted reach' is selected, each Rᵢ is multiplied by its index internally.")

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
            "Reach %": st.column_config.NumberColumn(
                "Reach %", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"
            ),
        },
        key="table_ind",
    )

    channels = edited["Channel"].fillna("").astype(str).tolist()
    r_input = (edited["Reach %"].fillna(0) / 100.0).clip(0, 1).tolist()
    r_eff = [apply_attention(ch, r) if use_attentive else r for ch, r in zip(channels, r_input)]

    cross = 1 - np.prod([1 - x for x in r_eff])
    st.metric(
        f"Overall Cross-Media Reach ({'Attentive' if use_attentive else 'Regular'})",
        f"{cross:.1%}",
    )

    # Chart with y-axis title "Media reach"
    chart_df = pd.DataFrame({"Channel": channels, "Media reach": r_eff})
    st.altair_chart(
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Channel:N", sort=None, title="Channel"),
            y=alt.Y("Media reach:Q", axis=alt.Axis(format="%", title="Media reach")),
            tooltip=[alt.Tooltip("Channel:N"), alt.Tooltip("Media reach:Q", format=".1%")],
        )
        .properties(height=320),
        use_container_width=True,
    )

    with st.expander("Details"):
        details = edited.copy()
        details["Reach (0–1) used"] = r_eff
        details["Reach used %"] = [x * 100 for x in r_eff]
        st.dataframe(details, use_container_width=True)

    if use_attentive and any(ch not in ADJ for ch in channels if ch.strip()):
        st.warning("Some channel names are not in the internal adjustment table; factor 1.0 was used for those.")

# =====================================================================
# Mode 2: Overlap-aware (monthly usage + media reach)
# =====================================================================
else:
    st.subheader("Overlap-aware: monthly usage matrix + media reach")
    st.write(
        "Enter **media reach** (%) for each selected channel. "
        "If 'Attention-adjusted reach' is selected, reaches are multiplied internally by the channel's index. "
        "The **monthly usage** matrix U(A) and U(A∩B) is editable at the end."
    )

    # 14-channel catalog (matches your matrix)
    catalog = [
        "Cinema",
        "Direct mail",
        "Influencers",
        "Magazines",
        "Newspapers",
        "News portals",
        "OOH",
        "Podcasts",
        "POS (Instore)",
        "Radio",
        "Search",
        "Social media",
        "TV",
        "VOD",
    ]
    chosen = st.multiselect("Channels to include", catalog, default=["TV", "Social media", "Radio"])
    if len(chosen) < 2:
        st.info("Select at least two channels to calculate.")
        st.stop()

    # --- Default MONTHLY USAGE matrix (proportions 0..1) ---
    matrix_rows = [
        [0.25, 0.16, 0.16, 0.14, 0.14, 0.22, 0.22, 0.16, 0.19, 0.22, 0.24, 0.24, 0.23, 0.23],  # Cinema
        [0.16, 0.41, 0.27, 0.21, 0.23, 0.36, 0.38, 0.24, 0.36, 0.36, 0.39, 0.39, 0.38, 0.38],  # Direct mail
        [0.16, 0.27, 0.53, 0.20, 0.21, 0.47, 0.42, 0.30, 0.38, 0.43, 0.51, 0.53, 0.47, 0.50],  # Influencers
        [0.14, 0.21, 0.20, 0.34, 0.26, 0.31, 0.28, 0.20, 0.27, 0.29, 0.33, 0.33, 0.31, 0.32],  # Magazines
        [0.14, 0.23, 0.21, 0.26, 0.36, 0.33, 0.30, 0.19, 0.28, 0.31, 0.35, 0.34, 0.33, 0.33],  # Newspapers
        [0.22, 0.36, 0.47, 0.31, 0.33, 0.82, 0.62, 0.38, 0.57, 0.66, 0.80, 0.79, 0.71, 0.76],  # News portals
        [0.22, 0.38, 0.42, 0.28, 0.30, 0.62, 0.72, 0.36, 0.62, 0.61, 0.71, 0.69, 0.65, 0.67],  # OOH
        [0.16, 0.24, 0.30, 0.20, 0.19, 0.38, 0.36, 0.43, 0.32, 0.35, 0.41, 0.43, 0.37, 0.43],  # Podcasts
        [0.19, 0.36, 0.38, 0.27, 0.28, 0.57, 0.62, 0.32, 0.66, 0.55, 0.64, 0.63, 0.58, 0.61],  # POS (Instore)
        [0.22, 0.36, 0.43, 0.29, 0.31, 0.66, 0.61, 0.35, 0.55, 0.77, 0.75, 0.74, 0.70, 0.71],  # Radio
        [0.24, 0.39, 0.51, 0.33, 0.35, 0.80, 0.71, 0.41, 0.64, 0.75, 0.95, 0.91, 0.83, 0.87],  # Search
        [0.24, 0.39, 0.53, 0.33, 0.34, 0.79, 0.69, 0.43, 0.63, 0.74, 0.91, 0.94, 0.81, 0.88],  # Social media
        [0.23, 0.38, 0.47, 0.31, 0.33, 0.71, 0.65, 0.37, 0.58, 0.70, 0.83, 0.81, 0.85, 0.77],  # TV
        [0.23, 0.38, 0.50, 0.32, 0.33, 0.76, 0.67, 0.43, 0.61, 0.71, 0.87, 0.88, 0.77, 0.90],  # VOD
    ]
    default_usage_df = pd.DataFrame(matrix_rows, index=catalog, columns=catalog)

    # --- Media reach inputs ---
    st.write("**Media reach by channel** (% of population).")
    marg_df = pd.DataFrame({"Channel": chosen, "Reach %": [None] * len(chosen)})
    marg_edit = st.data_editor(
        marg_df,
        column_config={
            "Channel": st.column_config.TextColumn(disabled=True),
            "Reach %": st.column_config.NumberColumn(
                "Reach %", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"
            ),
        },
        hide_index=True,
        use_container_width=True,
        key="media_reach_editor",
    )
    R_raw = {ch: pct_to_unit(marg_edit.loc[i, "Reach %"]) for i, ch in enumerate(chosen)}
    if any(R_raw[ch] is None for ch in chosen):
        st.info("Enter a **media reach (%)** for each selected channel to calculate results.")
        st.stop()
    if any(not (0 <= R_raw[ch] <= 1) for ch in chosen):
        st.error("Media reach values must be between 0 and 100%.")
        st.stop()

    # Apply attention (internally) if selected
    R = {ch: apply_attention(ch, R_raw[ch]) for ch in chosen} if use_attentive else R_raw

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

    # Validate: R(A) <= U(A), and pairwise bounds
    for a in chosen:
        ua = U.loc[a, a]
        if ua is None or not (0 <= ua <= 1):
            st.error("Monthly usage U(A) (diagonal) must be 0–100%.")
            st.stop()
        if R[a] - ua > 1e-9:
            st.error(
                f"{'Attentive ' if use_attentive else ''}reach for {a} ({R[a]:.2%}) "
                f"cannot exceed monthly usage U({a}) ({ua:.2%})."
            )
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

    # Convert usage → effective campaign pairwise via within-user reach r_i = R(A)/U(A)
    r = {a: (0.0 if U.loc[a, a] in (None, 0) else min(1.0, R[a] / U.loc[a, a])) for a in chosen}

    P2 = pd.DataFrame(index=chosen, columns=chosen, dtype=float)  # effective P(A∩B) for union
    for a in chosen:
        P2.loc[a, a] = R[a]
    for i, a in enumerate(chosen):
        for j, b in enumerate(chosen):
            if j <= i:
                continue
            uab = U.loc[a, b]
            pab = float(uab * r[a] * r[b])  # ≈ campaign-level overlap
            P2.loc[a, b] = P2.loc[b, a] = min(pab, R[a], R[b])  # clip to feasibility

    # Bounds + simple triple estimator (Kirkwood)
    chans = chosen[:]
    sum_R = sum(R[c] for c in chans)
    sum_pairs = sum(P2.loc[chans[i], chans[j]] for i in range(len(chans)) for j in range(i + 1, len(chans)))

    lower_pair = max(max(R[c] for c in chans), sum_R - sum_pairs)  # Bonferroni lower bound
    upper_simple = min(1.0, sum_R)  # simple upper bound

    def est_triple(a, b, c):
        R1, R2, R3 = R[a], R[b], R[c]
        P_ab, P_ac, P_bc = P2.loc[a, b], P2.loc[a, c], P2.loc[b, c]
        denom = max(1e-12, R1 * R2 * R3)
        t = (P_ab * P_ac * P_bc) / denom  # Kirkwood approx
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

    # Outputs
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            f"Estimated cross-media reach ({'Attentive' if use_attentive else 'Regular'})",
            f"{est_union:.1%}",
        )
    with c2:
        st.caption(f"Bounds: LB≈{lower_pair:.1%}, UB≈{upper_simple:.1%}")

    # Per-channel chart with y-axis title "Media reach"
    reach_df = pd.DataFrame({"Channel": chans, "Media reach": [R[c] for c in chans]})
    st.altair_chart(
        alt.Chart(reach_df)
        .mark_bar()
        .encode(
            x=alt.X("Channel:N", sort=None),
            y=alt.Y("Media reach:Q", axis=alt.Axis(format="%", title="Media reach")),
            tooltip=[alt.Tooltip("Channel:N"), alt.Tooltip("Media reach:Q", format=".1%")],
        )
        .properties(height=280),
        use_container_width=True,
    )

    # Matrix editor (percent values)
    with st.expander("Math & inputs ▸ Monthly usage matrix U (edit if needed)"):
        st.markdown("Diagonal = U(A), off-diagonals = U(A∩B). Values are % of population.")
        edited = st.data_editor(
            U_df_pct,
            use_container_width=True,
            num_rows="fixed",
            key="usage_matrix_editor_bottom",
        )
        st.session_state[key_mat] = edited

        # Diagnostics table (includes adjustment; hides row numbers)
        diag = pd.DataFrame({
            "Channel": chans,
            "Adjustment": [ADJ.get(c, 1.0) for c in chans],
            "Reach % (input)": [R_raw[c] * 100 for c in chans],
            "Reach % (used)": [R[c] * 100 for c in chans],
            "U(A) % (monthly users)": [U.loc[c, c] * 100 for c in chans],
        })
        st.markdown("**Diagnostics (per channel)**")
        st.dataframe(diag, use_container_width=True, hide_index=True)

        st.markdown("**Derived effective overlap P(A∩B) used for the union (%, after conversion & clipping)**")
        st.dataframe(P2.applymap(lambda v: None if v is None else round(v * 100, 2)))
