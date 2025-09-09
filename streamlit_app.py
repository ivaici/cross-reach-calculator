import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Cross-Reach Calculator", page_icon="📈", layout="centered")
st.title("📈 Cross-Reach Calculator")
st.caption("Modes: (1) Independence (Sainsbury), (2) Overlap-aware (monthly usage + media reach)")

MODE_LABELS = [
    "Independence (Sainsbury)",
    "Overlap-aware (monthly usage + media reach)",
]
mode = st.radio("Choose a mode", MODE_LABELS)

# ---------- helpers ----------
def pct_to_unit(x):
    """Accepts '0.84', '0,84', 84 -> returns 0.84 (proportion 0..1)."""
    if x is None or x == "":
        return None
    try:
        x = float(str(x).replace(",", "."))
    except:
        return None
    return x/100.0 if x > 1 else x

def unit_to_pct(x):
    return None if x is None else x * 100.0

def incremental_series(r):
    """Sainsbury-style incremental by row order for vector r (proportions)."""
    inc, unreached = [], 1.0
    for ri in r:
        inc.append(unreached * ri)
        unreached *= (1 - ri)
    return inc

# ---------- Mode 1: Independence ----------
if mode == MODE_LABELS[0]:
    st.subheader("Independence: Sainsbury formula")
    st.write("Enter channel names and media reach % (0–100). Add/remove rows as needed.")

    rows = st.sidebar.slider("Rows (channels)", 3, 30, 8, key="rows_ind")
    seed = [{"Channel":"TV","Reach %":65.0},
            {"Channel":"Social media","Reach %":35.0},
            {"Channel":"Print","Reach %":15.0}]
    if len(seed) < rows:
        seed += [{"Channel":"", "Reach %":None} for _ in range(rows - len(seed))]
    df = pd.DataFrame(seed[:rows])

    edited = st.data_editor(
        df, num_rows="dynamic", hide_index=True, use_container_width=True,
        column_config={
            "Channel": st.column_config.TextColumn("Channel", width="medium"),
            "Reach %": st.column_config.NumberColumn("Reach %", min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
        },
        key="table_ind"
    )

    r = (edited["Reach %"].fillna(0)/100.0).clip(0, 1)
    cross = 1 - np.prod(1 - r)
    st.metric("Overall Cross-Media Reach", f"{cross:.1%}")

    inc = incremental_series(r)
    chart_df = pd.DataFrame({"Channel": edited["Channel"], "Incremental": inc})
    st.altair_chart(
        alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Channel:N", sort=None, title="Channel"),
            y=alt.Y("Incremental:Q", axis=alt.Axis(format="%"), title="Incremental Reach"),
            tooltip=[alt.Tooltip("Channel:N"), alt.Tooltip("Incremental:Q", format=".1%")],
        ).properties(height=320),
        use_container_width=True
    )

    with st.expander("Details"):
        details = edited.copy()
        details["Reach (0–1)"] = r
        details["1 − Reach"]   = 1 - r
        details["Incremental"] = inc
        st.dataframe(details, use_container_width=True)

    st.info("Assumes independence across channels. For overlaps, use the next mode.")

# ---------- Mode 2: Overlap-aware (monthly usage + media reach) ----------
else:
    st.subheader("Overlap-aware: monthly usage matrix + media reach")
    st.write("Enter **media reach** (%) for each channel. The **monthly usage** matrix U(A), U(A∩B) is editable at the end.")

    # Fixed catalog & user selection
    catalog = ["TV","Radio","OOH","Print","Cinema","VOD","Social media","Search","Other sites"]
    chosen = st.multiselect("Channels to include", catalog, default=["TV","Social media","Print"])
    if len(chosen) < 2:
        st.info("Select at least two channels to calculate.")
        st.stop()

    # Default MONTHLY USAGE matrix (diagonal = U(A), off-diagonal = U(A∩B)) in proportions (0..1)
    default_usage = {
        ("TV","TV"):0.84, ("TV","Radio"):0.55, ("TV","OOH"):0.55, ("TV","Print"):0.39, ("TV","Cinema"):0.55, ("TV","VOD"):0.55, ("TV","Social media"):0.81, ("TV","Search"):0.55, ("TV","Other sites"):0.55,
        ("Radio","TV"):0.55, ("Radio","Radio"):0.55, ("Radio","OOH"):0.55, ("Radio","Print"):0.55, ("Radio","Cinema"):0.55, ("Radio","VOD"):0.55, ("Radio","Social media"):0.55, ("Radio","Search"):0.55, ("Radio","Other sites"):0.55,
        ("OOH","TV"):0.55, ("OOH","Radio"):0.55, ("OOH","OOH"):0.55, ("OOH","Print"):0.55, ("OOH","Cinema"):0.55, ("OOH","VOD"):0.55, ("OOH","Social media"):0.55, ("OOH","Search"):0.55, ("OOH","Other sites"):0.55,
        ("Print","TV"):0.39, ("Print","Radio"):0.55, ("Print","OOH"):0.55, ("Print","Print"):0.433, ("Print","Cinema"):0.55, ("Print","VOD"):0.55, ("Print","Social media"):0.42, ("Print","Search"):0.55, ("Print","Other sites"):0.55,
        ("Cinema","TV"):0.55, ("Cinema","Radio"):0.55, ("Cinema","OOH"):0.55, ("Cinema","Print"):0.55, ("Cinema","Cinema"):0.55, ("Cinema","VOD"):0.55, ("Cinema","Social media"):0.55, ("Cinema","Search"):0.55, ("Cinema","Other sites"):0.55,
        ("VOD","TV"):0.55, ("VOD","Radio"):0.55, ("VOD","OOH"):0.55, ("VOD","Print"):0.55, ("VOD","Cinema"):0.55, ("VOD","VOD"):0.55, ("VOD","Social media"):0.55, ("VOD","Search"):0.55, ("VOD","Other sites"):0.55,
        ("Social media","TV"):0.81, ("Social media","Radio"):0.55, ("Social media","OOH"):0.55, ("Social media","Print"):0.42, ("Social media","Cinema"):0.55, ("Social media","VOD"):0.55, ("Social media","Social media"):0.943, ("Social media","Search"):0.55, ("Social media","Other sites"):0.55,
        ("Search","TV"):0.55, ("Search","Radio"):0.55, ("Search","OOH"):0.55, ("Search","Print"):0.55, ("Search","Cinema"):0.55, ("Search","VOD"):0.55, ("Search","Social media"):0.55, ("Search","Search"):0.55, ("Search","Other sites"):0.55,
        ("Other sites","TV"):0.55, ("Other sites","Radio"):0.55, ("Other sites","OOH"):0.55, ("Other sites","Print"):0.55, ("Other sites","Cinema"):0.55, ("Other sites","VOD"):0.55, ("Other sites","Social media"):0.55, ("Other sites","Search"):0.55, ("Other sites","Other sites"):0.55,
    }

    # Media reach inputs first
    st.write("**Media reach by channel** (% of population).")
    marg_df = pd.DataFrame({"Channel": chosen, "Reach %": [None]*len(chosen)})
    marg_edit = st.data_editor(
        marg_df,
        column_config={
            "Channel": st.column_config.TextColumn(disabled=True),
            "Reach %": st.column_config.NumberColumn("Reach %", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
        },
        hide_index=True,
        use_container_width=True,
        key="media_reach_editor",
    )
    R = {ch: pct_to_unit(marg_edit.loc[i, "Reach %"]) for i, ch in enumerate(chosen)}
    if any(R[ch] is None for ch in chosen):
        st.info("Enter a **media reach (%)** for each selected channel to calculate results.")
        st.stop()
    if any(not (0 <= R[ch] <= 1) for ch in chosen):
        st.error("Media reach values must be between 0 and 100%.")
        st.stop()

    # Monthly usage matrix kept in session state; editor shown at the end
    key_mat = "usage_matrix_df"
    if (key_mat not in st.session_state 
        or list(st.session_state[key_mat].index) != chosen 
        or list(st.session_state[key_mat].columns) != chosen):
        mat_init = pd.DataFrame(index=chosen, columns=chosen, dtype=float)
        for a in chosen:
            for b in chosen:
                mat_init.loc[a,b] = unit_to_pct(default_usage.get((a,b), np.nan))
        st.session_state[key_mat] = mat_init

    # Use current usage matrix values (percent), convert to proportions
    U_df_pct = st.session_state[key_mat].copy()
    U = U_df_pct.copy()
    for a in chosen:
        for b in chosen:
            U.loc[a,b] = pct_to_unit(U.loc[a,b])

    # Symmetrize off-diagonals (average U(A∩B) and U(B∩A))
    for i,a in enumerate(chosen):
        for j,b in enumerate(chosen):
            if j <= i: continue
            ab, ba = U.loc[a,b], U.loc[b,a]
            both = [x for x in [ab,ba] if x is not None]
            v = 0.5*(ab+ba) if len(both)==2 else (both[0] if len(both)==1 else None)
            U.loc[a,b] = U.loc[b,a] = v

    # Validate usage diagonals and pairs against media reach
    for a in chosen:
        ua = U.loc[a,a]
        if ua is None or not (0 <= ua <= 1):
            st.error("Monthly usage U(A) must be set (0–100%) on the diagonal of the matrix in the Math & inputs section.")
            st.stop()
        if R[a] - ua > 1e-9:
            st.error(f"Media reach for {a} ({R[a]:.2%}) cannot exceed monthly usage U({a}) ({ua:.2%}).")
            st.stop()
    for i,a in enumerate(chosen):
        for j,b in enumerate(chosen):
            if j <= i: continue
            uab = U.loc[a,b]
            if uab is None:
                st.error(f"Please fill monthly usage overlap U({a}∩{b}) in the matrix below.")
                st.stop()
            if uab < 0 or uab > min(U.loc[a,a], U.loc[b,b]) + 1e-9:
                st.error(f"U({a}∩{b}) must be ≤ min(U({a}), U({b})).")
                st.stop()

    # Convert usage → effective campaign pairwise via within-user reach r_i = R(A)/U(A)
    r = {a: (0.0 if U.loc[a,a] in (None,0) else min(1.0, R[a]/U.loc[a,a])) for a in chosen}
    P2 = pd.DataFrame(index=chosen, columns=chosen, dtype=float)  # effective P(A∩B) used for union
    for a in chosen:
        P2.loc[a,a] = R[a]
    for i,a in enumerate(chosen):
        for j,b in enumerate(chosen):
            if j <= i: continue
            # P(A∩B) ≈ U(A∩B)*rA*rB, clipped to feasibility
            uab = U.loc[a,b]
            pab = float(uab * r[a] * r[b])
            P2.loc[a,b] = P2.loc[b,a] = min(pab, R[a], R[b])

    # Bounds + heuristic triples (Kirkwood)
    chans = chosen[:]  # keep UI order
    sum_R = sum(R[c] for c in chans)
    sum_pairs = sum(P2.loc[chans[i], chans[j]] for i in range(len(chans)) for j in range(i+1, len(chans)))

    lower_pair = max(max(R[c] for c in chans), sum_R - sum_pairs)   # Bonferroni lower bound
    upper_simple = min(1.0, sum_R)                                  # simple upper bound

    def est_triple(a, b, c):
        R1, R2, R3 = R[a], R[b], R[c]
        P_ab, P_ac, P_bc = P2.loc[a,b], P2.loc[a,c], P2.loc[b,c]
        denom = max(1e-12, R1*R2*R3)
        t = (P_ab * P_ac * P_bc) / denom
        lower = max(0.0, (P_ab + P_ac + P_bc) - R1 - R2 - R3)
        upper = min(P_ab, P_ac, P_bc)
        return float(np.clip(t, lower, upper))

    triple_sum = 0.0
    if len(chans) >= 3:
        for i in range(len(chans)):
            for j in range(i+1, len(chans)):
                for k in range(j+1, len(chans)):
                    triple_sum += est_triple(chans[i], chans[j], chans[k])

    est_union = float(np.clip(sum_R - sum_pairs + triple_sum, lower_pair, upper_simple))

    # Outputs
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Estimated cross-media reach", f"{est_union:.1%}")
    with c2:
        st.caption(f"Bounds: LB≈{lower_pair:.1%}, UB≈{upper_simple:.1%}")

    # Per-channel media reach bars
    reach_df = pd.DataFrame({"Channel": chans, "Media reach": [R[c] for c in chans]})
    st.altair_chart(
        alt.Chart(reach_df).mark_bar().encode(
            x=alt.X("Channel:N", sort=None),
            y=alt.Y("Media reach:Q", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("Channel:N"), alt.Tooltip("Media reach:Q", format=".1%")],
        ).properties(height=280),
        use_container_width=True
    )

    # Incremental (approx): scale independence to match estimated union
    r_vec = np.array([R[c] for c in chans])
    indep_union = 1 - np.prod(1 - r_vec)
    scale = 1.0

    if len(chans) == 2:
        # Exact scaling for 2 channels: solve 1 - (1 - s a)(1 - s b) = target
        a, b = r_vec
        target = est_union
        Acoef = a*b
        Bcoef = -(a+b)
        Ccoef = target
        if Acoef > 0:
            disc = max(0.0, Bcoef*Bcoef - 4*Acoef*Ccoef)
            s_root = ( -Bcoef - np.sqrt(disc) ) / (2*Acoef)
            scale = float(np.clip(s_root, 0.0, 2.0))
        else:
            # If one channel is zero, fall back to linear
            denom = (a + b) if (a + b) > 1e-12 else 1.0
            scale = float(np.clip(target / denom, 0.0, 2.0))
    else:
        # Newton step with correct derivative sign; tight bounds
        if indep_union > 1e-9 and est_union > 0:
            for _ in range(8):
                prod_term = np.prod(1 - scale*r_vec)
                f = 1 - prod_term - est_union
                if abs(f) < 1e-10:
                    break
                # d/ds [1 - Π (1 - s r_i)] = Π(1 - s r_i) * Σ r_i / (1 - s r_i)
                g = prod_term * np.sum(r_vec / (1 - scale*r_vec))
                if abs(g) < 1e-12:
                    break
                scale = float(np.clip(scale - f/g, 0.0, 2.0))

    r_scaled = np.clip(scale * r_vec, 0, 1)
    inc = incremental_series(r_scaled)

    inc_df = pd.DataFrame({"Channel": chans, "Incremental (approx)": inc})
    st.altair_chart(
        alt.Chart(inc_df).mark_bar().encode(
            x=alt.X("Channel:N", sort=None),
            y=alt.Y("Incremental (approx):Q", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("Channel:N"), alt.Tooltip("Incremental (approx):Q", format=".1%")],
        ).properties(height=320),
        use_container_width=True
    )

    # Monthly usage matrix editor at the end (edits persist via session_state)
    with st.expander("Math & inputs ▸ Monthly usage matrix U (edit if needed)"):
        st.markdown("Diagonal = U(A), off-diagonals = U(A∩B). Values are % of population.")
        edited = st.data_editor(
            U_df_pct,
            use_container_width=True,
            num_rows="fixed",
            key="usage_matrix_editor_bottom",
        )
        st.session_state[key_mat] = edited  # used on next interaction
        st.markdown("**Derived effective overlap P(A∩B) used for the union (%, after conversion & clipping)**")
        st.dataframe(P2.applymap(lambda v: None if v is None else round(v*100, 2)))
