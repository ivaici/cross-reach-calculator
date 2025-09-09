import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Cross-Reach Calculator", page_icon="📈", layout="centered")
st.title("📈 Cross-Reach Calculator")
st.caption(
    "Modes: (1) Independence (Sainsbury), (2) Overlap-aware (2–3 channels, exact), "
    "(3) Usage-adjusted (3 channels), (4) Overlap-aware (n channels, pairwise approx)."
)

MODE_LABELS = [
    "Independence (any # of channels)",
    "Overlap-aware (2–3 channels)",
    "Usage-adjusted (3 channels)",
    "Overlap-aware (n channels, pairwise approx)"
]
mode = st.radio("Choose a mode", MODE_LABELS)

# ---------- helpers ----------
def pct_to_unit(x):
    if x is None or x == "":
        return None
    try:
        # also handle commas "0,84"
        x = float(str(x).replace(",", "."))
    except:
        return None
    return x/100.0 if x > 1 else x

def unit_to_pct(x):
    if x is None: return None
    return x * 100.0

def warn_if_prob_invalid(val, name):
    if val is None: return False
    if val < 0 or val > 1:
        st.error(f"{name} must be between 0 and 1 (or 0–100%). You entered {val:.3f}.")
        return True
    return False

def incremental_series(r):
    # r: array of per-channel reach (0..1) in row order
    inc, unreached = [], 1.0
    for ri in r:
        inc.append(unreached * ri)
        unreached *= (1 - ri)
    return inc

# ---------- Mode 1 ----------
if mode == MODE_LABELS[0]:
    st.subheader("Independence: Sainsbury formula")
    st.write("Enter channel names and reach % (0–100). Add/remove rows as needed.")

    rows = st.sidebar.slider("Rows (channels)", 3, 30, 8, key="rows_ind")
    seed = [{"Channel":"TV","Reach %":65.0},
            {"Channel":"Social","Reach %":35.0},
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

    st.info("Assumes independence across channels. For overlaps, use modes 2–4.")

# ---------- Mode 2 ----------
elif mode == MODE_LABELS[1]:
    st.subheader("Overlap-aware: Inclusion–Exclusion (exact for 3 channels)")
    st.write("Provide campaign reaches (population-level) and overlaps. Use % or decimals.")

    cols = st.columns(3)
    names, reaches = [], []
    defaults_n = ["TV", "Social", "Print"]
    defaults_r = [65, 35, 15]
    for i, c in enumerate(cols):
        with c:
            nm = st.text_input(f"Channel {i+1} name", value=defaults_n[i] if i < 3 else f"Ch{i+1}")
            rv = pct_to_unit(st.text_input(f"{nm} reach", value=defaults_r[i] if i < 3 else ""))
            names.append(nm); reaches.append(rv)

    keep = [i for i in range(3) if (names[i] and reaches[i] is not None)]
    if len(keep) < 2:
        st.warning("Enter at least two channels with reaches.")
        st.stop()
    for i in keep:
        if warn_if_prob_invalid(reaches[i], f"{names[i]} reach"): st.stop()

    # pairwise overlaps
    pair = {}
    for i in keep:
        for j in keep:
            if j <= i: continue
            val = pct_to_unit(st.text_input(f"{names[i]} ∩ {names[j]}", value=""))
            if val is not None and (val < 0 or val > 1):
                st.error(f"Overlap {names[i]}∩{names[j]} must be 0–1.")
                st.stop()
            pair[(min(i,j),max(i,j))] = val

    # optional triple
    triple = None
    if len(keep) == 3:
        i,j,k = keep
        triple = pct_to_unit(st.text_input(f"{names[i]} ∩ {names[j]} ∩ {names[k]} (optional)", value=""))
        if triple is not None and (triple < 0 or triple > 1):
            st.error("Triple overlap must be 0–1."); st.stop()

        if all(pair.get(t) is not None for t in [(min(i,j),max(i,j)), (min(i,k),max(i,k)), (min(j,k),max(j,k))]):
            a,b,c = reaches[i], reaches[j], reaches[k]
            ab = pair[(min(i,j),max(i,j))]; ac = pair[(min(i,k),max(i,k))]; bc = pair[(min(j,k),max(j,k))]
            lower = max(0.0, ab + ac + bc - a - b - c)
            upper = min(ab, ac, bc)
            if triple is not None and not (lower - 1e-9 <= triple <= upper + 1e-9):
                st.warning(f"Triple outside feasible bounds [{lower:.2%}, {upper:.2%}].")

    # compute union
    used = [names[i] for i in keep]
    st.write(f"Channels used: {', '.join(used)}")

    if len(keep) == 2:
        i,j = keep
        ab = pair.get((min(i,j),max(i,j)))
        if ab is None:
            st.info("Enter the pairwise overlap to compute A ∪ B."); st.stop()
        cross = reaches[i] + reaches[j] - ab
        st.metric("Cross-media reach (A ∪ B)", f"{cross:.1%}")
    else:
        i,j,k = keep
        ab = pair.get((min(i,j),max(i,j))); ac = pair.get((min(i,k),max(i,k))); bc = pair.get((min(j,k),max(j,k)))
        if ab is None or ac is None or bc is None:
            st.info("Enter all three pairwise overlaps. Triple is optional (will be estimated)."); st.stop()
        if triple is None:
            a,b,c = reaches[i], reaches[j], reaches[k]
            est = (ab * ac * bc) / max(1e-12, a*b*c)  # heuristic
            lower = max(0.0, ab + ac + bc - a - b - c); upper = min(ab, ac, bc)
            triple = float(np.clip(est, lower, upper))
            st.caption(f"Estimated triple via heuristic: {triple:.2%} (clipped to feasible bounds).")
        cross = reaches[i] + reaches[j] + reaches[k] - ab - ac - bc + triple
        st.metric("Cross-media reach (A ∪ B ∪ C)", f"{cross:.1%}")

# ---------- Mode 3 ----------
elif mode == MODE_LABELS[2]:
    st.subheader("Usage-adjusted: campaign reach + monthly usage overlaps (3 channels)")
    st.write("Enter campaign reaches and monthly usage overlaps from survey. We convert to within-user reach and weight by usage segments.")

    cols = st.columns(3)
    names, camp = [], []
    for i, c in enumerate(cols):
        with c:
            nm = st.text_input(f"Channel {i+1} name", value=["TV","Social","Print"][i])
            rv = pct_to_unit(st.text_input(f"{nm} campaign reach (pop-level)", value=[65,35,15][i]))
            names.append(nm); camp.append(rv)

    st.markdown("**Monthly usage marginals**")
    usage, ucols = [], st.columns(3)
    defaults = [84, 94.3, 43.3]
    for i, c in enumerate(ucols):
        with c:
            uv = pct_to_unit(st.text_input(f"P({names[i]})", value=defaults[i]))
            usage.append(uv)

    st.markdown("**Monthly usage pairwise overlaps**")
    pair = {}
    defaults_pair = {(0,1): 81.3, (0,2): 39.3, (1,2): 41.5}
    for (i,j), dv in defaults_pair.items():
        val = pct_to_unit(st.text_input(f"P({names[i]} ∩ {names[j]})", value=dv))
        pair[(i,j)] = val

    triple = pct_to_unit(st.text_input(f"P({names[0]} ∩ {names[1]} ∩ {names[2]})", value=37.5))

    # validations
    for i in range(3):
        if warn_if_prob_invalid(camp[i], f"{names[i]} campaign reach"): st.stop()
        if warn_if_prob_invalid(usage[i], f"P({names[i]})"): st.stop()
        if usage[i] is not None and camp[i] is not None and camp[i] - usage[i] > 1e-9:
            st.error(f"{names[i]} campaign reach ({camp[i]:.2%}) cannot exceed usage P({names[i]}) ({usage[i]:.2%})."); st.stop()
    for (i,j), v in pair.items():
        if warn_if_prob_invalid(v, f"P({names[i]}∩{names[j]})"): st.stop()
        if usage[i] is not None and usage[j] is not None and v is not None:
            if v - min(usage[i], usage[j]) > 1e-9:
                st.error(f"P({names[i]}∩{names[j]}) cannot exceed min(P({names[i]}), P({names[j]}))."); st.stop()
    if triple is not None:
        ab, ac, bc = pair[(0,1)], pair[(0,2)], pair[(1,2)]
        if None not in (ab, ac, bc):
            lower = max(0.0, ab + ac + bc - usage[0] - usage[1] - usage[2])
            upper = min(ab, ac, bc)
            if not (lower - 1e-9 <= triple <= upper + 1e-9):
                st.warning(f"P(ABC) outside feasible bounds [{lower:.2%}, {upper:.2%}].")

    # within-user campaign reach
    r_user = [0.0 if usage[i] in (None, 0) else min(1.0, camp[i] / usage[i]) for i in range(3)]

    # disjoint usage segments
    A,B,C = usage
    AB,AC,BC = pair[(0,1)], pair[(0,2)], pair[(1,2)]
    ABC = triple
    onlyA  = A - AB - AC + ABC
    onlyB  = B - AB - BC + ABC
    onlyC  = C - AC - BC + ABC
    onlyAB = AB - ABC
    onlyAC = AC - ABC
    onlyBC = BC - ABC
    all3   = ABC
    none   = 1 - (onlyA + onlyB + onlyC + onlyAB + onlyAC + onlyBC + all3)

    segs = pd.DataFrame({
        "Segment": [
            f"{names[0]} only", f"{names[1]} only", f"{names[2]} only",
            f"{names[0]} & {names[1]} only", f"{names[0]} & {names[2]} only", f"{names[1]} & {names[2]} only",
            "All three", "None of these"
        ],
        "Share": [onlyA, onlyB, onlyC, onlyAB, onlyAC, onlyBC, all3, none]
    })
    if (segs["Share"] < -1e-6).any():
        st.error("Usage overlaps are inconsistent (some segment shares are negative). Please adjust inputs.")
        st.dataframe(segs); st.stop()

    rA, rB, rC = r_user
    seg_reach = {
        f"{names[0]} only": rA,
        f"{names[1]} only": rB,
        f"{names[2]} only": rC,
        f"{names[0]} & {names[1]} only": 1 - (1 - rA)*(1 - rB),
        f"{names[0]} & {names[2]} only": 1 - (1 - rA)*(1 - rC),
        f"{names[1]} & {names[2]} only": 1 - (1 - rB)*(1 - rC),
        "All three": 1 - (1 - rA)*(1 - rB)*(1 - rC),
        "None of these": 0.0
    }
    segs["Reach within segment"] = segs["Segment"].map(seg_reach)
    segs["Contribution"] = segs["Share"] * segs["Reach within segment"]

    total_cross = segs["Contribution"].sum()
    st.metric("Total cross-media reach (population)", f"{total_cross:.1%}")

    users_total = 1 - none
    if users_total > 1e-9:
        st.caption(f"Among users of any of these media (union={users_total:.1%}), campaign reaches {(total_cross/users_total):.1%}.")

    with st.expander("Segments & math"):
        st.dataframe(
            segs.style.format({"Share":"{:.2%}", "Reach within segment":"{:.2%}", "Contribution":"{:.2%}"}),
            use_container_width=True
        )

    base = 1 - (1 - camp[0])*(1 - camp[1])*(1 - camp[2])
    st.caption(f"Independence baseline on campaign reaches (ignoring usage): {base:.1%}")


# ---------- Mode 4 (MONTHLY USAGE MATRIX ONLY, n channels) ----------
elif mode == "Overlap-aware (n channels, pairwise approx)":
    st.subheader("Overlap-aware (n channels): monthly usage matrix + campaign reaches")
    st.write("Edit the monthly usage matrix U(A), U(A∩B) (% of population). Enter campaign reaches P(A). "
             "The app converts U(A∩B) → campaign pairwise P(A∩B) using within-user reach rA = P(A)/U(A).")

    # Catalog & selection
    catalog = ["TV","Radio","OOH","Print","Cinema","VOD","Social media","Search","Other sites"]
    chosen = st.multiselect("Channels to include", catalog, default=["TV","Social media","Print"])
    if len(chosen) < 2:
        st.info("Select at least two channels."); st.stop()

    # Default MONTHLY USAGE table (your values). Diagonal = U(A), off-diagonals = U(A∩B).
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

    # Helpers for parsing percent/decimal (incl. commas)
    def pct_to_unit_local(x):
        if x is None or x == "": return None
        try: x = float(str(x).replace(",", "."))
        except: return None
        return x/100.0 if x > 1 else x
    def unit_to_pct_local(x): return None if x is None else x*100

    # Editable USAGE matrix in %
    mat = pd.DataFrame(index=chosen, columns=chosen, dtype=float)
    for a in chosen:
        for b in chosen:
            mat.loc[a,b] = unit_to_pct_local(default_usage.get((a,b), np.nan))
    st.write("**Monthly usage matrix U** — edit values (% of population). Diagonal cells = U(A).")
    U_edit = st.data_editor(mat, use_container_width=True, num_rows="fixed", key="usage_matrix_editor")

    # Campaign reaches (user-entered) % of population
    st.write("**Campaign reaches by channel** (% of population).")
    marg_df = pd.DataFrame({"Channel": chosen, "Reach %": [None]*len(chosen)})
    marg_edit = st.data_editor(
        marg_df,
        column_config={
            "Channel": st.column_config.TextColumn(disabled=True),
            "Reach %": st.column_config.NumberColumn("Reach %", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
        },
        hide_index=True,
        use_container_width=True,
        key="campaign_marginals_editor",
    )
    A = {ch: pct_to_unit_local(marg_edit.loc[i, "Reach %"]) for i, ch in enumerate(chosen)}

    # Parse usage matrix to proportions & symmetrize off-diagonals
    U = U_edit.copy()
    for a in chosen:
        for b in chosen:
            U.loc[a,b] = pct_to_unit_local(U.loc[a,b])
    for i,a in enumerate(chosen):
        for j,b in enumerate(chosen):
            if j <= i: continue
            ab, ba = U.loc[a,b], U.loc[b,a]
            both = [x for x in [ab,ba] if x is not None]
            v = 0.5*(ab+ba) if len(both)==2 else (both[0] if len(both)==1 else None)
            U.loc[a,b] = U.loc[b,a] = v

    # Validations
    for a in chosen:
        ua, pa = U.loc[a,a], A[a]
        if ua is None:
            st.error(f"Diagonal U({a}) is required."); st.stop()
        if ua < 0 or ua > 1:
            st.error(f"U({a}) must be 0–1 (or 0–100%)."); st.stop()
        if pa is None:
            st.error(f"Campaign reach P({a}) is required."); st.stop()
        if pa < 0 or pa > 1:
            st.error(f"P({a}) must be 0–1 (or 0–100%)."); st.stop()
        if pa - ua > 1e-9:
            st.error(f"P({a}) ({pa:.2%}) cannot exceed usage U({a}) ({ua:.2%})."); st.stop()
    # Validate pairwise usage bounds
    for i,a in enumerate(chosen):
        for j,b in enumerate(chosen):
            if j <= i: continue
            uab = U.loc[a,b]
            if uab is None:
                st.error(f"Please fill U({a}∩{b})."); st.stop()
            if uab < 0 or uab > min(U.loc[a,a], U.loc[b,b]) + 1e-9:
                st.error(f"U({a}∩{b}) must be ≤ min(U({a}),U({b}))."); st.stop()

    # Convert to within-user reach and campaign pairwise
    r = {a: (0.0 if U.loc[a,a] in (None,0) else min(1.0, A[a]/U.loc[a,a])) for a in chosen}
    P2 = pd.DataFrame(index=chosen, columns=chosen, dtype=float)
    for a in chosen:
        P2.loc[a,a] = A[a]
    for i,a in enumerate(chosen):
        for j,b in enumerate(chosen):
            if j <= i: continue
            uab = U.loc[a,b]
            pab = float(uab * r[a] * r[b])  # P_campaign(A∩B) ≈ U(A∩B)*rA*rB
            # Clip to feasibility
            pab = min(pab, A[a], A[b])
            P2.loc[a,b] = P2.loc[b,a] = pab

    # Bounds + heuristic triples (Kirkwood)
    chans = chosen[:]  # preserve order for incremental view
    sum_A = sum(A[c] for c in chans)
    sum_pairs = 0.0
    for i in range(len(chans)):
        for j in range(i+1, len(chans)):
            sum_pairs += P2.loc[chans[i], chans[j]]

    lower_pair = max(max(A[c] for c in chans), sum_A - sum_pairs)   # Bonferroni LB
    upper_simple = min(1.0, sum_A)                                  # simple UB

    def est_triple(a, b, c):
        A1, A2, A3 = A[a], A[b], A[c]
        P_ab, P_ac, P_bc = P2.loc[a,b], P2.loc[a,c], P2.loc[b,c]
        denom = max(1e-12, A1*A2*A3)
        t = (P_ab * P_ac * P_bc) / denom
        lower = max(0.0, (P_ab + P_ac + P_bc) - A1 - A2 - A3)
        upper = min(P_ab, P_ac, P_bc)
        return float(np.clip(t, lower, upper))

    triple_sum = 0.0
    if len(chans) >= 3:
        for i in range(len(chans)):
            for j in range(i+1, len(chans)):
                for k in range(j+1, len(chans)):
                    triple_sum += est_triple(chans[i], chans[j], chans[k])

    est_union = float(np.clip(sum_A - sum_pairs + triple_sum, lower_pair, upper_simple))

    c1, c2 = st.columns(2)
    with c1: st.metric("Estimated cross-media reach (n-channel)", f"{est_union:.1%}")
    with c2: st.caption(f"Bounds: LB≈{lower_pair:.1%}, UB≈{upper_simple:.1%}")

    # OPTIONAL VISUALS
    show_reach = st.checkbox("Show per-channel campaign reach bars", value=True)
    show_within = st.checkbox("Show within-user reach rᵢ bars (P(A)/U(A))", value=False)

    if show_reach:
        reach_df = pd.DataFrame({"Channel": chans, "Reach": [A[c] for c in chans]})
        st.altair_chart(
            alt.Chart(reach_df).mark_bar().encode(
                x=alt.X("Channel:N", sort=None), y=alt.Y("Reach:Q", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("Channel:N"), alt.Tooltip("Reach:Q", format=".1%")]
            ).properties(height=280), use_container_width=True
        )

    if show_within:
        r_df = pd.DataFrame({"Channel": chans, "Within-user rᵢ": [r[c] for c in chans]})
        st.altair_chart(
            alt.Chart(r_df).mark_bar().encode(
                x=alt.X("Channel:N", sort=None), y=alt.Y("Within-user rᵢ:Q", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("Channel:N"), alt.Tooltip("Within-user rᵢ:Q", format=".1%")]
            ).properties(height=280), use_container_width=True
        )

    # Incremental (approx): scale independence to match est_union
    r_vec = np.array([A[c] for c in chans])
    indep_union = 1 - np.prod(1 - r_vec)
    scale = 1.0
    if indep_union > 1e-9 and est_union > 0:
        for _ in range(8):
            prod_term = np.prod(1 - scale*r_vec)
            f = 1 - prod_term - est_union
            if abs(f) < 1e-10: break
            g = -prod_term * np.sum(r_vec / (1 - scale*r_vec))
            if abs(g) < 1e-12: break
            scale = np.clip(scale - f/g, 0.0, 5.0)
    r_scaled = np.clip(scale * r_vec, 0, 1)
    # incremental by order
    inc, unreached = [], 1.0
    for ri in r_scaled:
        inc.append(unreached * ri)
        unreached *= (1 - ri)
    inc_df = pd.DataFrame({"Channel": chans, "Incremental (approx)": inc})
    st.altair_chart(
        alt.Chart(inc_df).mark_bar().encode(
            x=alt.X("Channel:N", sort=None),
            y=alt.Y("Incremental (approx):Q", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("Channel:N"), alt.Tooltip("Incremental (approx):Q", format=".1%")],
        ).properties(height=320),
        use_container_width=True
    )

    with st.expander("Matrices & details"):
        # Show U and the derived campaign P(A∩B)
        st.markdown("**Monthly usage U (after edits, %)**")
        st.dataframe(U.applymap(lambda v: None if v is None else round(v*100, 2)))
        st.markdown("**Derived campaign P(A∩B) used for union (%, after clipping)**")
        st.dataframe(P2.applymap(lambda v: None if v is None else round(v*100, 2)))
        details = pd.DataFrame({
            "Channel": chans,
            "U(A) monthly": [U.loc[c,c] for c in chans],
            "P(A) campaign": [A[c] for c in chans],
            "Within-user rᵢ": [r[c] for c in chans],
        }).style.format({"U(A) monthly":"{:.2%}", "P(A) campaign":"{:.2%}", "Within-user rᵢ":"{:.2%}"})
        st.markdown("**Per-channel summary**")
        st.dataframe(details, use_container_width=True)

