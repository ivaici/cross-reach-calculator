import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Cross-Reach Calculator", page_icon="📈", layout="centered")
st.title("📈 Cross-Reach Calculator")
st.caption(
    "Modes: (1) Independence (Sainsbury), (2) Overlap-aware (inclusion–exclusion, up to 3 channels), "
    "(3) Usage-adjusted (3 channels, using monthly usage overlaps)."
)

mode = st.radio(
    "Choose a mode",
    ["Independence (any # of channels)", "Overlap-aware (2–3 channels)", "Usage-adjusted (3 channels)"],
    help=("Independence: 1 − ∏(1 − Rᵢ).  "
          "Overlap-aware: inclusion–exclusion with pairwise/triple overlaps (exact for 3).  "
          "Usage-adjusted: convert campaign reach to within-user reach; weight by usage segments.")
)

# ---------- helpers ----------
def pct_to_unit(x):
    if x is None or x == "":
        return None
    try:
        x = float(x)
    except:
        return None
    return x/100.0 if x > 1 else x

def fmt_pct(x): return "–" if x is None else f"{x:.2%}"

def warn_if_prob_invalid(val, name):
    if val is None: return False
    if val < 0 or val > 1:
        st.error(f"{name} must be between 0 and 1 (or 0–100%). You entered {val:.3f}.")
        return True
    return False

# ---------- Mode 1 ----------
if mode == "Independence (any # of channels)":
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

    # incremental by order (row order = planning sequence)
    incrementals, unreached = [], 1.0
    for ri in r:
        inc = unreached * ri
        incrementals.append(inc)
        unreached *= (1 - ri)

    chart_df = pd.DataFrame({"Channel": edited["Channel"], "Incremental": incrementals})
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Channel:N", sort=None, title="Channel"),
            y=alt.Y("Incremental:Q", axis=alt.Axis(format="%"), title="Incremental Reach"),
            tooltip=[alt.Tooltip("Channel:N"), alt.Tooltip("Incremental:Q", format=".1%")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Details"):
        details = edited.copy()
        details["Reach (0–1)"] = r
        details["1 − Reach"]   = 1 - r
        details["Incremental"] = incrementals
        st.dataframe(details, use_container_width=True)

    st.info("Assumes independence across channels. For overlaps, use the next mode.")

# ---------- Mode 2 ----------
elif mode == "Overlap-aware (2–3 channels)":
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

    # optional triple for 3 channels
    triple = None
    if len(keep) == 3:
        i,j,k = keep
        triple = pct_to_unit(st.text_input(f"{names[i]} ∩ {names[j]} ∩ {names[k]} (optional)", value=""))
        if triple is not None and (triple < 0 or triple > 1):
            st.error("Triple overlap must be 0–1."); st.stop()
        # bounds hint
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
        st.caption(f"{names[i]} + {names[j]} − {names[i]}∩{names[j]}")
    else:
        i,j,k = keep
        ab = pair.get((min(i,j),max(i,j)))
        ac = pair.get((min(i,k),max(i,k)))
        bc = pair.get((min(j,k),max(j,k)))
        if ab is None or ac is None or bc is None:
            st.info("Enter all three pairwise overlaps. Triple is optional (will be estimated).")
            st.stop()
        if triple is None:
            a,b,c = reaches[i], reaches[j], reaches[k]
            est = (ab * ac * bc) / max(1e-12, a*b*c)  # heuristic
            lower = max(0.0, ab + ac + bc - a - b - c); upper = min(ab, ac, bc)
            triple = float(np.clip(est, lower, upper))
            st.caption(f"Estimated triple via heuristic: {fmt_pct(triple)} (clipped to feasible bounds).")
        cross = reaches[i] + reaches[j] + reaches[k] - ab - ac - bc + triple
        st.metric("Cross-media reach (A ∪ B ∪ C)", f"{cross:.1%}")

    st.info("Exact for 3 channels when pairwise and triple overlaps are provided.")

# ---------- Mode 3 ----------
else:
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
            st.error(f"{names[i]} campaign reach ({fmt_pct(camp[i])}) cannot exceed usage P({names[i]}) ({fmt_pct(usage[i])}).")
            st.stop()
    for (i,j), v in pair.items():
        if warn_if_prob_invalid(v, f"P({names[i]}∩{names[j]})"): st.stop()
        if usage[i] is not None and usage[j] is not None and v is not None:
            if v - min(usage[i], usage[j]) > 1e-9:
                st.error(f"P({names[i]}∩{names[j]}) cannot exceed min(P({names[i]}), P({names[j]}))."); st.stop()
    if triple is not None:
        if warn_if_prob_invalid(triple, f"P({names[0]}∩{names[1]}∩{names[2]})"): st.stop()
        ab, ac, bc = pair[(0,1)], pair[(0,2)], pair[(1,2)]
        if None not in (ab, ac, bc):
            lower = max(0.0, ab + ac + bc - usage[0] - usage[1] - usage[2])
            upper = min(ab, ac, bc)
            if not (lower - 1e-9 <= triple <= upper + 1e-9):
                st.warning(f"P(ABC) outside feasible bounds [{lower:.2%}, {upper:.2%}].")

    # within-user campaign reach
    r_user = [0.0 if usage[i] in (None, 0) else min(1.0, camp[i] / usage[i]) for i in range(3)]

    # disjoint usage segments via inclusion–exclusion
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

    # reach within each segment (Sainsbury within the channels they use)
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
