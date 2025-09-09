# ---------- Mode 4 (UPDATED) ----------
elif mode == "Overlap-aware (n channels, pairwise approx)":
    st.subheader("Overlap-aware (n channels): pairwise matrix + heuristic triples")
    st.write("Pick channels, enter campaign reaches (population %). Then choose whether your matrix is campaign pairwise or monthly usage overlaps; we'll convert if needed.")

    # Catalog & selection
    catalog = ["TV","Radio","OOH","Print","Cinema","VOD","Social media","Search","Other sites"]
    chosen = st.multiselect("Channels to include", catalog, default=["TV","Social media","Print"])
    if len(chosen) < 2:
        st.info("Select at least two channels."); st.stop()

    # Matrix type
    matrix_type = st.radio("This matrix represents", 
                           ["Campaign overlaps P(A∩B)", "Monthly usage overlaps U(A∩B)"],
                           index=1,  # default to usage for your defaults
                           help="If you only have monthly usage overlaps, choose 'Monthly usage'. We'll convert to campaign pairwise using your campaign reaches.")

    # Your default table (interpreted as usage by default). Diagonal are marginals for that matrix.
    default_pairs = {
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

    # Build editable matrix in % units (accept commas)
    def unit_to_pct(x): return None if x is None else x * 100.0
    def pct_to_unit(x):
        if x is None or x == "": return None
        try: x = float(str(x).replace(",", "."))
        except: return None
        return x/100.0 if x > 1 else x

    mat = pd.DataFrame(index=chosen, columns=chosen, dtype=float)
    for a in chosen:
        for b in chosen:
            mat.loc[a, b] = unit_to_pct(default_pairs.get((a,b), np.nan))
    st.write("**Matrix editor** — edit values (% of population). Diagonal cells are marginals for the selected matrix type.")
    mat_edit = st.data_editor(mat, use_container_width=True, num_rows="fixed", key="pairwise_matrix_editor2")

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
        key="marginals_editor2",
    )
    A = {ch: pct_to_unit(marg_edit.loc[i, "Reach %"]) for i, ch in enumerate(chosen)}
    for ch in chosen:
        if A[ch] is None:
            st.info("Enter all campaign reaches to proceed."); st.stop()
        if A[ch] < 0 or A[ch] > 1:
            st.error(f"P({ch}) must be 0–1 (or 0–100%)."); st.stop()

    # Parse matrix to proportions & symmetrize
    M = mat_edit.copy()
    for a in chosen:
        for b in chosen:
            M.loc[a,b] = pct_to_unit(M.loc[a,b])

    # Symmetrize off-diagonals (average A∩B and B∩A)
    for i,a in enumerate(chosen):
        for j,b in enumerate(chosen):
            if j <= i: continue
            ab = M.loc[a,b]; ba = M.loc[b,a]
            both = [x for x in [ab,ba] if x is not None]
            if len(both)==2: v = 0.5*(ab+ba)
            elif len(both)==1: v = both[0]
            else: v = None
            M.loc[a,b] = M.loc[b,a] = v

    # Derive campaign pairwise matrix P2 depending on matrix type
    P2 = pd.DataFrame(index=chosen, columns=chosen, dtype=float)
    if matrix_type == "Campaign overlaps P(A∩B)":
        # Diagonal should equal campaign marginals; warn if not close, but use campaign marginals for safety.
        for a in chosen:
            if M.loc[a,a] is not None and abs(M.loc[a,a] - A[a]) > 1e-6:
                st.warning(f"Diagonal for {a} = {M.loc[a,a]:.2%} does not match campaign P({a}) = {A[a]:.2%}. Using campaign marginal.")
        for a in chosen:
            for b in chosen:
                P2.loc[a,b] = (A[a] if a==b else M.loc[a,b])
    else:
        # Monthly usage overlaps -> convert to campaign pairs
        # r_a = campaign(A)/usage(A); pair(A,B) = usage(A∩B) * r_a * r_b
        usage = {a: M.loc[a,a] for a in chosen}
        # Validate campaign <= usage
        bad = [a for a in chosen if usage[a] is not None and A[a] is not None and A[a] - usage[a] > 1e-9]
        if bad:
            st.error("Campaign reach cannot exceed usage on the diagonal for: " + ", ".join(bad))
            st.stop()
        r = {a: (0.0 if (usage[a] in (None,0)) else min(1.0, A[a]/usage[a])) for a in chosen}
        for a in chosen:
            for b in chosen:
                if a==b:
                    P2.loc[a,b] = A[a]
                else:
                    uab = M.loc[a,b]
                    P2.loc[a,b] = None if uab is None else float(uab * r[a] * r[b])

    # Validate & clip pairs to ≤ min marginals
    auto_clip = st.checkbox("Auto-clip P(A∩B) to ≤ min(P(A),P(B))", value=True)
    clipped = False
    for a in chosen:
        for b in chosen:
            val = P2.loc[a,b]
            if val is None: continue
            if val < 0 or val > 1:
                st.error(f"P({a}∩{b}) must be 0–1 after conversion."); st.stop()
            if a != b:
                mx = min(A[a], A[b])
                if val > mx + 1e-9 and auto_clip:
                    P2.loc[a,b] = mx
                    clipped = True
    if clipped:
        st.warning("Some pairwise joints exceeded min(P(A), P(B)) and were clipped.")

    # Compute Bonferroni bounds and heuristic union (pairs + triples)
    chans = [c for c in chosen if A[c] is not None]
    sum_A = sum(A[c] for c in chans)
    sum_pairs = 0.0
    for i in range(len(chans)):
        for j in range(i+1, len(chans)):
            val = P2.loc[chans[i], chans[j]]
            sum_pairs += (0.0 if val is None else val)

    lower_pair = max(max(A[c] for c in chans), sum_A - sum_pairs)  # LB
    upper_simple = min(1.0, sum_A)                                 # UB

    def est_triple(a, b, c):
        A1, A2, A3 = A[a], A[b], A[c]
        P_ab, P_ac, P_bc = P2.loc[a,b], P2.loc[a,c], P2.loc[b,c]
        if None in (A1,A2,A3,P_ab,P_ac,P_bc): return None
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
                    tval = est_triple(chans[i], chans[j], chans[k])
                    if tval is not None:
                        triple_sum += tval

    est_union = sum_A - sum_pairs + triple_sum
    est_union = float(np.clip(est_union, lower_pair, upper_simple))

    c1, c2 = st.columns(2)
    with c1: st.metric("Estimated cross-media reach (n-channel)", f"{est_union:.1%}")
    with c2: st.caption(f"Bounds: LB≈{lower_pair:.1%}, UB≈{upper_simple:.1%}")

    # Approx incremental: scale independence to match union
    r = np.array([A[c] for c in chans])
    indep_union = 1 - np.prod(1 - r)
    scale = 1.0
    if indep_union > 1e-9 and est_union > 0:
        for _ in range(8):
            prod_term = np.prod(1 - scale*r)
            f = 1 - prod_term - est_union
            if abs(f) < 1e-10: break
            g = -prod_term * np.sum(r / (1 - scale*r))
            if abs(g) < 1e-12: break
            scale = np.clip(scale - f/g, 0.0, 5.0)
    r_scaled = np.clip(scale * r, 0, 1)
    inc = []
    unreached = 1.0
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

    with st.expander("Matrix used for union (campaign P(A∩B) after conversion/clipping)"):
        st.dataframe(P2.applymap(lambda v: None if v is None else round(v*100, 2)))
