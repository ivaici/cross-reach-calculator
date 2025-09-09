import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Cross-Reach Calculator", page_icon="📈", layout="centered")
st.title("📈 Cross-Reach Calculator")
st.caption("Sainsbury formula: Cross Reach = 1 − ∏(1 − Rᵢ)")

st.sidebar.header("Inputs")
rows = st.sidebar.slider("Rows (channels)", 3, 30, 8, step=1)

seed = [
    {"Channel":"TV","Reach %":65.0},
    {"Channel":"Social","Reach %":35.0},
    {"Channel":"Print","Reach %":15.0},
]
if len(seed) < rows:
    seed += [{"Channel":"", "Reach %":None} for _ in range(rows - len(seed))]

st.write("Enter channel names and reach % (0–100). Add/remove rows as needed.")
df = pd.DataFrame(seed[:rows])

edited = st.data_editor(
    df, num_rows="dynamic", hide_index=True, use_container_width=True,
    column_config={
        "Channel": st.column_config.TextColumn("Channel", width="medium"),
        "Reach %": st.column_config.NumberColumn("Reach %", min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
    }
)

# Clean + compute
r = (edited["Reach %"].fillna(0)/100.0).clip(0, 1)
cross = 1 - np.prod(1 - r)
st.metric("Overall Cross-Media Reach", f"{cross:.1%}")

# Incremental by order (planning sequence = row order)
incrementals, prod_running = [], 1.0
for ri in r:
    inc = prod_running * ri
    incrementals.append(inc)
    prod_running *= (1 - ri)

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
    details["Reach (0-1)"] = r
    details["1 - Reach"]    = 1 - r
    details["Incremental"]  = incrementals
    st.dataframe(details, use_container_width=True)

st.info("This version assumes independence across channels. We can add overlap inputs next.")
