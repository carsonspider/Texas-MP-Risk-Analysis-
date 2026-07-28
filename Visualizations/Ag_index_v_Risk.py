import pandas as pd
import altair as alt

df = pd.read_csv('Texas_Soil_Risk_with_Indices.csv')

alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
    x=alt.X('Ag_Index', title='MP Agriculture Risk'),
    y=alt.Y('Composite_Risk_Score', title='Composite Risk Score'),
    tooltip=['County', 'Ag_Index', 'Composite_Risk_Score']
).properties(width=500, height=400, title='Composite Risk Score vs MP Agriculture Risk')