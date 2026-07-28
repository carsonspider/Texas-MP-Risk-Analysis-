import pandas as pd
import altair as alt

df = pd.read_csv('/Users/juliettecarson/Desktop/Plots/Texas_Soil_Risk_with_Indices.csv')

chart = alt.Chart(df).mark_circle(size=40, opacity=0.5).encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    tooltip=['County']
).properties(width=200, height=200).repeat(
    row=['Ag_Index', 'Wastewater_Index', 'Hydro_Index'],
    column=['Ag_Index', 'Wastewater_Index', 'Hydro_Index']
)

chart.show()
chart.save('/Users/juliettecarson/Desktop/Plots/Pathway_Pairwise_Matrix.pdf')