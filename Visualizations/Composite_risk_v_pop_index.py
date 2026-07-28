import pandas as pd
import altair as alt

df = pd.read_csv('/Users/juliettecarson/Desktop/Plots/Texas_Soil_Risk_with_Indices.csv')

def norm(col):
    return (df[col] - df[col].min()) / (df[col].max() - df[col].min())

df['Population_Index'] = (norm('Population') + norm('Population_Density_per_sq_mi') + norm('Urbanization_Percent')) / 3

chart = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
    x=alt.X('Population_Index', title='Population Index'),
    y=alt.Y('Composite_Risk_Score', title='Composite Risk Score'),
    tooltip=['County', 'Population_Index', 'Composite_Risk_Score']
).properties(width=500, height=400, title='Composite Risk Score vs Population Index')

trend = chart.transform_regression('Population_Index', 'Composite_Risk_Score').mark_line(color='red')

sorted_df = df.sort_values('Population_Index', ascending=False)
frontier_rows = []
max_y = -float('inf')
for _, row in sorted_df.iterrows():
    if row['Composite_Risk_Score'] > max_y:
        frontier_rows.append(row)
        max_y = row['Composite_Risk_Score']
frontier_df = pd.DataFrame(frontier_rows).sort_values('Population_Index')

pareto_line = alt.Chart(frontier_df).mark_line(color='green', strokeDash=[4,2]).encode(x='Population_Index', y='Composite_Risk_Score')
pareto_points = alt.Chart(frontier_df).mark_point(color='green', size=80, filled=True).encode(x='Population_Index', y='Composite_Risk_Score', tooltip=['County'])

chart = chart + trend + pareto_line + pareto_points
chart.show()
chart.save('/Users/juliettecarson/Desktop/Plots/Population_index_v_Risk.pdf')