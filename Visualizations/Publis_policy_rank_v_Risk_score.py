import pandas as pd
import altair as alt

df = pd.read_csv('/Users/juliettecarson/Desktop/Plots/Texas_Soil_Risk_with_Indices.csv')

chart = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
    x=alt.X('Policy_Priority_Rank', title='Policy Priority Rank', scale=alt.Scale(reverse=True)),
    y=alt.Y('Composite_Risk_Score', title='Composite Risk Score'),
    tooltip=['County', 'Policy_Priority_Rank', 'Composite_Risk_Score']
).properties(width=500, height=400, title='Policy Priority Rank vs Composite Risk Score (where do they diverge?)')

trend = chart.transform_regression('Policy_Priority_Rank', 'Composite_Risk_Score').mark_line(color='red')

sorted_df = df.sort_values('Policy_Priority_Rank', ascending=True)
frontier_rows = []
max_y = -float('inf')
for _, row in sorted_df.iterrows():
    if row['Composite_Risk_Score'] > max_y:
        frontier_rows.append(row)
        max_y = row['Composite_Risk_Score']
frontier_df = pd.DataFrame(frontier_rows).sort_values('Policy_Priority_Rank')

pareto_line = alt.Chart(frontier_df).mark_line(color='green', strokeDash=[4,2]).encode(x='Policy_Priority_Rank', y='Composite_Risk_Score')
pareto_points = alt.Chart(frontier_df).mark_point(color='green', size=80, filled=True).encode(x='Policy_Priority_Rank', y='Composite_Risk_Score', tooltip=['County'])

chart = chart + trend + pareto_line + pareto_points
chart.show()
chart.save('/Users/juliettecarson/Desktop/Plots/PolicyRank_v_Risk.pdf')