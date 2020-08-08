import plotly.graph_objects as go
import pandas as pd

# 2d bar graph
# fig = go.FigureWidget(data=go.Bar(y=[2, 3, 1]))
# fig.show()
# fig.write_image('figure.png')


# Read data from a csv
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
fig = go.Figure(data=[go.Surface(z=z_data.values)])
fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
# fig.show()
pth = 'tmp/test_plolty.png'
fig.write_image(pth)