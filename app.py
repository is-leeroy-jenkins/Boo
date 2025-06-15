'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                app.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="main.py" company="Terry D. Eppler">

         Boo is a df analysis tool integrating GenAI, Text Processing, and Machine-Learning
         algorithms for federal analysts.
         Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    app.py
  </summary>
  ******************************************************************************************
'''
from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

app = Dash( )

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame( {
    'Fruit': [ 'Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 'Bananas' ],
    'Amount': [ 4, 1, 2, 2, 4, 5 ],
    'City': [ 'SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal' ]
})

fig = px.bar(df, x='Fruit', y='Amount', color='City', barmode='group')

fig.update_layout(
    plot_bgcolor=colors[ 'background' ],
    paper_bgcolor=colors[ 'background' ],
    font_color=colors['text']
)

app.layout = html.Div( style={ 'backgroundColor': colors['background' ]}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors[ 'text' ]
        }
    ),

    html.Div( children='Dash: A web application framework for your data.', style={
        'textAlign': 'center',
        'color': colors[ 'text' ]
    }),

    dcc.Graph(
        id='example-graph-2',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run( debug=True )