import dash
import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc, html, Input, Output, ClientsideFunction
import pandas as pd
from flask_caching import Cache
import os


app = dash.Dash(external_stylesheets=[dbc.themes.GRID],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app. server
app.title = "Hedging Strategy Dashboard"


cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '.'
})



TIMEOUT = 60

@cache.memoize(timeout=TIMEOUT)
def query_data(score_value):
    # This could be an expensive data querying step
    validation = prepare_df(score_value)
    return validation

import plotly.io as pio

skilling_template = go.layout.Template(
    # LAYOUT
    layout = {
        'plot_bgcolor':"white",
        # Fonts
        # Note - 'family' must be a single string, NOT a list or dict!
        'title':
            {'font': {'family': 'HelveticaNeue-CondensedBold, Helvetica, Sans-serif',
                      'size':30,
                      'color': '#333'}
            },
        'font': {'family': 'Roboto',
                      'size':16,
                      'color': '#333'},
        # Colorways
        'colorway': ['#bdd7e7','#6baed6','#3182bd','#31a354'],
        # Keep adding others as needed below
        'hovermode': 'x unified'
    },
    # DATA
    # data = {
    #     # Each graph object must be in a tuple or list for each trace
    #     'bar': [go.Bar(texttemplate = '%{value:$.2s}',
    #                    textposition='outside',
    #                    textfont={'family': 'Helvetica Neue, Helvetica, Sans-serif',
    #                              'size': 20,
    #                              'color': '#FFFFFF'
    #                              })]
    # }
)


pio.templates.default = skilling_template

#load data
df = px.data.stocks()
df2 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
x_test = pd.read_csv("data/x_test.csv")
# validation = pd.read_csv("data/validation_result.csv")
stats_df = pd.read_csv("data/thresholds.csv")



def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Define score threshold"),
                   dcc.Slider( id='slider',
                            min=0,
                            max=1,
                            step=0.01,
                            value=.3,
                            className="form-range",
                            marks=None,
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True
                            }),
            html.Br(),
            html.Div("* - The evaluation is done for limited list of countries",style={"font-style":"italic","font-size":10}),
            # html.P("Performance"),
            # dcc.Dropdown(
            #     id="dropdown",
            #     options=[
            #                       {
            #                           'label': 'Gross PnL',
            #                           'value': 'GOOG'
            #                       },
            #                       {
            #                           'label': 'Revenue',
            #                           'value': 'AAPL'
            #                       },
            #                       {
            #                           'label': 'Hybrid Revenue',
            #                           'value': 'AMZN'
            #                       },
            #                   ],
            #     value="GOOG",
            # ),
            # html.Br(),
            # html.P("Risk"),
            # dcc.Dropdown(
            #     id="admit-select",
            #     options=[
            #                       {
            #                           'label': 'PnL sd',
            #                           'value': 'GOOG'
            #                       },
            #                       {
            #                           'label': 'PnL variance',
            #                           'value': 'AAPL'
            #                       },
            #                       {
            #                           'label': 'Max drawdown',
            #                           'value': 'AMZN'
            #                       },
            #                   ],
            #     value="AMZN"
            # ),
            html.Br(),
            # html.Div(
            #     id="reset-btn-outer",
            #     children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
            # ),
        ],
    )








def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H3("Hedging strategy simulation",style={"color":"#3182bd"}),
            html.Div(
                id="intro",
                children=[html.P("Based on different score threshold explore potential performance* vs actual numbers using variety business measures")


                          ],
            ),
        ],
    )



def prepare_df(threshold):
    x_test['profitable'] =    x_test['score'].apply(lambda y: 1 if y>threshold else 0)
    x_test['simulated_pnl'] = x_test.apply(lambda x: x['theo_a_revenue_eur_sum'] if x['profitable']==0 else x['theo_b_revenue_eur_sum'], axis=1)
    x_test['simulated_hedge_volume'] = x_test.apply(lambda x: x['eur_volume_sum'] if x['profitable']==0 else 0, axis=1)


    df = pd.melt((
        x_test
        .assign(hedge_volume=lambda x: (1-x['warehoused_ratio'])*x['eur_volume_sum'])
        .groupby(['trade_monthname','trade_month'],as_index=False)['profitable','simulated_hedge_volume','gross_pnl_eur_sum','simulated_pnl',"theo_b_revenue_eur_sum","theo_a_revenue_eur_sum","eur_volume_sum","hedge_volume","simulated_hedge_volume"]
        .agg(profitable_cnt=("profitable","count"),
              profitable_sum=("profitable","sum"),
              gross_pnl_eur_sum=("gross_pnl_eur_sum","sum"),
              simulated_pnl = ("simulated_pnl","sum"),
              theo_a_revenue_eur_sum=("theo_a_revenue_eur_sum","sum"),
              theo_b_revenue_eur_sum=("theo_b_revenue_eur_sum","sum"),
              eur_volume_sum = ("eur_volume_sum","sum"),
              hedge_volume = ("hedge_volume","sum"),
              simulated_hedge_volume = ("simulated_hedge_volume","sum")
            )
        .assign(real_hedge_ratio = lambda x: x['hedge_volume']/x['eur_volume_sum'],
                simulated_hedge_ratio = lambda x: x['simulated_hedge_volume']/x['eur_volume_sum'],
                simulated_hedge_ratio2 = lambda x: (x['profitable_cnt']-x['profitable_sum'])/x['profitable_cnt'],  #absolute values
               )
        .sort_values('trade_month')

    ).assign(
            # assign the cumulative sum of each name as a new column
            cumulative_simulated_pnl=lambda x: x['simulated_pnl'].cumsum(),
            cumulative_theoa=lambda x: x['theo_a_revenue_eur_sum'].cumsum(),
            cumulative_theob=lambda x: x['theo_b_revenue_eur_sum'].cumsum(),
            cumulative_real_pnl=lambda x: x['gross_pnl_eur_sum'].cumsum()
        ),id_vars=['trade_monthname','trade_month'], value_vars=["simulated_hedge_ratio2",'cumulative_real_pnl',"theo_a_revenue_eur_sum","theo_b_revenue_eur_sum","cumulative_simulated_pnl","gross_pnl_eur_sum","simulated_pnl","real_hedge_ratio","simulated_hedge_ratio"])

    return df



@app.callback(Output(component_id='bar_plot', component_property='figure'),
              [Input(component_id='slider', component_property='value')])

def graph_update(score_value=0.3):
    # bar chart
    validation = query_data(score_value)
    validation['value']=abs(validation['value'])
    bar_fig = px.bar(validation.loc[validation['variable'].isin(['gross_pnl_eur_sum', 'theo_a_revenue_eur_sum','theo_b_revenue_eur_sum','simulated_pnl'])],
                     x="trade_monthname", y="value", color="variable", text_auto=True)
    bar_fig.update_traces(texttemplate="%{y:.2s}")
    bar_fig.update_layout(barmode='group',title_x=0)
    bar_fig.update_layout(legend_title=None, xaxis_title=None, legend=dict(
        orientation="h",

        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return bar_fig




@app.callback(Output(component_id='line_plot', component_property='figure'),
              [Input(component_id='slider', component_property='value')])

def graph_update(score_value=0.3):
    # bar chart
    validation = query_data(score_value)
    validation['value']=abs(validation['value'])

    # cumulative chart
    line_fig = px.line(validation.loc[validation['variable'].isin(['cumulative_simulated_pnl','cumulative_theoa','cumulative_theob', 'cumulative_real_pnl'])],
                       x="trade_monthname",
                       y="value", color="variable",text='value',color_discrete_map={
        'cumulative_real_pnl': '#3182bd',
        'cumulative_simulated_pnl': '#31a354'
    })
    line_fig.update_traces(textposition="bottom right",texttemplate="%{y:.2s}")
    line_fig.update_layout(
        xaxis_title=None,
        legend_title=None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
    return line_fig



# hedge ratio bar chart
@app.callback(Output(component_id='hedge_bar_plot', component_property='figure'),
              [Input(component_id='slider', component_property='value')])

def graph_update(score_value=0.3):
    # bar chart
    validation = query_data(score_value)
    validation['value']=abs(validation['value'])
    ratios = validation.loc[validation['variable'] == "simulated_hedge_ratio2"]
    bar_fig = px.bar(validation.loc[validation['variable'].isin(['real_hedge_ratio','simulated_hedge_ratio'])],
                     x="trade_monthname", y="value", color="variable", text_auto=True ,   color_discrete_map={
        'real_hedge_ratio': '#3182bd',
        'simulated_hedge_ratio': '#31a354'
    })
    bar_fig.update_traces(texttemplate="%{y:,.0%}")
    bar_fig.add_scatter(x=ratios['trade_monthname'], y=ratios['value'], text=ratios['value'], name="Absolute",mode="text,lines",marker=dict(color="gray"),texttemplate="%{y:,.0%}")
    bar_fig.update_layout(barmode='group',title_x=0)
    bar_fig.layout.yaxis.tickformat = ',.0%'
    bar_fig.update_layout(legend_title=None, xaxis_title=None, legend=dict(
        orientation="h",

        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return bar_fig



#threshold chart
score_fig = px.line(stats_df, x="thresholds", y="total_pnl",text="total_pnl")
score_fig.update_traces(textposition="bottom right",texttemplate="%{y:.2s}")
score_fig.add_hline(y=abs(x_test['gross_pnl_eur_sum'].sum()), line_width=2, line_dash="dash", line_color="#969696",annotation_text="Real", annotation_position="bottom right")
score_fig.add_hline(y=abs(x_test['theo_b_revenue_eur_sum'].sum()), line_width=2, line_dash="dash", line_color="#3182bd",annotation_text="Theo B", annotation_position="bottom right")


#threshold chart
risk_fig = px.line(stats_df, x="thresholds", y="stds",text="stds")
risk_fig.update_traces(textposition="bottom right",texttemplate="%{y:.2s}")
risk_fig.add_hline(y=x_test['gross_pnl_eur_sum'].std(), line_width=2, line_dash="dash", line_color="#969696",annotation_text="Real", annotation_position="bottom right")



app.layout = \
dbc.Container\
([
    html.Br(),
    dbc.Row([
    dbc.Col([html.Img(src=app.get_asset_url("skilling-logo.svg"))],width=1),
    dbc.Col([],width=11),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([description_card(), generate_control_card()], width=2, lg=2, md=10, xs=12),

        dbc.Col([html.B("Hedged volumes comparison"),
                 html.Hr(), dcc.Graph(id='hedge_bar_plot')], width=10, lg=10, md=5, xs=12),
    ]),
    dbc.Row([
    dbc.Col([],width=2, lg=2, md=4, xs=12),

    dbc.Col([html.B("Actual vs Potential PnL on validation set"),
                        html.Hr(),dcc.Graph(id='bar_plot')],width=5, lg=5, md=5, xs=12),
    dbc.Col([html.B("Cumulative PnL on validation set"),
                        html.Hr(),dcc.Graph(
                                  id="line_plot")],width=5, lg=5, md=5, xs=12),
    ]),
    html.Br(),

    dbc.Row([
    dbc.Col([],width=2),
    dbc.Col([dcc.Tabs(
            [
                dcc.Tab(label="PnL",
                        children=[ html.B("Potential PnL vs Actual across different thresholds"),
                                   html.Hr(),
                                   dcc.Graph(id="wait_time_table",
                                  figure=score_fig)]),
                dcc.Tab(label="SD", children=[html.B("Potential PnL vs Actual across different thresholds"),
                                   html.Hr(),dcc.Graph(figure=risk_fig)])
            ],
            id="tabs"
        )
       ],width=10, lg=10, md=10, xs=12),
    ])
], fluid=True,style={"margin":10,"padding":10})

if __name__ == "__main__":
    app.run_server(debug=False, port=7777, host='0.0.0.0')