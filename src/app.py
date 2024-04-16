import dash
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc, html, Input, Output, State
import pandas as pd
from flask_caching import Cache
import dash_ag_grid as dag
import joblib, base64, io


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
app.title = "Hedging Strategy Dashboard"

server = app. server

xgb_model = joblib.load('hedging_model.pkl')

features_df = pd.read_csv("data/processed_trades_2024.csv")
features_df['trade_hour'] = features_df['trade_hour'].astype('object')
features_df  = features_df.drop('Unnamed: 0',axis=1)
uploaded_df_predicted = pd.DataFrame()





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

)


pio.templates.default = skilling_template

#load data
x_test = pd.read_csv("data/x_test.csv")
# validation = pd.read_csv("data/validation_result.csv")
stats_df = pd.read_csv("data/thresholds.csv")

uploaded_df = pd.DataFrame()



#modal popup window
modal = html.Div(
    [
        dbc.Button(
            "More", id="open-body-scroll",n_clicks=0,className="btn btn-light"
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Implementation details")),
                dbc.ModalBody(["Training data: 01.2023- 08.2023 (Germany, Sweden, Norway only)",html.Br(),
                               html.Img(src=app.get_asset_url("metrics.png")),html.Br(),
                                html.Img(src=app.get_asset_url("fi.png")),html.Br(),
                               "Validation: (0.9.2023-12.2023)",html.Br(),
                               "Method: XGB",html.Br()]
                              ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="close-body-scroll",
                        className="ms-auto",
                        n_clicks=0,
                    )
                ),
            ],
            id="modal-body-scroll",
            scrollable=True,
            is_open=False,
        ),
    ]
)


def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


app.callback(
    Output("modal-scroll", "is_open"),
    [Input("open-scroll", "n_clicks"), Input("close-scroll", "n_clicks")],
    [State("modal-scroll", "is_open")],
)(toggle_modal)

app.callback(
    Output("modal-body-scroll", "is_open"),
    [
        Input("open-body-scroll", "n_clicks"),
        Input("close-body-scroll", "n_clicks"),
    ],
    [State("modal-body-scroll", "is_open")],
)(toggle_modal)

def batch_predict(model,df):
    predictions = pd.merge(df, features_df, how='left', left_on=["Customer ID", "Position ID"],
                           right_on=["customer_id", "position_id"])
    predictions = predictions.loc[~predictions['customer_id'].isna()]
    predictions['score'] = model.predict_proba(predictions)[:, 1]
    return predictions

def calculate_running_pnl(threshold, df):
    df['profitable'] =    df['score'].apply(lambda y: 1 if y>threshold else 0)
    df['simulated_pnl'] = df.apply(lambda x: x['theo_a_revenue_eur_sum'] if x['profitable']==0 else x['theo_b_revenue_eur_sum'], axis=1)
    pnl_std = df['simulated_pnl'].std()
    constant_std= df['gross_pnl_eur_sum'].std()
    constatnt_b_pnl = df['theo_b_revenue_eur_sum'].sum()
    constatnt_real_pnl = df['gross_pnl_eur_sum'].sum()
    return df['simulated_pnl'].sum(),pnl_std,constant_std,constatnt_b_pnl,constatnt_real_pnl

def get_thresholds(df):
    thresholds = [x/10 for x in range(10)]
    y = [calculate_running_pnl(x,df) for x in thresholds]
    stds = [x[1] for x in y]
    pnls=[x[0] for x in y]
    constant_b_pnl = [x[3] for x in y]
    constant_std = [x[2] for x in y]
    constant_real_pnl = [x[4] for x in y]

    return pd.DataFrame({"thresholds":thresholds,"total_pnl":pnls,"stds":stds,"theo_b_pnl_eur":constant_b_pnl,"real_std":constant_std,"real_pnl_eur":constant_real_pnl})




@app.callback(Output('agg_preds', 'data',allow_duplicate=True),
Output(component_id="predictions",component_property="data",allow_duplicate=True),
              Output(component_id="thresholds",component_property="data",allow_duplicate=True),
               Input(component_id='initial', component_property='children'),
prevent_initial_call='initial_duplicate'
               )
def initial_loading(n):
    # validation = prepare_df(0.3)
    validation = pd.read_csv("data/validation_results.csv")
    validation['value']=abs(validation['value'])
    return validation.to_json(date_format='iso', orient='split'), x_test.to_json(date_format='iso', orient='split'),get_thresholds(x_test).to_json(date_format='iso', orient='split')


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
            html.Hr(),
            dcc.Upload(id="upload-data",children=[html.Button('Upload trades')],multiple=True),
            html.Br(),
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
                children=[html.P("Based on different score threshold explore potential performance* vs actual numbers using variety business measures"),
                          modal


                          ],
            ),
        ],
    )



def prepare_df(threshold,df=x_test):
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['trade_monthname'] = df['trade_date'].dt.month_name()
    df['trade_month'] = df['trade_date'].dt.month
    df['profitable'] =    df['score'].apply(lambda y: 1 if y>threshold else 0)
    df['simulated_pnl'] = df.apply(lambda x: x['theo_a_revenue_eur_sum'] if x['profitable']==0 else x['theo_b_revenue_eur_sum'], axis=1)
    df['simulated_hedge_volume'] = df.apply(lambda x: x['eur_volume_sum'] if x['profitable']==0 else 0, axis=1)



    df = pd.melt((
        df
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
              [Input(component_id='agg_preds', component_property='data')])

def graph_update(data):
    # bar chart
    validation = pd.read_json(data, orient='split')
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
              [Input(component_id='agg_preds', component_property='data')])
def graph_update(data):
    # bar chart
    validation =  pd.read_json(data, orient='split')
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



@app.callback(Output('agg_preds', 'data'),
Input(component_id="predictions",component_property="data"),
               Input(component_id='slider', component_property='value'),
              prevent_initial_call=True

               )
def update_predictions(json_data,slider_value):
    df = pd.read_json(json_data, orient='split')

    validation = prepare_df(slider_value, df)
    validation['value']=abs(validation['value'])
    return validation.to_json(date_format='iso', orient='split')


# hedge ratio bar chart
@app.callback(Output(component_id='hedge_bar_plot', component_property='figure'),
               Input('agg_preds', 'data'))
def graph_update(data):
    validation = pd.read_json(data, orient='split')
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
@app.callback(Output(component_id="pnl_across_thresholds",component_property="figure"),
              Input(component_id="slider",component_property="value"),
              Input(component_id="thresholds",component_property="data"))
def update_plot(value,json_data):
    stats_df = pd.read_json(json_data, orient='split')
    score_fig = px.line(stats_df, x="thresholds", y="total_pnl",text="total_pnl")
    score_fig.update_traces(textposition="bottom right",texttemplate="%{y:.2s}")
    score_fig.add_hline(y=abs(stats_df['real_pnl_eur'].max()), line_width=2, line_dash="dash", line_color="#969696",annotation_text="Real", annotation_position="bottom right")
    score_fig.add_hline(y=abs(stats_df['theo_b_pnl_eur'].max()), line_width=2, line_dash="dash", line_color="#3182bd",annotation_text="Theo B: {0}".format(abs(stats_df['theo_b_pnl_eur'].max())), annotation_position="bottom right")
    score_fig.add_vline(x=value, line_width=2, line_dash="dash", line_color="red", annotation_text="Score: {0}".format(value),
                        annotation_position="top right")

    return score_fig




#threshold chart
#threshold chart
@app.callback(Output(component_id="std_across_thresholds",component_property="figure"),
              Input(component_id="slider",component_property="value"),
              Input(component_id="thresholds",component_property="data"))
def update_plot(value,json_data):
    stats_df = pd.read_json(json_data, orient='split')
    risk_fig = px.line(stats_df, x="thresholds", y="stds",text="stds")
    risk_fig.update_traces(textposition="bottom right",texttemplate="%{y:.2s}")
    risk_fig.add_hline(y=stats_df['real_std'].max(), line_width=2, line_dash="dash", line_color="#969696",annotation_text="Real", annotation_position="bottom right")
    risk_fig.add_vline(x=value, line_width=2, line_dash="dash", line_color="red", annotation_text="Score",
                        annotation_position="bottom right")
    return risk_fig


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))


        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    uploaded_df_predicted = batch_predict(xgb_model, df)

    return html.Div([
    dag.AgGrid(
            id="grid-page-size",
            columnDefs= [{"field": i} for i in uploaded_df_predicted.columns.tolist()],
            rowData=uploaded_df_predicted.to_dict("records"),
            columnSize="sizeToFit",
            defaultColDef={"filter": True},
            dashGridOptions={"pagination": True, "paginationPageSizeSelector": False, "animateRows": False},
        ),

        html.Hr()
    ]), uploaded_df_predicted.to_json(date_format='iso', orient='split'), get_thresholds(uploaded_df_predicted).to_json(date_format='iso', orient='split')


@app.callback(Output('output-data-upload', 'children'),
              Output('predictions', 'data'),
              Output('thresholds', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              prevent_initial_call=True)

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children =parse_contents(list_of_contents[0], list_of_names[0], list_of_dates[0])
        return children


tabs = dbc.Tabs(
    [
        dbc.Tab(dcc.Graph(id='hedge_bar_plot'), label="Hedge comparison"),
        dbc.Tab(html.Div(id='output-data-upload'), label="Raw predictions")
    ]
)


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

        dbc.Col([

            tabs,
                 dcc.Store(id='predictions'),
                 dcc.Store(id="agg_preds"),
                 dcc.Store(id="thresholds")], width=10, lg=10, md=5, xs=12),
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
    dbc.Col([dbc.Tabs(
    [
        dbc.Tab(children=[ html.B("Potential PnL vs Actual across different thresholds"),
                                   html.Hr(),
                                   dcc.Graph(id="pnl_across_thresholds")], label="PnL"),
        dbc.Tab(children=[html.B("Potential PnL vs Actual across different thresholds"),
                                   html.Hr(),dcc.Graph(id="std_across_thresholds")], label="SD")
    ]
)
       ],width=10, lg=10, md=10, xs=12),
    ]),
        dbc.Row([
            dbc.Col([], width=2),
            dbc.Col([
html.Div(id="initial")
            ], width=10, lg=10, md=10, xs=12),
        ])
], fluid=True,style={"margin":10,"padding":10})

if __name__ == "__main__":
    app.run_server(debug=False, port=7777, host='0.0.0.0')

