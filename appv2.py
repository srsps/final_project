import dash_bootstrap_components as dbc
from dash import dcc
from dash.dependencies import State
import plotly.graph_objects as go
import dash
from dash import html
from dash.dependencies import Output, Input
from utils.Trends_portugal import trending
import utils.entsoeapi as apii
import utils.main as tt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

#=================Cards===========================================
cardp= dbc.Card(
    [
        dbc.CardBody(
            [
                # html.H6("Strategies", className="card-subtitle"),
                dbc.ListGroup(
                    [html.H5("Positive", className="pos"),
                     dbc.ListGroupItem(html.Span(id='pos', children=''))
                     ])
            ]
        )

    ],
    color="Green",
    inverse=False,
    outline=False,
    style={"width": "10rem",
           "height": "6rem"
           })
cardp1= dbc.Card(
    [
        dbc.CardBody(
            [
                # html.H6("Strategies", className="card-subtitle"),
                dbc.ListGroup(
                    [html.H5("Positive-1", className="pos1"),
                     dbc.ListGroupItem(html.Span(id='pos1', children=''))
                     ])
            ]
        )

    ],
    color="Green",
    inverse=False,
    outline=False,
    style={"width": "10rem",
           "height": "6rem"
           })
cardn= dbc.Card(
    [
        dbc.CardBody(
            [
                # html.H6("Strategies", className="card-subtitle"),
                dbc.ListGroup(
                    [html.H5("Negetive", className="neg"),
                     dbc.ListGroupItem(html.Span(id='neg', children=''))
                     ])
            ]
        )

    ],
    color="Red",
    inverse=False,
    outline=False,
    style={"width": "10rem",
           "height": "6rem"
           })
cardn1= dbc.Card(
    [
        dbc.CardBody(
            [
                # html.H6("Strategies", className="card-subtitle"),
                dbc.ListGroup(
                    [html.H5("Negetive-1", className="neg1"),
                     dbc.ListGroupItem(html.Span(id='neg1', children=''))
                     ])
            ]
        )

    ],
    color="Red",
    inverse=False,
    outline=False,
    style={"width": "10rem",
           "height": "6rem"
           })

cardnu= dbc.Card(
    [
        dbc.CardBody(
            [
                # html.H6("Strategies", className="card-subtitle"),
                dbc.ListGroup(
                    [html.H5("Neutral", className="neu"),
                     dbc.ListGroupItem(html.Span(id='neu', children=''))
                     ])
            ]
        )

    ],
    color="light",
    inverse=False,
    outline=False,
    style={"width": "10rem",
           "height": "6rem"
           })
cardnu1= dbc.Card(
    [
        dbc.CardBody(
            [
                # html.H6("Strategies", className="card-subtitle"),
                dbc.ListGroup(
                    [html.H5("Neutral", className="neu1"),
                     dbc.ListGroupItem(html.Span(id='neu1', children=''))
                     ])
            ]
        )

    ],
    color="light",
    inverse=False,
    outline=False,
    style={"width": "10rem",
           "height": "6rem"
           })


cardgb= dbc.Card(
    [
        dbc.CardBody(
            [
                # html.H6("Strategies", className="card-subtitle"),
                dbc.ListGroup(
                    [html.H5("Gradient Boost", className="neg1"),
                     dbc.ListGroupItem(html.Span(id='gb', children=''))
                     ])
            ]
        )

    ],
    color="light",
    inverse=False,
    outline=False,
    style={"width": "10rem",
           "height": "10rem"
           })

cardnn= dbc.Card(
    [
        dbc.CardBody(
            [
                # html.H6("Strategies", className="card-subtitle"),
                dbc.ListGroup(
                    [html.H5("Neural Net", className="nn"),
                     dbc.ListGroupItem(html.Span(id='nn', children=''))
                     ])
            ]
        )

    ],
    color="light",
    inverse=False,
    outline=False,
    style={"width": "10rem",
           "height": "10rem"
           })
cardrf= dbc.Card(
    [
        dbc.CardBody(
            [
                # html.H6("Strategies", className="card-subtitle"),
                dbc.ListGroup(
                    [html.H5("Random Forest", className="neu1"),
                     dbc.ListGroupItem(html.Span(id='rf', children=''))
                     ])
            ]
        )

    ],
    color="light",
    inverse=False,
    outline=False,
    style={"width": "10rem",
           "height": "10rem"
           })



#==================================================================
Y_train = pd.read_csv("./dat/train.csv")
X_train = pd.read_csv("./dat/xtrain.csv")

# X_train['Date']=pd.to_datetime(X_train['Date'])
# X_train['Hour'] = X_train['Date'].dt.hour
# X_train['Day'] = X_train['Date'].dt.day
# X_train['Month'] = X_train['Date'].dt.month
# X_train['Price-1']=Y_train['Price'].shift(1)
# X_train.drop(columns=['Date'],inplace=True)
# X_train.dropna(0)
# X_train.to_csv('xtrain.csv')

# clf = tree.DecisionTreeRegressor()
# clf = clf.fit(X_train, Y_train)
# shap_values = shap.TreeExplainer(clf).shap_values(X_train)
# plt.figure(111)
# shap.summary_plot(shap_values, X_train, plot_type='bar', show=False)
# plt.savefig('Features.png')
process = RobustScaler()
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)


def neural(a, b, c, d, e, f):
    model1 = MLPRegressor(hidden_layer_sizes=(10, 10, 10, 10))
    model1.fit(x_train, y_train)
    value = model1.predict([[a, b, c, d, e, f], ])
    return value


def rf(a, b, c, d, e, f):
    model2 = RandomForestRegressor(bootstrap=True, min_samples_leaf=1,
                                   n_estimators=20, min_samples_split=15, max_features='sqrt', max_depth=20)
    model2.fit(x_train, y_train)
    value = model2.predict([[a, b, c, d, e, f], ])
    return value


def gb(a, b, c, d, e, f):
    model3 = GradientBoostingRegressor()
    model3.fit(x_train, y_train)
    value = model3.predict([[a, b, c, d, e, f], ])
    return value

# ======================================================================
# API ENTSOE Pricing Day Ahead
# =============================================================================

price_dat = apii.get_energy_price()
price = [price_dat['price'][key] for key in price_dat['price']]
f_price = list(map(float, price))
f_hour = list(map(int, price_dat['price'].keys()))
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=f_hour, y=f_price, name="Day Ahead Price - ENTSOE",
                               line_shape='hv'))
fig_price.update_layout(title='Day ahead pricing',
                        xaxis_title='Hours',
                        yaxis_title='EUR/Mwh')

# =================================Data Prepp===================================================#
now_pt = datetime.now()
a = now_pt.hour
b = now_pt.day
c = now_pt.month
# arr_set = {0:'Portugal',1:'USA',2:'Algeria',3:'Russia',4: 'UAE', 5:'Saudi',6: 'Oil', 7:'LNG',8:'Gas'}
# full_stc=[0]*len(arr_set)
# tweets=[0]*len(arr_set)
# for i in range(len(arr_set)):
#     full_stc[i] = tt.per(arr_set[i])
#     tweets[i] = tt.main(arr_set[i])
# def button_press1(id):
#     d=full_stc[i][4]
#     e=full_stc[i][0]
#     f=full_stc[i][2]
#     x=neural(a,b,c,d,e,f)
#     y=gb(a,b,c,d,e,f)
#     z=rf(a, b, c, d, e, f)
#     dev={0:x,
#          1:y,
#          2:z
#     }
#     return dev









# ==========================Dash app===========================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
app.title = "PEGE Dash"
server = app.server
navbar = dbc.NavbarSimple(
    children=[
        dbc.Button("Sidebar", outline=True, color="secondary", className="mr-1", id="btn_sidebar"),
        dbc.NavItem(dbc.NavLink("Home", href="/")),

    ],
    brand="PEGE Final Dash",
    brand_href="#",
    color='#042032',
    dark=True,
    fluid=True,
)
# footer = html.Footer(children=html.P("A product by FiInGe")
#
#                      )

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 62.5,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#F1FFFE ",
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 62.5,
    "left": "-16rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#F1FFFE",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [html.Img(src='./assets/IST_Logo.png', style={'height': '30%', 'width': '90%'}),
     html.Br(),
     html.P("Intelligent Market Watch ", className="title"),
     html.Hr(),
     html.P(
         "Portugese Energy Markets", className="lead"
     ),
     dbc.Nav(
         [
             dbc.NavLink("Home", href="/", active="exact"),
             dbc.NavLink("Production Analysis",href="/prod-dat",active='exact'),
             dbc.NavLink("Tweets Live Feed", href="/page-2", active="exact"),
             dbc.NavLink("Sentiment Analyser", href="/page-3", active="exact"),
             dbc.NavLink(dbc.DropdownMenu([

                 dbc.DropdownMenuItem("Pricing Day Ahead", href="/page-prod", active="exact"),
                 dbc.DropdownMenuItem("Predicted Dev", href="/page-price", active="exact")],
                 label=' Forecasts ', color="primary", className="mr-1", id="btn_sidebar1")),
         ],
         vertical=True,
         pills=True,
     ),
     ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div(

    id="page-content",
    style=CONTENT_STYLE)

app.layout = html.Div(
    [
        dcc.Store(id='side_click'),
        dcc.Location(id="url"),
        navbar,
        sidebar,
        content,
        # footer
    ],
)


# ========================================================================================
# Sidebar Hide call backs
# ========================================================================================
@app.callback(
    [
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
    ],

    [Input("btn_sidebar", "n_clicks")],
    [
        State("side_click", "data"),
    ]
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick


# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]


# =======================================================================
# Page content rendering and styling
# =======================================================================
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]:
        return [
            html.H3('Project in Energy Engineering and Management 2022'),
            html.Div(
                dbc.Row([
                    html.H6('Created By : Sreejith P Sajeev IST1101483 - sreejith.sajeev@tecnico.ulisboa.pt '),
                    html.Hr(),
                ])),
            html.Div(
                html.P(
                    'The market watches which forecasts future economic trends are quite complex. For energy\
markets its even more. Due to market price uncertainty and volatility, electricity sales companies today are facing greater risks in regard to the day-ahead market and the real-time market.\
Portuguese energy sector is quite distributed, in fact there is a good percentage of the total production coming from renewable resources such as wind,solar etc. With ever increasing rise of these\
resources, the generation is becoming more distributed and disorganized'),
            ),
            html.Br(),
            html.P('The need for a common market place where utility providers can see the economic trends of the\
generation and consumers can access the real time optimum energy pricing. Since the economics\
of energy markets are very dependant on world events, there is a need for event/sentiment analysis\
of the masses to accurately predict the future pricing.'

            ),
            html.Div(
                dbc.Row([
                    html.H6('Solution')])),
            html.P('The forecasting of pricing requires strenuous and complex machine learning models. This research\
will be focused on creating a machine learning paradigm modelled with historical data and statistical methods. This will be executed along with the visual representation of real time data of\
energy output and real time pricing using python dashboard.\
The project will deploy a local server based dash with multiple features, mainly;\
1. Real time energy data representation with map elements\
2. Real time energy pricing & forecasting\
3. Sentiment analysis based price forecasting\
The model will be tested and validated using historical data sets and the error rate of the model\
will be determined. The reinforcements to improve the model will also be discussed.'),


        ]
    elif pathname == "/page-2":
        return [
            html.H5("Live Tweets"),
            html.Hr(),
            html.Div([
            dcc.Dropdown(
                id='select',
                options=[
                    {'label': 'Portugal', 'value': 0},
                    {'label': 'USA', 'value': 1},
                    {'label': 'Algeria', 'value': 2},
                    {'label': 'Russia', 'value': 3},
                    {'label': 'UAE', 'value': 4},
                    {'label': 'Saudi', 'value': 5},
                    {'label': 'Oil', 'value': 6},
                    {'label': 'LNG', 'value': 7},
                 {'label': 'Gas', 'value': 8} ],

            )]),

            html.Br(),
            html.H5("Tweet Dict Sample "),
            html.Hr(),
            dcc.Textarea(
                id='textarea1',
                style={'width': '100%', 'height': 500},
            ),
            html.Br(),
            html.Iframe(
                srcDoc='''
                    <a class="twitter-timeline" data-theme="dark" href="https://twitter.com/economics">
                        Tweets by Bloomberg
                    </a>
                    <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                ''',
                height=800,
                width='75%'
            ),
            html.Br(),
            html.H5('Globel Trends'),
            dcc.Textarea(
                id='textarea2',
                style={'width': '100%', 'height': 500},
                value=str(trending())
            ),
        ]

    elif pathname == "/prod-dat":
        return [
            html.Iframe(id='mapper', src='https://app.electricitymap.org/zone/PT?wind=true', width='100%',
                        height='600')

        ]


    elif pathname == "/page-3":
        return [
            html.H5("Sentiment Analyzer"),
            html.Hr(),
            html.Div([
                dcc.Dropdown(
                    id='select2',
                    options=[
                        {'label': 'Portugal', 'value': 0},
                        {'label': 'USA', 'value': 1},
                        {'label': 'Algeria', 'value': 2},
                        {'label': 'Russia', 'value': 3},
                        {'label': 'UAE', 'value': 4},
                        {'label': 'Saudi', 'value': 5},
                        {'label': 'Oil', 'value': 6},
                        {'label': 'LNG', 'value': 7},
                        {'label': 'Gas', 'value': 8}],

                ),
            ]),
            html.Br(),
            html.H5("Text Blob analyzer"),
            html.Div([dbc.Row([
                dbc.Col(cardp, width=3),
                dbc.Col(cardn, width=3),
                dbc.Col(cardnu, width=3),

            ])

            ]),
            html.Br(),
            html.H5("NTLK analyzer"),
            html.Br(),
            html.Div([dbc.Row([
                dbc.Col(cardp1, width=3),
                dbc.Col(cardn1, width=3),
                dbc.Col(cardnu1, width=3),

            ])

            ])



        ]


    elif pathname=="/page-price":
        return[
            html.Div([
                html.H5("Deviation for each Subject Instance: Next Hour Price"),
                html.Hr(),
                dcc.Dropdown(
                    id='select3',
                    options=[
                        {'label': 'Portugal', 'value': 0},
                        {'label': 'USA', 'value': 1},
                        {'label': 'Algeria', 'value': 2},
                        {'label': 'Russia', 'value': 3},
                        {'label': 'UAE', 'value': 4},
                        {'label': 'Saudi', 'value': 5},
                        {'label': 'Oil', 'value': 6},
                        {'label': 'LNG', 'value': 7},
                        {'label': 'Gas', 'value': 8}],

                ),
            ]),
            html.Br(),
            html.Hr(),
            html.Div([dbc.Row([
                dbc.Col(cardgb, width=3),
                dbc.Col(cardnn, width=3),
                dbc.Col(cardrf, width=3),

            ])

            ])

        ]


    elif pathname == "/page-prod":
        return [
            html.H3('Day Ahead Market Price Forecasting'),
            html.Hr(),
            html.Div(
                dcc.Graph(id='pricing', figure=fig_price)
            )

        ]

    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


@app.callback(Output('textarea1', 'value'),
              Input('select', 'value'))
def render_figure_png(value):
    arr_set = {0: 'Portugal', 1: 'USA', 2: 'Algeria', 3: 'Russia', 4: 'UAE', 5: 'Saudi', 6: 'Oil', 7: 'LNG', 8: 'Gas'}
    tweets = tt.main(arr_set[value])
    sete=str(tweets)
    return sete

@app.callback(
  [  Output("pos", "children"),
     Output("neg","children"),
     Output("neu","children"),
     Output("pos1","children"),
    Output("neg1","children"),
    Output("neu1","children"),
     ],
     [Input("select2", "value")]
)

def on_button_click2(n):
    arr_set = {0: 'Portugal', 1: 'USA', 2: 'Algeria', 3: 'Russia', 4: 'UAE', 5: 'Saudi', 6: 'Oil', 7: 'LNG', 8: 'Gas'}
    full_stc = tt.per(arr_set[n])
    return round(full_stc[0],2),round(full_stc[2],2),round(full_stc[4],2),round(full_stc[1],2),round(full_stc[3],2),round(full_stc[5],2)
@app.callback(
  [  Output("gb", "children"),
     Output("nn","children"),
     Output("rf","children"),
     ],
     [Input("select3", "value")]
)

def on_button_click3(n):
    arr_set = {0: 'Portugal', 1: 'USA', 2: 'Algeria', 3: 'Russia', 4: 'UAE', 5: 'Saudi', 6: 'Oil', 7: 'LNG', 8: 'Gas'}
    full_stc = tt.per(arr_set[n])
    d = full_stc[4]
    e = full_stc[0]
    f = full_stc[2]
    x = neural(a, b, c, d, e, f)
    y = gb(a, b, c, d, e, f)
    z = rf(a, b, c, d, e, f)
    return y,x,z



if __name__ == "__main__":
    app.run_server(debug=False, port=9050)
