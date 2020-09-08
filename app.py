import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from chat_res import *

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions = True)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Posterio Toys", className="display-4"),
        html.Hr(),
        html.P(
            "Some funny toys play around", className="lead"
        ),
        # Sidebar Pages
        dbc.Nav(
            [
                dbc.NavLink("GPT_ChatBot_CN", href="/page-1", id="page-1-link"),
                dbc.NavLink("GPT_ChatBot_EN", href="/page-2", id="page-2-link")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), 
                        sidebar, 
                        content,
                        ])


#ChatBot Page

chatbot_page = html.Div(
    [
        dbc.Input(id="cn_chatbot_input", placeholder="Type something...", type="text"),
        html.Br(),
        dbc.ButtonGroup(
                    [dbc.Button("发送信息",id='cn-submit-button', color="primary", className="mr-1",n_clicks=0), 
                    dbc.Button("消除记忆",id='cl-submit-button', color="info", className="mr-1",n_clicks=0),
                    dbc.Collapse(dbc.Card(dbc.CardBody("记忆已清除")),id="collapse",),
                    ]
                    ),
        
        html.Br(),
        html.Br(),
        html.P(id="cn_chatbot_output"),
        #html.Br(),
        
        #dbc.Output(id="cn_chatbot_output", placeholder="I'm happy to chat with you...", type="text")
    ]
)
#chat input button logic
@app.callback(Output("cn_chatbot_output", "children"), 
                  [Input('cn-submit-button', 'n_clicks')],
                  [State('cn_chatbot_input', 'value')],)
def output_text(n_clicks,input1):
    #change input here
    #print(type(input1))
    print(input1)
    if input1 is None:
        input1 = "很高兴跟你聊天, 在上面的聊天框跟我讲话吧"
        #print('good')
    elif input1 is not None:
        input1 = chat_response(input1)
        #print('bad')
    return 'Posterio:     {}'.format(input1)

#chat output button
''' 
@app.callback([Input('cl-submit-button', 'n_clicks')])
def output_text(n_clicks):
    global history
    respon = 'Posterio:  很高兴刚才跟你聊了 {} 句，现在我的大脑已经清空了'.format(len(history))
    history = []
    return respon
'''
@app.callback(
    Output("collapse", "is_open"),
    [Input("cl-submit-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output("page-{}-link".format(i), "active") for i in range(1, 3)],
    [Input("url", "pathname")],)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False
    return [pathname == "/page-{}".format(i) for i in range(1, 3)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]:
        return chatbot_page
    elif pathname in ["/", "/page-2"]:
        return html.P("Working")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
        ]
    )


if __name__ == "__main__":
    #app.run_server(debug=True,host='0.0.0.0',port=8888)
    app.run_server(host='0.0.0.0',port=8888)