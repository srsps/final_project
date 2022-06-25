from dash import Dash, html

app = Dash()
app.layout = html.Div([
    html.Iframe(
        srcDoc='''
            <a class="twitter-timeline" data-theme="dark" href="https://twitter.com/economics">
                Tweets by Elon Musk
            </a> 
            <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
        ''',
        height=800,
        width=300
    )
])

if __name__ == '__main__':
    app.run_server()