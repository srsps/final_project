# Authors
`Sreejith P Sajeev-101483, Tobbias Steggemann,-101485 Albert Rehnberg-102018`

# Setting up the environment
$ cat python runtime - 3.7.13
>macOS/Linux

You may need to run sudo apt-get install python3-venv first\
$python3 -m venv .venv
>Windows

You can also use py -3 -m venv .venv\
$python -m venv .venv\
$ pip install -r requirements.txt

# Running the app
$python run  app.py\
\
Dashboard deployed at `localhost:8050`

# Dashboard Features

The dashboard has 5  pages `About, Weather Monitoring, Real time Data, Forecasting~Production, Forecasting~Pricing`.

All ML and live tracking of energy data excuted using REST apis. More information on REST api service at: https://docs.github.com/en/rest\

The Running of APIs are in Google Cloud. The deployement is automated using cloud build.\

For local running of APIs refer to ./energy-service-api-py






