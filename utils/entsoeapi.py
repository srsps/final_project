import requests
from pandas import read_csv, concat, to_datetime
from datetime import datetime, timedelta
from pandas import DataFrame
from pytz import timezone
from datetime import time


def get_energy_price():
    """
    This function reads the day-ahead market data with an api from Entsoe.eu.
    Between 0:00 and 12:00, the function is currently giving an error, because no day ahead data is published yet
    """
    #Get time for tomorrow
    now = datetime.now()+timedelta(days=1)
    mid_day = datetime.strftime(datetime.now()+timedelta(days=1), '%Y-%m-%d %H:00:00')
    mid_day = to_datetime(mid_day)

    #Day-ahead is published 'around' mid-day, will need to adjust var:mid-day to match this time later
    #If day ahead data is unavailable, the function crashes -> except -> return empty dict
    try:

        #Get start and end date for the day ahead price, in other words H=0 & H=24 of tomorrow
        end = datetime.strftime(now,'%Y%m%d2300')               #Required format: yyyyMMddHHmm
        start = datetime.strftime(now,'%Y%m%d0000')             #Required format: yyyyMMddHHmm

        #Save tomorrows date in datetime format as a different variable
        df_start = datetime.strftime(now,'%Y-%m-%d 00:00:00')

        #Get api data: Day-ahead prices in Finland
        payload = {'securityToken':'d13f79f0-6262-44ed-b2bd-16c2a21c3198','documentType':'A44',
                  'in_Domain':'10YPT-REN------W','out_Domain':'10YPT-REN------W','periodStart':start,'periodEnd':end}
        r = requests.get('https://transparency.entsoe.eu/api?',params=payload)

        #Data is read as text, because of the challenging format of the api return
        response = r.text

        #Clean up data

        response = response.split('<Point>')
        response.pop(0)
        response = [row.split('<position>') for row in response]
        df = DataFrame(response)
        df = df[1].str.split('<', expand=True)
        df['price'] = [row[2][13:] for row in df.iloc]

        #Temporary list of index to avoid api formatting problem
        df['list'] = [int(df[0][row])-1 for row in df.index]
        df['index'] = range(0,24)
        df = df.set_index(df['index'])
        #Create date and time columns, and combine into one datetime column
        df['date'] = to_datetime(df_start).date()
        df['time'] = [to_datetime(df['list'][row], format='%H').time() for row in df.index]
        df['datetime_start'] = [datetime.combine(df['date'][row], df['time'][row]) for row in df.index]
        df['datetime_end'] = df['datetime_start']+timedelta(hours=1)

        #Drop excess data
        df = df.drop([0,1,2,3,4,5,6,7,'date','time','list','index'], axis=1)

        #Turn df to dictionary - formatting for the output api
        result = df.to_dict()

        return result

    except:
        result = {'price': {1:'00.00'}, 'datetime_start': {1:'00:00:00'}, 'datetime_end': {1:'00:00:00'}}
        return result