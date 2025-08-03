import numpy as np
import requests
import datetime

'''
if __name__ == "__main__":
    api_key = 'aSE_Q2Vmnh9BCPSumGUceCJphILlB3ag'
    ticker = 'AAPL'
    limit = 20000
    today = datetime.date.today()
    from_date = today - datetime.timedelta(days=30)
'''
class PolygonData:

    #Dictionary to store type of data retrieved from json response of the polygon API request.
    polydat = {"close": "c", "open": "p", "low": "l", "high": "h", "time": "t", "volume": "v", "vwap": "vw"}

    def __init__(self, API_key: str, days=600):
        self.api_key = API_key
        self.today = datetime.date.today()
        self.from_date = self.today - datetime.timedelta(days=days)
    
    def retrieve_data(self, ticker: str, num_points: int, timeframe='minute', data_type="close"):
        url: str = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timeframe}/{self.from_date}/{self.today}?limit={num_points}&apiKey={self.api_key}"
        response = requests.get(url)
        data = response.json()
        target_param = self.polydat[data_type]

        output = [item[target_param] for item in data.get('results', [])]

        return output
    
