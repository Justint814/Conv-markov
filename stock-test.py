import numpy as np
from polygonscraper import PolygonData

api_key = 'aSE_Q2Vmnh9BCPSumGUceCJphILlB3ag'
ticker = 'AAPL'
limit = 20000

min_data = PolygonData(api_key).retrieve_data(ticker, limit)


