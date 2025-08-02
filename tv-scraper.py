import numpy as np
from tradingview_scraper.symbols.stream import RealTimeData
from tradingview_scraper.symbols.stream import Streamer


# Create an instance of the Streamer class
streamer = Streamer(
    export_result=True,
    export_type='json',
    websocket_jwt_token="Your-Tradingview-Websocket-JWT"
    )

data_generator = streamer.stream(
    exchange="NASDAQ",
    symbol="AAPL",
    timeframe="4h",
    numb_price_candles=1000,
    indicator_id="STD;RSI",
    indicator_version="31.0"
    )

#Testing git commit
#Testing push command