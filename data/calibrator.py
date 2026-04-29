# Lobster provides a message file with every event and an orderbook file with a snapshot of the LOB at the moment of the message file
# We load side-by-side to claculate midprice and microprice

# We calibrate volatility  by computing the annualized standard deviation of log return of midprice over fixed intervals
# We calibrate temporary impact by integrating the volume at each price level until your trade of size X is filled, and regressing against various trade sizes gives eta
# We calibrate permanent impact by calculating either order flow imbalance or net signed volume over an interval and regression one against the change in mid-price.

import pandas as pd
import numpy as np

class LobsterCalibrator:
    def __init__(self, message_path: str, orderbook_path: str, levels: int = 50):
        self.message_path = message_path
        self.orderbook_path = orderbook_path
        self.levels = levels

    def load_data(self) -> pd.DataFrame:
        """Loads and aligns message and orderbook data."""
        msg_cols = ["Time", "Event", "OrderID", "Size", "Price", "Direction"]

        ob_cols = []
        for i in range(1, self.levels + 1):
            ob_cols.extend([f"Ask_Price_{i}", f"Ask_Size_{i}", f"Bid_Price_{i}", f"Bid_Size_{i}"])

        print("Loading message file...")
        messages = pd.read_csv(self.message_path, names=msg_cols)

        print("Loading orderbook file...")
        orderbook = pd.read_csv(self.orderbook_path, names=ob_cols) / 10000.0

        df = pd.concat([messages, orderbook], axis=1)

        df['Mid_Price'] = (df['Ask_Price_1'] + df['Bid_Price_1']) / 2.0
        df['Micro_Price'] = (df['Bid_Size_1']*df['Ask_Price_1']+df['Ask_Size_1']*df['Bid_Price_1']) / (df['Bid_Size_1'] + df['Ask_Size_1'])

        return df
    
    def estimate_volatility(self, df: pd.DataFrame, freq: str = '5min') -> float:
        """Calculate annualized volatility from mid-price log returns."""
        df['Time_Delta'] = pd.to_timedelta(df['Time'], unit='s')
        df.set_index('Time_Delta', inplace=True)

        resampled = df['Mid_Price'].resample(freq).last().dropna()
        log_returns = np.log(resampled/resampled.shift(1)).diff().dropna()

        # 252 trading days, 78 5-minute intervals per day
        annualized_vol = log_returns.std() * np.sqrt(252*(390 / int(freq.replace('min',''))))
        return annualized_vol
    
    def estimate_impact_parameters(self, df: pd.DataFrame):
        pass
    
    def estimate_heston_parameters(self, df: pd.DataFrame):
        pass