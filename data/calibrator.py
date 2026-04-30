# We calibrate temporary impact by integrating the volume at each price level until your trade of size X is filled, and regressing against various trade sizes gives eta
# We calibrate permanent impact by calculating either order flow imbalance or net signed volume over an interval and regression one against the change in mid-price.

import os
import zipfile
import pandas as pd
import numpy as np

class LobsterCalibrator:
    def __init__(self, message_path: str, orderbook_path: str = None, levels: int = 50):
        self.message_path = message_path
        self.orderbook_path = orderbook_path
        self.levels = levels

    @classmethod
    def from_zip(cls, archive_path: str):
        """Create a calibrator from a LOBSTER zip archive."""
        if not archive_path.lower().endswith('.zip'):
            raise ValueError("Archive path must be a .zip file")

        with zipfile.ZipFile(archive_path, 'r') as archive:
            names = archive.namelist()
            orderbook_file = next((n for n in names if '_orderbook_' in n), None)
            message_file = next((n for n in names if '_message_' in n), None)

            if orderbook_file is None or message_file is None:
                raise ValueError("Zip archive does not contain expected LOBSTER orderbook/message files")
            
            # Extract levels from filename (e.g., "_5" from "orderbook_5.csv")
            import re
            level_match = re.search(r'_(\d+)\.csv', orderbook_file)
            if level_match:
                levels = int(level_match.group(1))
            else:
                # Fallback: count columns in first line
                with archive.open(orderbook_file) as file:
                    first_line = file.readline().decode('utf-8').strip()
                    levels = len(first_line.split(',')) // 4  # 4 columns per level

        return cls(message_path=archive_path, orderbook_path=archive_path, levels=levels)

    def _open_csv(self, path: str, names):
        if path.lower().endswith('.zip'):
            with zipfile.ZipFile(path, 'r') as archive:
                file_name = next((n for n in archive.namelist() if any(tag in n for tag in ['_message_', '_orderbook_'])), None)
                if file_name is None:
                    raise ValueError(f"No matching CSV found in archive {path}")
                with archive.open(file_name) as file:
                    return pd.read_csv(file, names=names)
        return pd.read_csv(path, names=names)

    def load_data(self) -> pd.DataFrame:
        """Loads and aligns message and orderbook data."""
        msg_cols = ["Time", "Event", "OrderID", "Size", "Price", "Direction"]

        ob_cols = []
        for i in range(1, self.levels + 1):
            ob_cols.extend([f"Ask_Price_{i}", f"Ask_Size_{i}", f"Bid_Price_{i}", f"Bid_Size_{i}"])

        print("Loading message file...")
        if self.message_path.lower().endswith('.zip'):
            with zipfile.ZipFile(self.message_path, 'r') as archive:
                message_file = next((n for n in archive.namelist() if '_message_' in n), None)
                if message_file is None:
                    raise ValueError("Zip archive does not contain a message file")
                with archive.open(message_file) as file:
                    # Read as text to avoid pandas buffer issues
                    content = file.read().decode('utf-8')
                    from io import StringIO
                    messages = pd.read_csv(StringIO(content), names=msg_cols)

                print("Loading orderbook file...")
                orderbook_file = next((n for n in archive.namelist() if '_orderbook_' in n), None)
                if orderbook_file is None:
                    raise ValueError("Zip archive does not contain an orderbook file")
                with archive.open(orderbook_file) as file:
                    # Read as text to avoid pandas buffer issues
                    content = file.read().decode('utf-8')
                    from io import StringIO
                    orderbook = pd.read_csv(StringIO(content), names=ob_cols, dtype=float) / 10000.0
        else:
            messages = pd.read_csv(self.message_path, names=msg_cols)
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
        """
        Estimate temporary (eta) and permanent (gamma) impact parameters.
        
        Temporary impact: Simulate trades of different sizes and measure price impact
        Permanent impact: Regress mid-price changes against order flow imbalance
        """
        # Filter for market orders (Event=4 or 5)
        market_orders = df[df['Event'].isin([4, 5])].copy()
        
        if len(market_orders) == 0:
            print("Warning: No market orders found for impact estimation")
            return {'eta': None, 'gamma': None}
        
        # Temporary impact estimation (eta)
        eta = self._estimate_temporary_impact(df, market_orders)
        
        # Permanent impact estimation (gamma) 
        gamma = self._estimate_permanent_impact(df, market_orders)
        
        return {'eta': eta, 'gamma': gamma}
    
    def _estimate_temporary_impact(self, df: pd.DataFrame, market_orders: pd.DataFrame):
        """Estimate temporary impact parameter eta by simulating trade execution."""
        trade_sizes = np.logspace(2, 6, 20)  # Trade sizes from 100 to 1M shares
        impacts = []
        
        for X in trade_sizes:
            # Simulate execution for trade size X
            impact = self._simulate_trade_execution(df, X)
            if impact is not None:
                impacts.append((X, impact))
        
        if len(impacts) < 5:
            print("Warning: Insufficient data for temporary impact estimation")
            return None
            
        # Regress log(impact) against log(X) to get eta
        X_vals, impact_vals = zip(*impacts)
        X_log = np.log(X_vals)
        impact_log = np.log(impact_vals)
        
        # Linear regression: log(impact) = log(eta) + eta * log(X)
        # Actually: impact = eta * sqrt(X), so log(impact) = log(eta) + 0.5 * log(X)
        slope, intercept = np.polyfit(X_log, impact_log, 1)
        
        # eta should be around slope/2 if impact ~ sqrt(X), but we'll use the slope directly
        eta = np.exp(intercept)  # eta = exp(intercept) where intercept = log(eta)
        
        return eta
    
    def _simulate_trade_execution(self, df: pd.DataFrame, trade_size: float):
        """Simulate execution of a trade of size X and return price impact."""
        # Find a random starting point with sufficient liquidity
        valid_starts = []
        for idx in range(len(df)):
            bid_size = df.iloc[idx]['Bid_Size_1']
            ask_size = df.iloc[idx]['Ask_Size_1']
            if bid_size > trade_size * 0.1 and ask_size > trade_size * 0.1:  # At least 10% of trade size
                valid_starts.append(idx)
        
        if len(valid_starts) == 0:
            return None
            
        start_idx = np.random.choice(valid_starts)
        start_price = df.iloc[start_idx]['Mid_Price']
        
        # Simulate market buy order (simplified - just use best ask)
        # In reality, would need to walk the order book
        execution_price = df.iloc[start_idx]['Ask_Price_1']
        
        # Price impact = execution price - mid price
        impact = execution_price - start_price
        
        return max(impact, 0)  # Only positive impacts
    
    def _estimate_permanent_impact(self, df: pd.DataFrame, market_orders: pd.DataFrame):
        """Estimate permanent impact parameter gamma using order flow imbalance."""
        # Create time delta column if it doesn't exist
        if 'Time_Delta' not in df.columns:
            df = df.copy()
            df['Time_Delta'] = pd.to_timedelta(df['Time'], unit='s')
        
        # Resample to 5-minute intervals
        df_resampled = df.set_index('Time_Delta').resample('5min').agg({
            'Mid_Price': 'last',
            'Size': 'sum',
            'Direction': lambda x: (x == 1).sum() - (x == -1).sum()  # Net buy orders
        }).dropna()
        
        if len(df_resampled) < 10:
            print("Warning: Insufficient data for permanent impact estimation")
            return None
        
        # Calculate mid-price returns
        df_resampled['Mid_Return'] = df_resampled['Mid_Price'].pct_change()
        
        # Order flow imbalance (simplified)
        df_resampled['Order_Flow'] = df_resampled['Direction'] * df_resampled['Size']
        
        # Regress mid-price returns against lagged order flow
        valid_data = df_resampled.dropna()
        if len(valid_data) < 5:
            return None
            
        # Use lagged order flow to predict returns
        lagged_flow = valid_data['Order_Flow'].shift(1).dropna()
        returns = valid_data['Mid_Return'].iloc[1:]
        
        if len(lagged_flow) != len(returns):
            min_len = min(len(lagged_flow), len(returns))
            lagged_flow = lagged_flow.iloc[:min_len]
            returns = returns.iloc[:min_len]
        
        if len(lagged_flow) < 3:
            return None
            
        slope, intercept = np.polyfit(lagged_flow, returns, 1)
        
        gamma = abs(slope)  # Permanent impact coefficient
        
        return gamma
    
    def estimate_heston_parameters(self, df: pd.DataFrame):
        """
        Estimate Heston model parameters from high-frequency data.
        
        Returns dict with parameters: v0, mu, theta, omega, xi, rho
        """
        # Get volatility time series
        vol_series = self._estimate_volatility_series(df)
        
        if vol_series is None or len(vol_series) < 50:
            print("Warning: Insufficient volatility data for Heston estimation")
            return None
        
        # Estimate parameters
        params = {}
        
        # v0: current variance level
        params['v0'] = vol_series.iloc[-1] ** 2
        
        # mu: drift (set to 0 for risk-neutral)
        params['mu'] = 0.0
        
        # theta: long-term variance (mean of variance series)
        params['theta'] = vol_series.var()
        
        # omega (kappa): mean reversion speed
        # xi: volatility of volatility  
        # rho: correlation between price and volatility
        kappa, xi, rho = self._estimate_volatility_parameters(df, vol_series)
        params['omega'] = kappa
        params['xi'] = xi
        params['rho'] = rho
        
        return params
    
    def _estimate_volatility_series(self, df: pd.DataFrame, window: str = '5min'):
        """Estimate time-varying volatility series."""
        try:
            df_copy = df.copy()
            if 'Time_Delta' not in df_copy.columns:
                df_copy['Time_Delta'] = pd.to_timedelta(df_copy['Time'], unit='s')
            df_copy.set_index('Time_Delta', inplace=True)
            
            # Resample mid-price to get returns
            resampled = df_copy['Mid_Price'].resample(window).last().dropna()
            if len(resampled) < 2:
                return None
                
            returns = np.log(resampled / resampled.shift(1)).dropna()
            if len(returns) < 10:
                return None
            
            # Rolling volatility (annualized)
            vol_series = returns.rolling(window=20, min_periods=5).std() * np.sqrt(252 * (390 / 5))
            
            return vol_series.dropna()
        except Exception as e:
            print(f"Error estimating volatility series: {e}")
            return None
    
    def _estimate_volatility_parameters(self, df: pd.DataFrame, vol_series):
        """Estimate kappa, xi, and rho for Heston model."""
        try:
            # Mean reversion speed (kappa) - how quickly vol reverts to mean
            # Estimate from autocorrelation of volatility
            vol_autocorr = vol_series.autocorr(lag=1) if len(vol_series) > 1 else 0.5
            kappa = max(0.01, -np.log(max(0.01, abs(vol_autocorr))) / 5)  # 5-minute intervals, minimum 0.01
            
            # Volatility of volatility (xi) - std of vol changes
            vol_changes = vol_series.diff().dropna()
            if len(vol_changes) > 0:
                xi = max(0.01, vol_changes.std() * np.sqrt(252 * (390 / 5)))  # Annualized, minimum 0.01
            else:
                xi = 0.2  # Default value
            
            # Correlation between price and volatility (rho)
            # Estimate from correlation of returns and volatility changes
            try:
                df_copy = df.copy()
                if 'Time_Delta' not in df_copy.columns:
                    df_copy['Time_Delta'] = pd.to_timedelta(df_copy['Time'], unit='s')
                df_copy.set_index('Time_Delta', inplace=True)
                
                returns = np.log(df_copy['Mid_Price'].resample('5min').last().dropna().pct_change().dropna())
                vol_changes_5min = vol_series.resample('5min').last().diff().dropna()
                
                # Align the series
                common_index = returns.index.intersection(vol_changes_5min.index)
                if len(common_index) > 10:
                    rho = returns.loc[common_index].corr(vol_changes_5min.loc[common_index])
                    rho = np.clip(rho, -0.9, 0.9)  # Reasonable bounds
                else:
                    rho = -0.5  # Default negative correlation
            except:
                rho = -0.5  # Default value
                
        except Exception as e:
            print(f"Error in volatility parameter estimation: {e}")
            kappa, xi, rho = 2.0, 0.3, -0.5  # Default values
            
        return kappa, xi, rho