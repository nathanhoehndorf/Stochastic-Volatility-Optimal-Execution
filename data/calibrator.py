# calibrator.py

import os
import re
import zipfile
import pandas as pd
import numpy as np


class LobsterCalibrator:
    """
    Calibrator for LOBSTER message/orderbook data.

    Important convention used by the rest of this project:

        T = 1 means one trading day.

    Therefore:
        sigma is daily volatility.
        Heston variance parameters are in daily simulation units.
        theta = mean reversion speed / kappa.
        omega = long-run variance.
    """

    PRICE_SCALE = 10000.0
    TRADING_MINUTES_PER_DAY = 390

    def __init__(self, message_path: str, orderbook_path: str = None, levels: int = 50):
        self.message_path = message_path
        self.orderbook_path = orderbook_path
        self.levels = levels

    # ------------------------------------------------------------------
    # Dataset constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_zip(cls, archive_path: str):
        """Create a calibrator from a LOBSTER zip archive."""
        if not archive_path.lower().endswith(".zip"):
            raise ValueError("Archive path must be a .zip file")

        with zipfile.ZipFile(archive_path, "r") as archive:
            names = archive.namelist()

            orderbook_file = next((n for n in names if "_orderbook_" in n), None)
            message_file = next((n for n in names if "_message_" in n), None)

            if orderbook_file is None or message_file is None:
                raise ValueError(
                    "Zip archive does not contain expected LOBSTER orderbook/message files"
                )

            level_match = re.search(r"_(\d+)\.csv$", orderbook_file)
            if level_match:
                levels = int(level_match.group(1))
            else:
                with archive.open(orderbook_file) as file:
                    first_line = file.readline().decode("utf-8").strip()
                    levels = max(1, len(first_line.split(",")) // 4)

        return cls(
            message_path=archive_path,
            orderbook_path=archive_path,
            levels=levels,
        )

    @classmethod
    def from_directory(cls, directory_path: str):
        """Create a calibrator from a directory containing LOBSTER CSVs."""
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory path not found: {directory_path}")

        entries = os.listdir(directory_path)

        message_file = next(
            (f for f in entries if "_message_" in f and f.lower().endswith(".csv")),
            None,
        )
        orderbook_file = next(
            (f for f in entries if "_orderbook_" in f and f.lower().endswith(".csv")),
            None,
        )

        if message_file is None or orderbook_file is None:
            raise ValueError(
                "Directory does not contain expected LOBSTER message and orderbook CSV files"
            )

        message_path = os.path.join(directory_path, message_file)
        orderbook_path = os.path.join(directory_path, orderbook_file)

        level_match = re.search(r"_(\d+)\.csv$", orderbook_file)
        if level_match:
            levels = int(level_match.group(1))
        else:
            with open(orderbook_path, "r") as file:
                first_line = file.readline().strip()
            levels = max(1, len(first_line.split(",")) // 4)

        return cls(
            message_path=message_path,
            orderbook_path=orderbook_path,
            levels=levels,
        )

    @classmethod
    def from_dataset(cls, dataset_path: str):
        """Create a calibrator from either a zip archive or a directory."""
        if dataset_path.lower().endswith(".zip"):
            return cls.from_zip(dataset_path)

        if os.path.isdir(dataset_path):
            return cls.from_directory(dataset_path)

        raise ValueError(
            "Dataset path must be a .zip archive or a directory containing LOBSTER files"
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load and align LOBSTER message and orderbook data."""
        msg_cols = ["Time", "Event", "OrderID", "Size", "Price", "Direction"]

        ob_cols = []
        for i in range(1, self.levels + 1):
            ob_cols.extend(
                [
                    f"Ask_Price_{i}",
                    f"Ask_Size_{i}",
                    f"Bid_Price_{i}",
                    f"Bid_Size_{i}",
                ]
            )

        if self.message_path.lower().endswith(".zip"):
            with zipfile.ZipFile(self.message_path, "r") as archive:
                names = archive.namelist()

                message_file = next((n for n in names if "_message_" in n), None)
                orderbook_file = next((n for n in names if "_orderbook_" in n), None)

                if message_file is None:
                    raise ValueError("Zip archive does not contain a message file")
                if orderbook_file is None:
                    raise ValueError("Zip archive does not contain an orderbook file")

                print("Loading message file...")
                with archive.open(message_file) as file:
                    messages = pd.read_csv(
                        file,
                        names=msg_cols,
                        dtype={
                            "Time": "float64",
                            "Event": "int16",
                            "OrderID": "int64",
                            "Size": "int32",
                            "Price": "int64",
                            "Direction": "int8",
                        },
                    )

                print("Loading orderbook file...")
                with archive.open(orderbook_file) as file:
                    orderbook = pd.read_csv(file, names=ob_cols, dtype="float64")

        else:
            print("Loading message file...")
            messages = pd.read_csv(
                self.message_path,
                names=msg_cols,
                dtype={
                    "Time": "float64",
                    "Event": "int16",
                    "OrderID": "int64",
                    "Size": "int32",
                    "Price": "int64",
                    "Direction": "int8",
                },
            )

            print("Loading orderbook file...")
            orderbook = pd.read_csv(
                self.orderbook_path,
                names=ob_cols,
                dtype="float64",
            )

        # LOBSTER prices are usually stored as integers scaled by 10000.
        price_cols = [c for c in orderbook.columns if "Price" in c]
        orderbook[price_cols] = orderbook[price_cols] / self.PRICE_SCALE

        # Message price is also scaled, but we mostly use orderbook mid prices.
        if "Price" in messages.columns:
            messages["Price"] = messages["Price"] / self.PRICE_SCALE

        df = pd.concat([messages, orderbook], axis=1)

        df["Mid_Price"] = (df["Ask_Price_1"] + df["Bid_Price_1"]) / 2.0

        denom = df["Bid_Size_1"] + df["Ask_Size_1"]
        df["Micro_Price"] = np.where(
            denom > 0,
            (
                df["Bid_Size_1"] * df["Ask_Price_1"]
                + df["Ask_Size_1"] * df["Bid_Price_1"]
            )
            / denom,
            df["Mid_Price"],
        )

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["Time", "Mid_Price", "Ask_Price_1", "Bid_Price_1"])

        df = df[df["Mid_Price"] > 0].copy()
        df = df.reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def _time_indexed_mid_prices(self, df: pd.DataFrame, freq: str) -> pd.Series:
        temp = df[["Time", "Mid_Price"]].copy()
        temp["Time_Delta"] = pd.to_timedelta(temp["Time"], unit="s")
        temp = temp.set_index("Time_Delta").sort_index()

        prices = temp["Mid_Price"].resample(freq).last().dropna()
        prices = prices[prices > 0]

        return prices

    def _log_returns(self, df: pd.DataFrame, freq: str) -> pd.Series:
        prices = self._time_indexed_mid_prices(df, freq=freq)

        if len(prices) < 2:
            return pd.Series(dtype=float)

        returns = np.log(prices / prices.shift(1)).dropna()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        return returns

    def _intervals_per_day(self, freq: str) -> float:
        if not freq.endswith("min"):
            raise ValueError("Only minute frequencies like '1min' or '5min' are supported")

        minutes = int(freq.replace("min", ""))
        return self.TRADING_MINUTES_PER_DAY / minutes

    # ------------------------------------------------------------------
    # Volatility calibration
    # ------------------------------------------------------------------

    def estimate_volatility(self, df: pd.DataFrame, freq: str = "5min") -> float:
        """
        Estimate daily volatility from mid-price log returns.

        Since the simulator uses T = 1 as one trading day, this should return
        daily volatility, not annualized volatility.
        """
        log_returns = self._log_returns(df, freq=freq)

        if len(log_returns) < 2:
            print("Warning: Insufficient returns for volatility estimation")
            return None

        intervals_per_day = self._intervals_per_day(freq)
        daily_vol = log_returns.std(ddof=1) * np.sqrt(intervals_per_day)

        if not np.isfinite(daily_vol) or daily_vol <= 0:
            print("Warning: Invalid volatility estimate")
            return None

        return float(daily_vol)

    def estimate_annualized_volatility(self, df: pd.DataFrame, freq: str = "5min") -> float:
        """
        Optional helper.

        Use this only if you change the simulator convention to T = 1/252
        for a one-day simulation.
        """
        daily_vol = self.estimate_volatility(df, freq=freq)

        if daily_vol is None:
            return None

        return float(daily_vol * np.sqrt(252.0))

    def _estimate_volatility_series(self, df: pd.DataFrame, window: str = "1min") -> pd.Series:
        """
        Estimate rolling intraday volatility series.

        Returns volatility per local interval, not annualized volatility.
        """
        try:
            returns = self._log_returns(df, freq=window)

            if len(returns) < 20:
                return None

            # Robustly clip microstructure spikes before rolling std.
            lo, hi = returns.quantile([0.001, 0.999])
            returns = returns.clip(lower=lo, upper=hi)

            vol_series = returns.rolling(window=10, min_periods=3).std()
            vol_series = vol_series.replace([np.inf, -np.inf], np.nan).dropna()
            vol_series = vol_series[vol_series > 0]

            if len(vol_series) < 10:
                return None

            return vol_series

        except Exception as e:
            print(f"Error estimating volatility series: {e}")
            return None

    # ------------------------------------------------------------------
    # Impact calibration
    # ------------------------------------------------------------------

    def estimate_impact_parameters(self, df: pd.DataFrame):
        """
        Estimate temporary eta and permanent gamma.

        Temporary impact:
            Simulated market buy order walking the ask book.

        Permanent impact:
            Regression of future mid-price change against lagged signed market-order flow.
        """
        market_orders = df[df["Event"].isin([4, 5])].copy()

        if len(market_orders) == 0:
            print("Warning: No market orders found for impact estimation")
            return {"eta": None, "gamma": None}

        eta = self._estimate_temporary_impact(df, market_orders)
        gamma = self._estimate_permanent_impact(df, market_orders)

        return {"eta": eta, "gamma": gamma}

    def _estimate_temporary_impact(self, df: pd.DataFrame, market_orders: pd.DataFrame):
        """
        Estimate temporary impact by walking the ask book over several trade sizes.

        Fits:
            impact ~= eta * X

        where impact is average execution price minus mid price.
        """
        total_ask_depth = np.zeros(len(df), dtype=float)

        for level in range(1, self.levels + 1):
            size_col = f"Ask_Size_{level}"
            if size_col in df.columns:
                sizes = pd.to_numeric(df[size_col], errors="coerce").fillna(0.0)
                total_ask_depth += sizes.to_numpy(dtype=float)

        total_ask_depth = total_ask_depth[np.isfinite(total_ask_depth)]
        total_ask_depth = total_ask_depth[total_ask_depth > 0]

        if len(total_ask_depth) < 10:
            print("Warning: Insufficient ask depth for temporary impact estimation")
            return None

        min_size = max(1.0, np.percentile(total_ask_depth, 10))
        max_size = max(min_size + 1.0, np.percentile(total_ask_depth, 80))

        trade_sizes = np.linspace(min_size, max_size, 20)

        sample_size = min(len(df), 1000)
        sampled_indices = np.random.default_rng(42).choice(
            df.index.to_numpy(),
            size=sample_size,
            replace=False,
        )

        impacts = []

        for X in trade_sizes:
            trial_impacts = []

            for idx in sampled_indices:
                row = df.loc[idx]
                avg_execution_price = self._walk_ask_book_price(row, X)

                if avg_execution_price is None:
                    continue

                impact = avg_execution_price - row["Mid_Price"]

                if np.isfinite(impact) and impact > 0:
                    trial_impacts.append(float(impact))

            if len(trial_impacts) > 0:
                impacts.append((float(X), float(np.mean(trial_impacts))))

        if len(impacts) < 5:
            print("Warning: Insufficient data for temporary impact estimation")
            return None

        X_vals = np.asarray([x for x, _ in impacts], dtype=float)
        impact_vals = np.asarray([impact for _, impact in impacts], dtype=float)

        valid = (
            np.isfinite(X_vals)
            & np.isfinite(impact_vals)
            & (X_vals > 0)
            & (impact_vals > 0)
        )

        X_vals = X_vals[valid]
        impact_vals = impact_vals[valid]

        if len(X_vals) < 3:
            return None

        # Linear coefficient through origin: impact = eta * X.
        denom = np.dot(X_vals, X_vals)

        if denom <= 0:
            return None

        eta = np.dot(X_vals, impact_vals) / denom

        if not np.isfinite(eta) or eta <= 0:
            return None

        return float(eta)

    def _walk_ask_book_price(self, row, trade_size: float):
        """
        Computes average execution price for a market buy order
        by walking through ask levels.
        """
        remaining = float(trade_size)
        total_cash = 0.0
        total_filled = 0.0

        for level in range(1, self.levels + 1):
            price_col = f"Ask_Price_{level}"
            size_col = f"Ask_Size_{level}"

            if price_col not in row or size_col not in row:
                break

            price = row[price_col]
            size = row[size_col]

            if pd.isna(price) or pd.isna(size) or price <= 0 or size <= 0:
                continue

            fill = min(remaining, float(size))

            total_cash += fill * float(price)
            total_filled += fill
            remaining -= fill

            if remaining <= 0:
                break

        if total_filled <= 0 or remaining > 0:
            return None

        return total_cash / total_filled

    def _estimate_permanent_impact(self, df: pd.DataFrame, market_orders: pd.DataFrame):
        """
        Estimate permanent impact parameter gamma using signed market-order flow.

        Fits:
            future mid-price change ~= gamma * lagged signed market-order flow

        This is still noisy, but it is better than using all message events.
        """
        temp = market_orders[["Time", "Mid_Price", "Size", "Direction"]].copy()

        if len(temp) < 10:
            print("Warning: Insufficient market-order data for permanent impact estimation")
            return None

        temp["Time_Delta"] = pd.to_timedelta(temp["Time"], unit="s")

        temp["Signed_Size"] = np.where(
            temp["Direction"] == 1,
            temp["Size"],
            np.where(temp["Direction"] == -1, -temp["Size"], 0),
        )

        temp = temp.set_index("Time_Delta").sort_index()

        df_resampled = (
            temp.resample("5min")
            .agg(
                {
                    "Mid_Price": "last",
                    "Signed_Size": "sum",
                }
            )
            .dropna()
        )

        if len(df_resampled) < 10:
            print("Warning: Insufficient resampled data for permanent impact estimation")
            return None

        # Use price change, not percent return, because gamma in AC is price/share.
        df_resampled["Mid_Change"] = df_resampled["Mid_Price"].diff()
        df_resampled["Lagged_Order_Flow"] = df_resampled["Signed_Size"].shift(1)

        valid_data = df_resampled.dropna()

        valid_data = valid_data[
            np.isfinite(valid_data["Mid_Change"])
            & np.isfinite(valid_data["Lagged_Order_Flow"])
        ]

        valid_data = valid_data[valid_data["Lagged_Order_Flow"] != 0]

        if len(valid_data) < 3:
            print("Warning: Not enough valid points for permanent impact regression")
            return None

        x = valid_data["Lagged_Order_Flow"].to_numpy(dtype=float)
        y = valid_data["Mid_Change"].to_numpy(dtype=float)

        if np.std(x) <= 0 or np.std(y) <= 0:
            print("Warning: Degenerate permanent impact regression")
            return None

        slope, intercept = np.polyfit(x, y, 1)

        gamma = abs(float(slope))

        if not np.isfinite(gamma) or gamma <= 0:
            return None

        return gamma

    # ------------------------------------------------------------------
    # Heston calibration
    # ------------------------------------------------------------------

    def estimate_heston_parameters(self, df: pd.DataFrame):
        """
        Estimate Heston-like parameters from intraday data.

        Returns:
            {
                "v0": initial/current variance,
                "mu": drift,
                "theta": mean reversion speed / kappa,
                "omega": long-run variance,
                "xi": volatility of variance,
                "rho": return/variance-shock correlation,
            }
        """
        vol_series = self._estimate_volatility_series(df, window="1min")

        if vol_series is None or len(vol_series) < 50:
            print("Warning: Insufficient volatility data for Heston estimation")
            return None

        var_series = (vol_series ** 2).replace([np.inf, -np.inf], np.nan).dropna()
        var_series = var_series[var_series > 0]

        if len(var_series) < 50:
            print("Warning: Insufficient variance data for Heston estimation")
            return None

        kappa, omega, xi, rho = self._estimate_volatility_parameters(df, var_series)

        params = {
            "v0": float(max(var_series.iloc[-1], 1e-12)),
            "mu": 0.0,
            "theta": float(kappa),   # mean reversion speed
            "omega": float(omega),   # long-run variance
            "xi": float(xi),
            "rho": float(rho),
        }

        # Safety bounds to prevent catastrophic Heston paths.
        params["v0"] = float(np.clip(params["v0"], 1e-12, 1.0))
        params["theta"] = float(np.clip(params["theta"], 1e-6, 50.0))
        params["omega"] = float(np.clip(params["omega"], 1e-12, 1.0))
        params["xi"] = float(np.clip(params["xi"], 1e-6, 5.0))
        params["rho"] = float(np.clip(params["rho"], -0.95, 0.95))

        # Soft Feller guard:
        # 2 * kappa * omega > xi^2.
        # Instead of rejecting, reduce xi if needed.
        feller_lhs = 2.0 * params["theta"] * params["omega"]
        feller_rhs = params["xi"] ** 2

        if feller_lhs <= feller_rhs:
            adjusted_xi = 0.95 * np.sqrt(max(feller_lhs, 1e-12))
            print(
                "Warning: calibrated Heston parameters violate the Feller condition; "
                f"adjusting xi from {params['xi']:.6e} to {adjusted_xi:.6e}."
            )
            params["xi"] = float(max(adjusted_xi, 1e-6))

        return params

    def _estimate_volatility_parameters(self, df: pd.DataFrame, var_series: pd.Series):
        """
        Estimate kappa, omega, xi, and rho for Heston variance dynamics.

        Heston variance approximation:
            dv = kappa * (omega - v_t) * dt + xi * sqrt(v_t) * dW
        """
        try:
            var_series = var_series.replace([np.inf, -np.inf], np.nan).dropna()
            var_series = var_series[var_series > 0]

            if len(var_series) < 20:
                return 2.0, float(var_series.mean()) if len(var_series) else 1e-4, 0.3, -0.5

            # _estimate_volatility_series uses 1-minute sampling.
            # Since T = 1 means one trading day:
            dt = 1.0 / self.TRADING_MINUTES_PER_DAY

            v_t = var_series.iloc[:-1].to_numpy(dtype=float)
            v_next = var_series.iloc[1:].to_numpy(dtype=float)
            dv = v_next - v_t

            valid = np.isfinite(v_t) & np.isfinite(dv) & (v_t > 0)

            v_t = v_t[valid]
            dv = dv[valid]

            if len(v_t) < 10:
                omega = float(var_series.mean())
                return 2.0, omega, 0.3, -0.5

            # Least-squares fit:
            # dv = a + b * v_t
            # b = -kappa * dt
            # a = kappa * omega * dt
            X = np.column_stack([np.ones_like(v_t), v_t])
            beta, *_ = np.linalg.lstsq(X, dv, rcond=None)

            a, b = beta

            kappa = -b / dt

            if np.isfinite(kappa) and kappa > 0:
                omega_from_regression = a / (kappa * dt)
            else:
                kappa = 2.0
                omega_from_regression = float(np.mean(v_t))

            if not np.isfinite(omega_from_regression) or omega_from_regression <= 0:
                omega_from_regression = float(np.mean(v_t))

            kappa = float(np.clip(kappa, 1e-4, 50.0))
            omega = float(np.clip(omega_from_regression, 1e-12, 1.0))

            residuals = dv - kappa * (omega - v_t) * dt

            denom = np.sqrt(np.maximum(v_t * dt, 1e-12))
            xi_samples = residuals / denom

            xi = float(np.nanstd(xi_samples, ddof=1))

            if not np.isfinite(xi) or xi <= 0:
                xi = 0.3

            xi = float(np.clip(xi, 1e-6, 5.0))

            rho = self._estimate_return_variance_correlation(df, var_series)

            return kappa, omega, xi, rho

        except Exception as e:
            print(f"Error in volatility parameter estimation: {e}")

            fallback_omega = 1e-4
            try:
                fallback_omega = float(np.nanmean(var_series))
                if not np.isfinite(fallback_omega) or fallback_omega <= 0:
                    fallback_omega = 1e-4
            except Exception:
                pass

            return 2.0, fallback_omega, 0.3, -0.5

    def _estimate_return_variance_correlation(
        self,
        df: pd.DataFrame,
        var_series: pd.Series,
    ) -> float:
        """Estimate rho from 1-minute log returns and variance changes."""
        try:
            returns = self._log_returns(df, freq="1min")
            var_changes = var_series.diff().replace([np.inf, -np.inf], np.nan).dropna()

            common_index = returns.index.intersection(var_changes.index)

            if len(common_index) > 10:
                rho = returns.loc[common_index].corr(var_changes.loc[common_index])

                if np.isfinite(rho):
                    return float(np.clip(rho, -0.95, 0.95))

            return -0.5

        except Exception:
            return -0.5

    # ------------------------------------------------------------------
    # Optional diagnostic helper
    # ------------------------------------------------------------------

    def print_calibration_diagnostics(self, df: pd.DataFrame):
        """Print useful sanity checks for calibrated values."""
        sigma = self.estimate_volatility(df)
        impact = self.estimate_impact_parameters(df)
        heston = self.estimate_heston_parameters(df)

        print("\n========== CALIBRATION DIAGNOSTICS ==========")

        print(f"Rows: {len(df)}")
        print(f"Mid price start: {df['Mid_Price'].iloc[0]:.6f}")
        print(f"Mid price end:   {df['Mid_Price'].iloc[-1]:.6f}")

        print("\nVolatility:")
        print(f"  daily sigma = {sigma}")

        print("\nImpact:")
        print(f"  eta   = {impact.get('eta')}")
        print(f"  gamma = {impact.get('gamma')}")

        print("\nHeston:")
        if heston is None:
            print("  Heston calibration failed.")
        else:
            for key, value in heston.items():
                print(f"  {key:<6} = {value:.12e}")

            feller_lhs = 2.0 * heston["theta"] * heston["omega"]
            feller_rhs = heston["xi"] ** 2
            print(f"\n  Feller lhs 2*theta*omega = {feller_lhs:.12e}")
            print(f"  Feller rhs xi^2          = {feller_rhs:.12e}")

        print("=============================================\n")