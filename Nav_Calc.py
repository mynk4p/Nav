# nav_calculator.py
"""
Fund NAV Calculator
-------------------
Calculates daily NAV from holdings, market prices, cashflows, and units outstanding.

Usage:
    python nav_calculator.py

Author: Mayank Prajapat
Date: 2025-08-11
"""

import pandas as pd
from pathlib import Path


class NAVCalculator:
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.holdings = None
        self.prices = None
        self.cashflows = None
        self.units = None
        self.all_dates = None

    def load_data(self):
        """Load all CSV data files."""
        self.holdings = pd.read_csv(self.data_dir / "holdings.csv", parse_dates=["date"])
        self.prices = pd.read_csv(self.data_dir / "prices.csv", parse_dates=["date"])
        self.cashflows = pd.read_csv(self.data_dir / "cashflows.csv", parse_dates=["date"])
        self.units = pd.read_csv(self.data_dir / "units_outstanding.csv", parse_dates=["date"])

        # Normalize dates to midnight
        for df in (self.holdings, self.prices, self.cashflows, self.units):
            df["date"] = df["date"].dt.normalize()

        # Define full date range
        min_date = min(df["date"].min() for df in (self.holdings, self.prices, self.cashflows, self.units))
        max_date = max(df["date"].max() for df in (self.holdings, self.prices, self.cashflows, self.units))
        self.all_dates = pd.date_range(min_date, max_date, freq="D")

    def prepare_holdings(self):
        """Pivot holdings into daily positions per ticker."""
        daily = self.holdings.groupby(["date", "ticker"], as_index=False)["quantity"].sum()
        pivot = daily.pivot(index="date", columns="ticker", values="quantity")
        pivot = pivot.reindex(self.all_dates).fillna(method="ffill").fillna(0)
        return pivot

    def prepare_prices(self):
        """Pivot prices into daily close per ticker."""
        pivot = self.prices.pivot(index="date", columns="ticker", values="close")
        pivot = pivot.reindex(self.all_dates).fillna(method="ffill")
        return pivot

    def calculate_assets(self, holdings_pivot, prices_pivot):
        """Calculate market value of all holdings."""
        # Align tickers in both dataframes
        all_tickers = sorted(set(holdings_pivot.columns) | set(prices_pivot.columns))
        holdings_pivot = holdings_pivot.reindex(columns=all_tickers, fill_value=0)
        prices_pivot = prices_pivot.reindex(columns=all_tickers)

        market_value = holdings_pivot * prices_pivot
        market_value = market_value.fillna(0)
        return market_value.sum(axis=1)

    def calculate_cash_and_liabilities(self):
        """Calculate daily cash balance and liabilities."""
        cf_by_date = self.cashflows.groupby("date")["amount"].sum().reindex(self.all_dates).fillna(0)
        cash_balance = cf_by_date.cumsum()

        fees = self.cashflows[self.cashflows["type"].str.lower().str.contains("fee", na=False)]
        fees_by_date = fees.groupby("date")["amount"].sum().reindex(self.all_dates).fillna(0)
        liabilities = (-fees_by_date).clip(lower=0)

        dividends = self.cashflows[self.cashflows["type"].str.lower().str.contains("dividend", na=False)]
        div_by_date = dividends.groupby("date")["amount"].sum().reindex(self.all_dates).fillna(0)

        return cash_balance, liabilities, div_by_date

    def prepare_units(self):
        """Forward-fill units outstanding."""
        units_series = self.units.set_index("date")["units_outstanding"]
        units_series = units_series.reindex(self.all_dates).fillna(method="ffill")
        if units_series.isna().any():
            raise ValueError("Units outstanding missing for some dates")
        return units_series

    def compute_nav(self):
        """Run full NAV calculation pipeline."""
        holdings_pivot = self.prepare_holdings()
        prices_pivot = self.prepare_prices()
        assets_securities = self.calculate_assets(holdings_pivot, prices_pivot)
        cash_balance, liabilities, dividends = self.calculate_cash_and_liabilities()
        units_series = self.prepare_units()

        total_assets = assets_securities + cash_balance + dividends.cumsum()
        net_assets = total_assets - liabilities
        nav_per_unit = net_assets / units_series

        result = pd.DataFrame({
            "date": self.all_dates.date,
            "assets_from_securities": assets_securities,
            "cash_balance": cash_balance,
            "cumulative_dividends": dividends.cumsum(),
            "liabilities": liabilities,
            "net_assets": net_assets,
            "units_outstanding": units_series,
            "nav_per_unit": nav_per_unit
        })
        return result

    def save_nav(self, df, filename="daily_nav.csv"):
        """Save NAV history to CSV."""
        output_path = self.data_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Saved NAV data to {output_path}")


if __name__ == "__main__":
    nav = NAVCalculator()
    nav.load_data()
    nav_df = nav.compute_nav()
    nav.save_nav(nav_df)
    print(nav_df.head(10).to_string(index=False, float_format="%.2f"))
