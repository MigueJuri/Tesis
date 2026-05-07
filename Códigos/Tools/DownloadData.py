import yfinance as yf
import pandas as pd
from pandas.testing import assert_frame_equal


def load_tickers(ticker_file: str) -> list[str]:
    with open(ticker_file, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f.readlines() if line.strip()]

    if not tickers:
        raise ValueError(f"No tickers found in {ticker_file}")

    return tickers


def download_and_save_data(ticker_file: str, start_date: str, end_date: str, output_dir: str) -> tuple[pd.DataFrame, str]:
    tickers = load_tickers(ticker_file)

    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError("Downloaded data is empty. Check tickers and date range.")

    output_file = rf"{output_dir}\sp500_data_only_{data.index[0].date()}_to_{data.index[-1].date()}.csv"
    data.to_csv(output_file)

    return data, output_file


ticker_file = r"G:\Mi unidad\2026\Tesis\Códigos\Data\tickers\sp500_tickers_Only.txt"
output_dir = r"G:\Mi unidad\2026\Tesis\Códigos\Data"

download_and_save_data(
    ticker_file=ticker_file,
    start_date="1992-01-01",
    end_date="2026-01-03",
    output_dir=output_dir,
)