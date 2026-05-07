import requests
import pandas as pd
from io import StringIO
from pathlib import Path

def get_sp500_tickers_by_sector(target_sector: str = "Energy") -> list:
    """
    Scrapes the S&P 500 constituents and filters them by GICS Sector.
    This restricts our state-space to a highly coupled sub-manifold.
    
    Args:
        target_sector: GICS Sector name to filter by, or "All" to get all S&P 500 tickers
    
    Returns:
        List of tickers in the specified sector(s)
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Required header to bypass CDN bot-protection
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ConnectionError(f"HTTP Error: {response.status_code}")
        
    # Read the first table into a DataFrame
    df = pd.read_html(StringIO(response.text))[0]
    
    # If "All" is requested, return all tickers without filtering
    if target_sector.lower() == "all":
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    
    # Check available sectors to prevent silent failures
    available_sectors = df['GICS Sector'].unique()
    if target_sector not in available_sectors:
        raise ValueError(f"Sector '{target_sector}' not found. Valid sectors are: {available_sectors}")
    
    # Apply the topological constraint (boolean filtering)
    sector_df = df[df['GICS Sector'] == target_sector]
    
    # Format tickers for yfinance compatibility
    tickers = sector_df['Symbol'].str.replace('.', '-', regex=False).tolist()
    
    return tickers


def save_tickers_to_file(tickers: list, output_file: str = "tickers.txt") -> Path:
    """
    Saves ticker symbols (one per line) to a text file.

    Args:
        tickers: List of ticker symbols
        output_file: Path to the output text file

    Returns:
        Path object of the created file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(tickers), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    sector = "All"  # Use "All" to export all S&P 500 tickers
    tickers = get_sp500_tickers_by_sector(sector)
    saved_file = save_tickers_to_file(
        tickers,
        rf"G:\Mi unidad\2026\Tesis\Códigos\Data\tickers\sp500_tickers_{sector}.txt",

    )
    print(f"Saved {len(tickers)} tickers to {saved_file.resolve()}")