from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest
from datetime import datetime, timedelta
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import os
import time
import threading
from pathlib import Path
from dotenv import load_dotenv

class CleanTradesDownloader:
    """
    High-performance, thread-safe transactions downloader with integrated data cleaning.
    Uses file partitioning to prevent OOM errors and data corruption.
    """
    
    def __init__(self, api_key: str, secret_key: str, max_workers: int = 3):
        self.client = StockHistoricalDataClient(api_key, secret_key)
        self.max_workers = max_workers
        self.write_lock = threading.Lock()
        self.session_stats = {
            'downloaded_raw': 0,
            'saved_clean': 0,
            'cached_days': 0,
            'failed_chunks': 0,
            'start_time': time.time()
        }
    
    def _build_daily_chunks(self, start_date: datetime, end_date: datetime) -> list:
        """Forces daily chunking to prevent memory exhaustion."""
        chunks = []
        current = start_date
        while current <= end_date:
            next_day = current + timedelta(days=1)
            chunks.append((current, next_day))
            current = next_day
        return chunks

    def _clean_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard academic/institutional cleaning filters to raw trade data.
        """
        if df.empty:
            return df

        # 1. Timezone Conversion (UTC to EST/EDT)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df = df.tz_convert('America/New_York')

        # 2. Filter Main Market Hours (09:30:00 to 16:00:00)
        df = df.between_time('09:30', '16:00')

        if df.empty:
            return df

        # 3. Filter Invalid/Anomalous Conditions
        # Alpaca conditions are lists of strings. We convert to string to search efficiently.
        # Exclude: 'Z' (Out of Sequence), 'C' (Cash), 'V' (Stock Option), 'T' (Extended), 'U' (Extended)
        # Note: 'T' and 'U' should already be removed by the time filter, but included for safety.
        invalid_conditions = ['Z', 'C', 'V', 'T', 'U']
        
        # Keep rows where NONE of the invalid conditions exist in the conditions list
        condition_mask = df['conditions'].astype(str).apply(
            lambda x: not any(cond in x for cond in invalid_conditions)
        )
        df = df[condition_mask]

        # 4. Basic Sanity Check
        df = df[(df['price'] > 0) & (df['size'] > 0)]

        return df
    
    def _download_chunk_with_retry(self, symbol: str, start: datetime, end: datetime, max_retries: int = 3):
        """Downloads a single chunk with exponential backoff."""
        for attempt in range(max_retries):
            try:
                request_params = StockTradesRequest(
                    symbol_or_symbols=symbol,
                    start=start,
                    end=end,
                    feed='sip' # Change to 'iex' if on the free tier
                )
                
                trades_response = self.client.get_stock_trades(request_params)
                if not trades_response.data or symbol not in trades_response.data:
                    return (pd.DataFrame(), pd.DataFrame(), None)
                
                raw_df = trades_response.df
                if not raw_df.empty:
                    # Drop the 'symbol' multi-index to flatten
                    raw_df = raw_df.reset_index(level=0, drop=True)
                    
                    # Apply the cleaning pipeline
                    clean_df = self._clean_trades(raw_df.copy())
                    
                    return (raw_df, clean_df, None)
                return (pd.DataFrame(), pd.DataFrame(), None)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return (None, None, str(e))
        return (None, None, "Max retries exceeded")
    
    def download_to_partitions(self, symbol: str, start_date: datetime, end_date: datetime, output_dir: str):
        """Downloads trades, cleans them, and saves them as daily partitioned CSVs."""
        base_dir = Path(output_dir) / symbol
        base_dir.mkdir(parents=True, exist_ok=True)
        
        chunks = self._build_daily_chunks(start_date, end_date)
        chunks_to_download = []
        
        # O(1) Caching: Check if the file already exists
        for start, end in chunks:
            date_str = start.strftime('%Y%m%d')
            file_path = base_dir / f"{symbol}_clean_{date_str}.csv"
            
            if file_path.exists():
                self.session_stats['cached_days'] += 1
            else:
                chunks_to_download.append((start, end, file_path))
        
        print(f"Target: {symbol} | Total Days: {len(chunks)} | Cached: {self.session_stats['cached_days']} | To Download: {len(chunks_to_download)}")
        
        if not chunks_to_download:
            print("All requested data is cached locally.")
            return
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._download_chunk_with_retry, symbol, start, end): file_path
                for start, end, file_path in chunks_to_download
            }
            
            with tqdm(total=len(chunks_to_download), desc=f"↓ {symbol} trades") as pbar:
                for future in as_completed(futures):
                    file_path = futures[future]
                    raw_df, clean_df, error = future.result()
                    
                    if error:
                        self.session_stats['failed_chunks'] += 1
                    elif clean_df is not None and not clean_df.empty:
                        # Save only the cleaned data
                        clean_df.to_csv(file_path, index_label='timestamp')
                        
                        with self.write_lock:
                            self.session_stats['downloaded_raw'] += len(raw_df)
                            self.session_stats['saved_clean'] += len(clean_df)
                    
                    pbar.update(1)
        
        filtered_out = self.session_stats['downloaded_raw'] - self.session_stats['saved_clean']
        print(f"\nDownload complete.")
        print(f"Raw rows fetched: {self.session_stats['downloaded_raw']:,}")
        print(f"Clean rows saved: {self.session_stats['saved_clean']:,}")
        print(f"Noise filtered out: {filtered_out:,} rows")

# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Ensure this is running in the correct directory to find the .env file
    script_dir = Path(__file__).parent
    env_path = script_dir / '.env'
    load_dotenv(dotenv_path=env_path)
    
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    
    if not API_KEY or not SECRET_KEY:
        raise ValueError("Missing API credentials. Set them in a .env file.")

    downloader = CleanTradesDownloader(API_KEY, SECRET_KEY, max_workers=3)

    downloader.download_to_partitions(
        symbol="SPY",
        start_date=datetime(2025, 6, 6),
        end_date=datetime(2025, 12, 31),
        output_dir=str(script_dir / "market_data")
    )