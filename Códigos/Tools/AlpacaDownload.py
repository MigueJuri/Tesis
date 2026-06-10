from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from tqdm.auto import tqdm
import pandas as pd

# 1. Setup Client
API_KEY = "PKNPLUUHRK5EF67XWNIKDYP5LX"
SECRET_KEY = "B3jdCgy9eUJWqrq6h7Gi34QpPxHiA5d7HMxU7rkHY95E"
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

symbol = "SPY"
start_date = datetime(2015, 1, 1)
end_date = datetime(2025, 12, 31)
time_frame = TimeFrame.Minute
output_file = rf"G:\Mi unidad\2026\Tesis\Códigos\Tools\{symbol}_{start_date.strftime('%Y')}_to_{end_date.strftime('%Y')}_{time_frame.value}.csv"
def download_stock_data(symbol, start_date, end_date, time_frame, output_file):
    # 2. Build monthly chunks
    chunks = []
    chunk_start = pd.Timestamp(start_date)
    final_end = pd.Timestamp(end_date)

    while chunk_start < final_end:
        chunk_end = min(chunk_start + pd.DateOffset(months=1), final_end)
        chunks.append((chunk_start.to_pydatetime(), chunk_end.to_pydatetime()))
        chunk_start = chunk_end

    # 3. Stream directly to CSV (faster than concat + single huge write)
    header_written = False
    last_written_ts = None
    total_rows_written = 0

    with tqdm(total=len(chunks), desc=f"Downloading {symbol}", unit="chunk") as pbar:
        for i, (current_start, current_end) in enumerate(chunks, start=1):
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=time_frame,
                start=current_start,
                end=current_end,
            )
            bars = client.get_stock_bars(request_params)
            part = bars.df

            if not part.empty:
                # Ensure deterministic order and avoid duplicate boundary rows between chunks.
                part = part.sort_index()
                if last_written_ts is not None:
                    part = part[part.index > last_written_ts]

                if not part.empty:
                    part.to_csv(
                        output_file,
                        mode='w' if not header_written else 'a',
                        header=not header_written
                    )
                    header_written = True
                    last_written_ts = part.index.max()
                    total_rows_written += len(part)

            pbar.set_postfix(chunk=f"{i}/{len(chunks)}", rows=f"{total_rows_written:,}")
            pbar.update(1)

    # Keep a DataFrame in memory only if needed later in notebook
    if header_written:
        df = pd.read_csv(output_file, parse_dates=['timestamp'])
    else:
        df = pd.DataFrame()
        df.to_csv(output_file, index=False)

    print(f"Download complete. File saved as {output_file}. Rows: {total_rows_written:,}")