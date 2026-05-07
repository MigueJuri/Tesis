import pandas as pd

def load_data_from_csv(csv_file: str) -> pd.DataFrame:
    # yfinance CSV export uses a 2-row header for MultiIndex columns and a standalone index-name row.
    data = pd.read_csv(csv_file, header=[0, 1], skiprows=[2], index_col=0)

    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"

    return data


output_file = r"G:\Mi unidad\2026\Tesis\Códigos\Data\sp500_data_2015-01-12_to_2026-01-02.csv"

data_from_csv = load_data_from_csv(output_file)
