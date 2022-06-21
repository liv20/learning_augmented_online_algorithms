"""Module for interfacing with BTC data in csv files."""
from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd


BASE_URL = 'https://raw.githubusercontent.com/jrrhuang/data/main/BTCUSD/'
DT_FORMAT = '%Y-%m-%d'
DATA_PATHS = []
for year in range(2022, 2014, -1):
    DATA_PATHS.append(BASE_URL + f'gemini_BTCUSD_{year}_1min.csv')

START_DATE_2015_STR = '2015-10-08'
END_DATE_2022_STR = '2022-03-20'
START_DATE_2015 = datetime.strptime(START_DATE_2015_STR, DT_FORMAT)
END_DATE_2022 = datetime.strptime(END_DATE_2022_STR, DT_FORMAT)


class BTCDataProcessor:
    """
    Data processor for loading Bitcoin data from csv files.
    
    Includes methods for
    downsampling, getting data, and adding temporal structure.
    """

    def __init__(self, interval: int = 5) -> None:
        """Initialize BTCDataLoader object."""
        self.df = pd.concat([pd.read_csv(data, skiprows=1, parse_dates=['Date'], index_col=['Date'])
                        for data in DATA_PATHS])
        self.df = self.df.sort_index()
        self.already_downsampled = False

    def downsample(self, sample_method='5T', inplace=False) -> pd.DataFrame:
        """
        Downsample BTC data according to resample_method.

        Arguments:
        sample_method (str) - resampling method used in pd.DataFrame.resample.
            The first part of the string tells how many periods to resample.
            The second part tells what the periods are.
            'T': minute
            'D': day
            'W': week
            'M': month
            'Q': quarterly
            'Y': yearly
        inplace (bool) - replaces internal DataFrame if True, otherwise returns
            DataFrame
        
        Returns:
        pd.DataFrame - downsampled DataFrame
        """
        ndf = pd.DataFrame()
        ndf = self.df.resample(sample_method).first()
        ndf['High'] = self.df['High'].resample(sample_method).max()
        ndf['Low'] = self.df['Low'].resample(sample_method).min()
        ndf['Close'] = self.df['Close'].resample(sample_method).last()
        ndf['Volume'] = self.df['Volume'].resample(sample_method).sum()

        if inplace:
            self.df = ndf
        else:
            return ndf

    def get_window(self, start_dt: str, end_dt: str) -> pd.DataFrame:
        """
        Return a slice of the internal DataFrame from start_dt to end_dt.

        Arguments:
        start_dt (str) - start datetime
        end_dt (str) - end datetime

        Notes:
        Both `start_dt` and `end_dt` should have format "YYYY-MM-DD HH:mm:ss"

        Returns:
        pd.DataFrame - slice of internal DataFrame
        """
        return self.df[start_dt: end_dt]

    def add_temporal_structure(self) -> None:
        """
        Add day/time signals using sine/cosine transforms to internal DataFrame.
        """
        self.df['Day of Year'] = pd.to_numeric(self.df.index.strftime('%j'), errors='coerce')
        self.df['Time of Day'] = self.df.index.hour * 60 + self.df.index.minute

        day = 24 * 60
        year = 365.2425
        self.df['Year cos'] = np.cos(self.df['Day of Year'] * (2 * np.pi / year))
        self.df['Year sin'] = np.sin(self.df['Day of Year'] * (2 * np.pi / year))
        self.df['Day cos'] = np.cos(self.df['Time of Day'] * (2 * np.pi / day))
        self.df['Day sin'] = np.sin(self.df['Time of Day'] * (2 * np.pi / day))

        del self.df['Day of Year'], self.df['Time of Day']


class BTCDataIterator:
    """Iterator for BTC data from a pd.DataFrame object."""

    def __init__(self, df: pd.DataFrame, start_date: str=None, end_date: str=None):
        """
        Initialize iterator for BTC data from a pd.DataFrame object.

        Loads one-week of data starting on the Monday of each week.

        Arguments:
        df (pd.DataFrame) - BTC data
        start_date (str) - start date of BTC data trace
        end_date (str) - end date of BTC data trace

        Notes:
        Both `start_date` and `end_date` should be in YYYY-MM-DD format
        """
        # find start and end date in data, starting on Monday of the week
        if start_date is None:
            start_date = START_DATE_2015_STR
        if end_date is None:
            end_date = END_DATE_2022_STR

        start_date = datetime.strptime(start_date, DT_FORMAT)
        end_date = datetime.strptime(end_date, DT_FORMAT)
        # bound start and end date
        if start_date < START_DATE_2015:
            start_date = START_DATE_2015
        if end_date > END_DATE_2022:
            end_date = END_DATE_2022
        
        # find first Monday
        days = (7 - start_date.weekday()) % 7
        self.start_date = start_date + timedelta(days=days)
        self.end_date = end_date

        self.df = df

    def __iter__(self):
        """Iterate through data by week."""
        self.curweek = self.start_date
        self.nextweek = self.curweek + timedelta(days=6)
        return self

    def __next__(self):
        """Get next week's data until end of data set."""
        if self.nextweek > self.end_date:
            raise StopIteration

        curweekstr = self.curweek.strftime(DT_FORMAT)
        nextweekstr = self.nextweek.strftime(DT_FORMAT)
        data = self.df.loc[curweekstr: nextweekstr]

        self.curweek = self.curweek + timedelta(days=7)
        self.nextweek = self.nextweek + timedelta(days=7)

        return data
