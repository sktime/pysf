
from pysf.data import MultiSeries, load_dummy_data_df, download_ramsay_weather_data_dfs, download_ramsay_growth_data_dfs, download_ecg_data_dfs
import pandas as pd

# Load up dummy data and visualise
(dummy_vs_times_df, dummy_vs_series_df) = load_dummy_data_df()
dummy_data = MultiSeries(data_vs_times_df = dummy_vs_times_df, data_vs_series_df = dummy_vs_series_df, time_colname='timestamp', series_id_colnames='series')
dummy_data.visualise()

# Download Canadian weather data and visualise
(weather_vs_times_df, weather_vs_series_df) = download_ramsay_weather_data_dfs()
data_weather = MultiSeries(data_vs_times_df=weather_vs_times_df, data_vs_series_df=weather_vs_series_df, time_colname='day_of_year', series_id_colnames='weather_station')
data_weather.visualise()
       
# Download Berkeley growth data and visualise
(growth_vs_times_df, growth_vs_series_df) = download_ramsay_growth_data_dfs()
growth_vs_series_df['gender'] = growth_vs_series_df['gender'].astype('category')
growth_vs_series_df = pd.concat([growth_vs_series_df, pd.get_dummies(growth_vs_series_df['gender'])], axis=1)
data_growth = MultiSeries(data_vs_times_df=growth_vs_times_df, data_vs_series_df=growth_vs_series_df, time_colname='age', series_id_colnames=['gender', 'cohort_id'])
data_growth.visualise()    

# Download ECG data and visualise
(ecg_vs_times_df, ecg_vs_series_df) = download_ecg_data_dfs()
data_ecg = MultiSeries(data_vs_times_df=ecg_vs_times_df, data_vs_series_df=ecg_vs_series_df, time_colname='timestamp', series_id_colnames='heartbeat')
data_ecg.visualise()

