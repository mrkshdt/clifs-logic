from darts import TimeSeries
from darts.metrics import mape, rmse, coefficient_of_variation
from darts.models import NBEATSModel

import plotly.express as px

import pandas as pd
import statistics
import numpy as np

class nbeats:
    result = 0
    model_nbeats = NBEATSModel(
                    input_chunk_length=30,
                    output_chunk_length=7,
                    generic_architecture=True,
                    num_stacks=10,
                    num_blocks=1,
                    num_layers=4,
                    layer_widths=512,
                    n_epochs=100,
                    nr_epochs_val_period=1,
                    batch_size=800,
                    model_name="nbeats_run",
                    )
    
    
    def __init__(self, horizon, df):
        self.horizon = horizon
        
        self.original = df
        df = df.set_index("date")[["values"]]
        series = TimeSeries.from_dataframe(df)
        self.df = series
        
    def backtest(self):
        pred_series = self.model_nbeats.historical_forecasts(
                                    self.df,
                                    forecast_horizon=self.horizon,
                                    stride=5,
                                    retrain=False,
                                    verbose=True,
                                    )
        
        print("MAPE = {:.2f}%".format(mape(pred_series, self.df)))
        print("RMSPE = {:.2f}%".format(coefficient_of_variation(pred_series, self.df)))
        self.df.plot(label="data")
        pred_series.plot(label="backtest")
        return
    
    def predict(self):
        self.model_nbeats.fit(self.df)
        self.result_ts = self.model_nbeats.predict(self.horizon)
        result_df = self.result_ts.pd_dataframe()
        tmp = pd.concat([self.original.set_index("date").rename(columns={"values":"historical"}),result_df.rename(columns={"values":"forecast"})])
        self.output = tmp
        return self.output
    
    def plot_prediction(self):
        fig = px.line(self.output[["historical","forecast"]],color_discrete_sequence=["black", "grey"])
        fig.add_hline(y=0.15, line_width=3, line_dash="dash", line_color="red")
        fig.add_vline(x=self.original.date.iloc[-1], line_width=1)
        fig.update_layout(
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',
                ),
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
            ),
            autosize=False,
            margin=dict(
                autoexpand=False,
                l=100,
                r=20,
                t=110,
            ),
            showlegend=False,
            plot_bgcolor='white'
        )

        fig.update_layout(
            title="Financial Stress "+str(self.horizon)+ "-Month Forecast",
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Financial Stress",
            legend_title="Legend Title",
            font=dict(
                size=14
            )
        )
        return fig.show()

