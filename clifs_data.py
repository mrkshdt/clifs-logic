import requests
import plotly.express as px
import pandas as pd
import pandas.api.types as ptypes
import io

class clifs_data:
    url_front = "https://sdw.ecb.europa.eu/quickviewexport.do;jsessionid=5BC0745335A1AC119A62EB6CE8107BCC?SERIES_KEY=383.CLIFS.M."
    url_back = "._Z.4F.EC.CLIFS_CI.IDX&type=csv"
    df = 0
    
    def __init__(self, country):
        self.country = country
           
    def get_data(self):
        r = requests.get(self.url_front+str(self.country)+self.url_back)
        t = [x.split(",") for x in r.text.split("\r\n")[6:]]
        try:
            df = pd.DataFrame(t, columns =['date', 'values', 'type'])
        except:
            df = pd.DataFrame(t, columns =['date', 'values', 'type','tmp'])
            df.drop(columns="tmp")
        df["values"] = df["values"].astype(str).apply(lambda x: x.replace(',', '.'))
        df['values'] = df['values'].astype(float)
        df.date = pd.to_datetime(df.date.astype(str),format='%Y%b')
        self.df = df.sort_values(by="date")
        return self.df
    
    def plot_map(self, country):
        tmp = self.df
        tmp["location"] = [str(country) for i in tmp.iterrows()]
        fig = px.choropleth(tmp,locations="location", color="values"
                            ,scope="europe",range_color=(0,1),color_continuous_scale="magma")
        return fig.show()
    
    def type_test(self,df):
        assert ptypes.is_numeric_dtype(df['values']), "values column contains non-numeric values"
        assert ptypes.is_datetime64_any_dtype(df.date), "date is not datetime"
        return True