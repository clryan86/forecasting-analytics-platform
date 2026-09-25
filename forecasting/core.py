import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

def metrics(a,p):
    a=np.asarray(a,float); p=np.asarray(p,float); e=a-p
    mae=np.mean(abs(e)); rmse=np.sqrt(np.mean(e**2))
    mape=np.nanmean(abs(e)/np.where(abs(a)<1e-9,np.nan,abs(a)))*100
    smape=np.mean(np.where(abs(a)+abs(p)==0,0,2*abs(e)/(abs(a)+abs(p))))*100
    return {k:round(float(v),4) for k,v in {"mae":mae,"rmse":rmse,"mape":mape,"smape":smape}.items()}

class SeasonalNaive:
    def __init__(self,period=7): self.period=period; self.history=[]
    def fit(self,v): self.history=list(map(float,v)); return self
    def predict(self,h): return np.array([self.history[-self.period+(i%self.period)] for i in range(h)])

class GradientLag:
    def __init__(self): self.model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.06,random_state=42); self.history=[]
    def row(self,h): return [h[-1],h[-2],h[-7],h[-14],np.mean(h[-7:]),np.std(h[-7:])]
    def fit(self,v):
        h=list(map(float,v)); X=[]; y=[]
        for i in range(14,len(h)): X.append(self.row(h[:i])); y.append(h[i])
        self.model.fit(X,y); self.history=h; return self
    def predict(self,n):
        h=self.history.copy(); out=[]
        for _ in range(n):
            x=float(self.model.predict([self.row(h)])[0]); h.append(x); out.append(x)
        return np.array(out)

def train_compare(path,date_col,target_col,horizon,period):
    df=pd.read_csv(path)
    if date_col not in df or target_col not in df: raise ValueError("required columns missing")
    df=df[[date_col,target_col]].copy(); df[date_col]=pd.to_datetime(df[date_col],errors="raise")
    df[target_col]=pd.to_numeric(df[target_col],errors="raise"); df=df.dropna().sort_values(date_col).drop_duplicates(date_col)
    if len(df)<=horizon+14: raise ValueError("not enough observations")
    v=df[target_col].tolist(); train=v[:-horizon]; actual=v[-horizon:]
    factories={"seasonal_naive":lambda:SeasonalNaive(period),"gradient_lag":GradientLag}
    comparison={}
    for name,f in factories.items(): comparison[name]=metrics(actual,f().fit(train).predict(horizon))
    winner=min(comparison,key=lambda x:comparison[x]["mae"])
    return factories[winner]().fit(v),winner,comparison,len(df)
