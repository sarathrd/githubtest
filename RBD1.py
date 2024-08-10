import pandas as pd;
import numpy as np
from scipy.stats import stats
import statsmodels.api as sm;
from statsmodels.formula.api import ols

data=pd.read_csv("rbd1.csv");
print(data);
d=pd.melt(data.reset_index(),id_vars=['index'],value_vars=['p','q','r','s']);

print(d);

mod=ols('value~C(index)+C(variable)',data=d).fit()
ano=sm.stats.anova_lm(mod,type=2)
print(ano)