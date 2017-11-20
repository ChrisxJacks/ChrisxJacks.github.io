
# Project 3 - Regression Challange 



```python
# Data Analysis & Wrangling 
import numpy as np 
import pandas as pd 

# Visualizations
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline


# Modeling and Learning 
from sklearn.linear_model import LinearRegression, SGDClassifier, Perceptron #Look back on this one as well 
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB # Look back on this one 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
```


```python
housing = pd.read_csv('./RegressionChallange/train.csv')
```


```python
housing.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>109</td>
      <td>533352170</td>
      <td>60</td>
      <td>RL</td>
      <td>NaN</td>
      <td>13517</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>130500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>544</td>
      <td>531379050</td>
      <td>60</td>
      <td>RL</td>
      <td>43.0</td>
      <td>11492</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>220000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>153</td>
      <td>535304180</td>
      <td>20</td>
      <td>RL</td>
      <td>68.0</td>
      <td>7922</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>109000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>318</td>
      <td>916386060</td>
      <td>60</td>
      <td>RL</td>
      <td>73.0</td>
      <td>9802</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>174000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>255</td>
      <td>906425045</td>
      <td>50</td>
      <td>RL</td>
      <td>82.0</td>
      <td>14235</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>138500</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>




```python
housing.shape
```




    (2051, 82)




```python
#Columns Columns
housing.describe(include=['O']).columns
```




    Index(['MS Zoning', 'Street', 'Lot Shape', 'Land Contour', 'Utilities',
           'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1',
           'Condition 2', 'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl',
           'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type', 'Exter Qual',
           'Exter Cond', 'Foundation', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
           'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating', 'Heating QC',
           'Central Air', 'Electrical', 'Kitchen Qual', 'Functional',
           'Garage Type', 'Garage Finish', 'Garage Cond', 'Paved Drive',
           'Sale Type', 'Sale Condition'],
          dtype='object')




```python
housing.iloc[:,:22].describe(include=['O'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MS Zoning</th>
      <th>Street</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>Utilities</th>
      <th>Lot Config</th>
      <th>Land Slope</th>
      <th>Neighborhood</th>
      <th>Condition 1</th>
      <th>Condition 2</th>
      <th>Bldg Type</th>
      <th>House Style</th>
      <th>Roof Style</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>28</td>
      <td>9</td>
      <td>8</td>
      <td>5</td>
      <td>8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>top</th>
      <td>RL</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>Gable</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1598</td>
      <td>2044</td>
      <td>1295</td>
      <td>1843</td>
      <td>2049</td>
      <td>1503</td>
      <td>1953</td>
      <td>310</td>
      <td>1767</td>
      <td>2025</td>
      <td>1700</td>
      <td>1059</td>
      <td>1619</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.iloc[:,22:40].describe(include=['O'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Roof Matl</th>
      <th>Exterior 1st</th>
      <th>Exterior 2nd</th>
      <th>Mas Vnr Type</th>
      <th>Exter Qual</th>
      <th>Exter Cond</th>
      <th>Foundation</th>
      <th>Bsmt Qual</th>
      <th>Bsmt Cond</th>
      <th>Bsmt Exposure</th>
      <th>BsmtFin Type 1</th>
      <th>BsmtFin Type 2</th>
      <th>Heating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2029</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>1996</td>
      <td>1996</td>
      <td>1993</td>
      <td>1996</td>
      <td>1995</td>
      <td>2051</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>6</td>
      <td>15</td>
      <td>15</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>6</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>top</th>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>Unf</td>
      <td>GasA</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>2025</td>
      <td>724</td>
      <td>721</td>
      <td>1218</td>
      <td>1247</td>
      <td>1778</td>
      <td>926</td>
      <td>887</td>
      <td>1834</td>
      <td>1339</td>
      <td>615</td>
      <td>1749</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.iloc[:,40:].describe(include=['O'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Heating QC</th>
      <th>Central Air</th>
      <th>Electrical</th>
      <th>Kitchen Qual</th>
      <th>Functional</th>
      <th>Garage Type</th>
      <th>Garage Finish</th>
      <th>Garage Cond</th>
      <th>Paved Drive</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
      <td>2051</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>8</td>
      <td>7</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>TA</td>
      <td>Typ</td>
      <td>Attchd</td>
      <td>Unf</td>
      <td>TA</td>
      <td>Y</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1065</td>
      <td>1910</td>
      <td>1868</td>
      <td>1047</td>
      <td>1915</td>
      <td>1213</td>
      <td>849</td>
      <td>1868</td>
      <td>1861</td>
      <td>1781</td>
      <td>1696</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Columns Containing Numerical Features 
housing.describe(exclude=['O']).columns
```




    Index(['Id', 'PID', 'MS SubClass', 'Lot Frontage', 'Lot Area', 'Overall Qual',
           'Overall Cond', 'Year Built', 'Year Remod/Add', 'Mas Vnr Area',
           'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF',
           '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area',
           'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath',
           'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces',
           'Garage Yr Blt', 'Garage Cars', 'Garage Area', 'Wood Deck SF',
           'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch',
           'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold', 'SalePrice',
           'Garage Qual Score', 'Garage Cond Score'],
          dtype='object')




```python
housing.iloc[:,:22].describe(exclude=['O'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Overall Qual</th>
      <th>Overall Cond</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2051.000000</td>
      <td>2.051000e+03</td>
      <td>2051.000000</td>
      <td>1721.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1474.033642</td>
      <td>7.135900e+08</td>
      <td>57.008776</td>
      <td>69.055200</td>
      <td>10065.208191</td>
      <td>6.112140</td>
      <td>5.562165</td>
      <td>1971.708922</td>
      <td>1984.190151</td>
    </tr>
    <tr>
      <th>std</th>
      <td>843.980841</td>
      <td>1.886918e+08</td>
      <td>42.824223</td>
      <td>23.260653</td>
      <td>6742.488909</td>
      <td>1.426271</td>
      <td>1.104497</td>
      <td>30.177889</td>
      <td>21.036250</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>5.263011e+08</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>753.500000</td>
      <td>5.284581e+08</td>
      <td>20.000000</td>
      <td>58.000000</td>
      <td>7500.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1953.500000</td>
      <td>1964.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1486.000000</td>
      <td>5.354532e+08</td>
      <td>50.000000</td>
      <td>68.000000</td>
      <td>9430.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1974.000000</td>
      <td>1993.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2198.000000</td>
      <td>9.071801e+08</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11513.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2001.000000</td>
      <td>2004.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2930.000000</td>
      <td>9.241520e+08</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>159000.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.iloc[:,22:47].describe(exclude=['O'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mas Vnr Area</th>
      <th>BsmtFin SF 1</th>
      <th>BsmtFin SF 2</th>
      <th>Bsmt Unf SF</th>
      <th>Total Bsmt SF</th>
      <th>1st Flr SF</th>
      <th>2nd Flr SF</th>
      <th>Low Qual Fin SF</th>
      <th>Gr Liv Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2029.000000</td>
      <td>2050.000000</td>
      <td>2050.000000</td>
      <td>2050.000000</td>
      <td>2050.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>99.695909</td>
      <td>442.300488</td>
      <td>47.959024</td>
      <td>567.728293</td>
      <td>1057.987805</td>
      <td>1164.488055</td>
      <td>329.329108</td>
      <td>5.512921</td>
      <td>1499.330083</td>
    </tr>
    <tr>
      <th>std</th>
      <td>174.963129</td>
      <td>461.204124</td>
      <td>165.000901</td>
      <td>444.954786</td>
      <td>449.410704</td>
      <td>396.446923</td>
      <td>425.671046</td>
      <td>51.068870</td>
      <td>500.447829</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>334.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>220.000000</td>
      <td>793.000000</td>
      <td>879.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1129.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>368.000000</td>
      <td>0.000000</td>
      <td>474.500000</td>
      <td>994.500000</td>
      <td>1093.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1444.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>161.000000</td>
      <td>733.750000</td>
      <td>0.000000</td>
      <td>811.000000</td>
      <td>1318.750000</td>
      <td>1405.000000</td>
      <td>692.500000</td>
      <td>0.000000</td>
      <td>1728.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>1474.000000</td>
      <td>2336.000000</td>
      <td>6110.000000</td>
      <td>5095.000000</td>
      <td>1862.000000</td>
      <td>1064.000000</td>
      <td>5642.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.iloc[:,47:59].describe(exclude=['O'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bsmt Full Bath</th>
      <th>Bsmt Half Bath</th>
      <th>Full Bath</th>
      <th>Half Bath</th>
      <th>Bedroom AbvGr</th>
      <th>Kitchen AbvGr</th>
      <th>TotRms AbvGrd</th>
      <th>Fireplaces</th>
      <th>Garage Yr Blt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2049.000000</td>
      <td>2049.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>1937.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.427526</td>
      <td>0.063446</td>
      <td>1.577279</td>
      <td>0.371039</td>
      <td>2.843491</td>
      <td>1.042906</td>
      <td>6.435885</td>
      <td>0.590931</td>
      <td>1978.707796</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.522673</td>
      <td>0.251705</td>
      <td>0.549279</td>
      <td>0.501043</td>
      <td>0.826618</td>
      <td>0.209790</td>
      <td>1.560225</td>
      <td>0.638516</td>
      <td>25.441094</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1895.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>1961.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>1980.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>2002.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>15.000000</td>
      <td>4.000000</td>
      <td>2207.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.iloc[:,59:71].describe(exclude=['O'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Wood Deck SF</th>
      <th>Open Porch SF</th>
      <th>Enclosed Porch</th>
      <th>3Ssn Porch</th>
      <th>Screen Porch</th>
      <th>Pool Area</th>
      <th>Misc Val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2050.000000</td>
      <td>2050.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.776585</td>
      <td>473.671707</td>
      <td>93.833740</td>
      <td>47.556802</td>
      <td>22.571916</td>
      <td>2.591419</td>
      <td>16.511458</td>
      <td>2.397855</td>
      <td>51.574354</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.764537</td>
      <td>215.934561</td>
      <td>128.549416</td>
      <td>66.747241</td>
      <td>59.845110</td>
      <td>25.229615</td>
      <td>57.374204</td>
      <td>37.782570</td>
      <td>573.393985</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>319.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>480.000000</td>
      <td>0.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>576.000000</td>
      <td>168.000000</td>
      <td>70.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>1418.000000</td>
      <td>1424.000000</td>
      <td>547.000000</td>
      <td>432.000000</td>
      <td>508.000000</td>
      <td>490.000000</td>
      <td>800.000000</td>
      <td>17000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.iloc[:,71:].describe(exclude=['O'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>SalePrice</th>
      <th>Garage Qual Score</th>
      <th>Garage Cond Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.219893</td>
      <td>2007.775719</td>
      <td>181469.701609</td>
      <td>2.803023</td>
      <td>2.810336</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.744736</td>
      <td>1.312014</td>
      <td>79258.659352</td>
      <td>0.721253</td>
      <td>0.716094</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>12789.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>2007.000000</td>
      <td>129825.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>162500.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>611657.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
#make all columnnames lower case and add a underscore between words 
```


```python
housing['Misc Feature'].value_counts()
```




    Shed    56
    Gar2     4
    Othr     3
    TenC     1
    Elev     1
    Name: Misc Feature, dtype: int64




```python
for col in housing:
    if housing[col].isnull().sum() > 999:
        del housing[col]
```


```python
housing.shape
```




    (2051, 77)




```python
housing.iloc[:,:41].isnull().sum()
```




    Id                  0
    PID                 0
    MS SubClass         0
    MS Zoning           0
    Lot Frontage      330
    Lot Area            0
    Street              0
    Lot Shape           0
    Land Contour        0
    Utilities           0
    Lot Config          0
    Land Slope          0
    Neighborhood        0
    Condition 1         0
    Condition 2         0
    Bldg Type           0
    House Style         0
    Overall Qual        0
    Overall Cond        0
    Year Built          0
    Year Remod/Add      0
    Roof Style          0
    Roof Matl           0
    Exterior 1st        0
    Exterior 2nd        0
    Mas Vnr Type       22
    Mas Vnr Area       22
    Exter Qual          0
    Exter Cond          0
    Foundation          0
    Bsmt Qual          55
    Bsmt Cond          55
    Bsmt Exposure      58
    BsmtFin Type 1     55
    BsmtFin SF 1        1
    BsmtFin Type 2     56
    BsmtFin SF 2        1
    Bsmt Unf SF         1
    Total Bsmt SF       1
    Heating             0
    Heating QC          0
    dtype: int64




```python
housing.iloc[:,41:].isnull().sum()
```




    Central Air          0
    Electrical           0
    1st Flr SF           0
    2nd Flr SF           0
    Low Qual Fin SF      0
    Gr Liv Area          0
    Bsmt Full Bath       2
    Bsmt Half Bath       2
    Full Bath            0
    Half Bath            0
    Bedroom AbvGr        0
    Kitchen AbvGr        0
    Kitchen Qual         0
    TotRms AbvGrd        0
    Functional           0
    Fireplaces           0
    Garage Type        113
    Garage Yr Blt      114
    Garage Finish      114
    Garage Cars          1
    Garage Area          1
    Garage Qual        114
    Garage Cond        114
    Paved Drive          0
    Wood Deck SF         0
    Open Porch SF        0
    Enclosed Porch       0
    3Ssn Porch           0
    Screen Porch         0
    Pool Area            0
    Misc Val             0
    Mo Sold              0
    Yr Sold              0
    Sale Type            0
    Sale Condition       0
    SalePrice            0
    dtype: int64




```python
housing['Garage Cond'].value_counts()
```




    TA    1868
    Fa      47
    Gd      12
    Po       8
    Ex       2
    Name: Garage Cond, dtype: int64




```python
housing['Garage Type'].fillna('None', inplace=True)
housing['Garage Qual'].fillna('None', inplace=True)
housing['Garage Cond'].fillna('None', inplace=True)
housing['Garage Finish'].fillna('None', inplace=True)
```


```python
housing['Garage Cond'].value_counts()
```




    TA      1868
    None     114
    Fa        47
    Gd        12
    Po         8
    Ex         2
    Name: Garage Cond, dtype: int64




```python
housing['Garage Qual Score'] = housing['Garage Qual'].map({'Ex':5,'Gd':4,'TA':3, 'Fa':2, 'Po':1, 'None':0})
housing['Garage Cond Score'] = housing['Garage Cond'].map({'Ex':5,'Gd':4,'TA':3, 'Fa':2, 'Po':1, 'None':0})
```


```python
housing['Garage Qual Score']
```




    0       3
    1       3
    2       3
    3       3
    4       3
    5       3
    6       3
    7       3
    8       3
    9       3
    10      3
    11      3
    12      3
    13      3
    14      3
    15      3
    16      2
    17      3
    18      3
    19      3
    20      3
    21      3
    22      3
    23      3
    24      3
    25      3
    26      3
    27      3
    28      0
    29      3
           ..
    2021    3
    2022    3
    2023    3
    2024    3
    2025    3
    2026    3
    2027    0
    2028    4
    2029    3
    2030    3
    2031    3
    2032    3
    2033    3
    2034    3
    2035    3
    2036    3
    2037    3
    2038    3
    2039    0
    2040    3
    2041    3
    2042    0
    2043    3
    2044    3
    2045    3
    2046    3
    2047    3
    2048    2
    2049    3
    2050    3
    Name: Garage Qual Score, Length: 2051, dtype: int64




```python
del housing['Garage Qual']

```


```python
#Set Garage Year Built to Year Built 
```


```python
housing.dtypes
```




    Id                     int64
    PID                    int64
    MS SubClass            int64
    MS Zoning             object
    Lot Frontage         float64
    Lot Area               int64
    Street                object
    Lot Shape             object
    Land Contour          object
    Utilities             object
    Lot Config            object
    Land Slope            object
    Neighborhood          object
    Condition 1           object
    Condition 2           object
    Bldg Type             object
    House Style           object
    Overall Qual           int64
    Overall Cond           int64
    Year Built             int64
    Year Remod/Add         int64
    Roof Style            object
    Roof Matl             object
    Exterior 1st          object
    Exterior 2nd          object
    Mas Vnr Type          object
    Mas Vnr Area         float64
    Exter Qual            object
    Exter Cond            object
    Foundation            object
                          ...   
    Bsmt Full Bath       float64
    Bsmt Half Bath       float64
    Full Bath              int64
    Half Bath              int64
    Bedroom AbvGr          int64
    Kitchen AbvGr          int64
    Kitchen Qual          object
    TotRms AbvGrd          int64
    Functional            object
    Fireplaces             int64
    Garage Type           object
    Garage Yr Blt        float64
    Garage Finish         object
    Garage Cars          float64
    Garage Area          float64
    Garage Cond           object
    Paved Drive           object
    Wood Deck SF           int64
    Open Porch SF          int64
    Enclosed Porch         int64
    3Ssn Porch             int64
    Screen Porch           int64
    Pool Area              int64
    Misc Val               int64
    Mo Sold                int64
    Yr Sold                int64
    Sale Type             object
    Sale Condition        object
    SalePrice              int64
    Garage Qual Score      int64
    Length: 77, dtype: object




```python

```


      File "<ipython-input-88-12e7bce94552>", line 1
        del
           ^
    SyntaxError: invalid syntax




```python
housing_c = housing._get_numeric_data().drop('SalePrice', axis=1).copy()
```


```python
housing_c = housing.dropna().copy()
```


```python
X = housing_c.drop(['Id', 'PID'], axis=1)
Y = housing_c['SalePrice']
```


```python

for i in housing:
    hmm = [float(x) for x in housing[i] if x != '']
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-242-5dc076071d87> in <module>()
          1 
          2 for i in housing:
    ----> 3     hmm = [float(x) for x in housing[i] if x != '']
    

    <ipython-input-242-5dc076071d87> in <listcomp>(.0)
          1 
          2 for i in housing:
    ----> 3     hmm = [float(x) for x in housing[i] if x != '']
    

    ValueError: could not convert string to float: 'RL'



```python
X = housing_c.drop(['MS Zoning'], axis=1).copy()
```


```python
X = housing_c.loc[:, housing_c.dtypes != object]
```


```python
X.dtypes
```




    Id                     int64
    PID                    int64
    MS SubClass            int64
    Lot Frontage         float64
    Lot Area               int64
    Overall Qual           int64
    Overall Cond           int64
    Year Built             int64
    Year Remod/Add         int64
    Mas Vnr Area         float64
    BsmtFin SF 1         float64
    BsmtFin SF 2         float64
    Bsmt Unf SF          float64
    Total Bsmt SF        float64
    1st Flr SF             int64
    2nd Flr SF             int64
    Low Qual Fin SF        int64
    Gr Liv Area            int64
    Bsmt Full Bath       float64
    Bsmt Half Bath       float64
    Full Bath              int64
    Half Bath              int64
    Bedroom AbvGr          int64
    Kitchen AbvGr          int64
    TotRms AbvGrd          int64
    Fireplaces             int64
    Garage Yr Blt        float64
    Garage Cars          float64
    Garage Area          float64
    Wood Deck SF           int64
    Open Porch SF          int64
    Enclosed Porch         int64
    3Ssn Porch             int64
    Screen Porch           int64
    Pool Area              int64
    Misc Val               int64
    Mo Sold                int64
    Yr Sold                int64
    SalePrice              int64
    Garage Qual Score      int64
    Garage Cond Score      int64
    dtype: object




```python
X.shape
```




    (1556, 41)




```python
Y.shape
```




    (1556,)




```python
logreg = LogisticRegression()
logreg.fit(X,Y)
y_pred = logreg.predict(X)
acc_log = round(logreg.score(X, Y) * 100, 2)
acc_log

```




    2.5099999999999998




```python

```


```python

```


```python

```
