## PROJECT SCOPE


1. Project Title

Analyzing the Relationship Between GDP and Life Expectancy Across Six Countries



2. Project Purpose

To identify and understand how Gross Domestic Product (GDP) relates to life expectancy across six selected countries by analyzing historical data from the World Health Organization (WHO) and the World Bank.



3. Objectives

The project aims to:

Preprocess and clean GDP and life expectancy datasets.

Explore and analyze trends over time for each country.

Visualize data using Matplotlib and Seaborn.

Evaluate whether GDP is correlated with life expectancy.

Present findings in a blog-style post suitable for publication on the WHO website.



Data sources

- GDP Source: [World Bank](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD) national accounts data, and OECD National Accounts data files.

- Life expectancy Data Source: [World Health Organization](http://apps.who.int/gho/data/node.main.688)



### Here are the core questions this project is designed to answer. These match the objectives of the GDP–Life Expectancy analysis and the expectations of the project.



Primary Research Question

1. What is the relationship between a country's GDP and its life expectancy?




Secondary Analysis Questions


2. How has life expectancy changed over time in each of the six countries?

3. How has GDP changed over time in each of the six countries?

4. Do countries with higher GDP tend to have higher life expectancy? (Investigated with scatterplots and regression.)

5. Are the trends consistent across all six countries, or do some behave differently?

6. What patterns emerge when comparing developing vs developed countries?

7. Are increases in GDP associated with increases in life expectancy year-over-year?

8. Is there evidence of diminishing returns? (i.e., does life expectancy level off at high GDP levels?)

9. What similarities and differences exist between the six countries in terms of


- life expectancy growth
- GDP growth
- rate of change
- overall health outcomes






Data Quality / Preparation Questions

These are addressed indirectly during cleaning:

10. Are there missing values or inconsistencies in the dataset?

11. Are the variables properly formatted for analysis (numeric, categorical)?

12. Is the dataset balanced across countries and years?






Final Interpretation / Insight Questions


13. What factors beyond GDP might explain differences in life expectancy?

14. What does the relationship imply about global development and public health?

15. Which countries improved the most—and why might that be?


## Importing the libraries


```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

```

## laoding and observing the data


```python
df = pd.read_csv('all_data.csv')
```


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Year</th>
      <th>Life expectancy at birth (years)</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chile</td>
      <td>2000</td>
      <td>77.3</td>
      <td>7.786093e+10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chile</td>
      <td>2001</td>
      <td>77.3</td>
      <td>7.097992e+10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chile</td>
      <td>2002</td>
      <td>77.8</td>
      <td>6.973681e+10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Chile</td>
      <td>2003</td>
      <td>77.9</td>
      <td>7.564346e+10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chile</td>
      <td>2004</td>
      <td>78.0</td>
      <td>9.921039e+10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chile</td>
      <td>2005</td>
      <td>78.4</td>
      <td>1.229650e+11</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Chile</td>
      <td>2006</td>
      <td>78.9</td>
      <td>1.547880e+11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Chile</td>
      <td>2007</td>
      <td>78.9</td>
      <td>1.736060e+11</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Chile</td>
      <td>2008</td>
      <td>79.6</td>
      <td>1.796380e+11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Chile</td>
      <td>2009</td>
      <td>79.3</td>
      <td>1.723890e+11</td>
    </tr>
  </tbody>
</table>
</div>



## checking the shape and necessary info about the data


```python
df.shape
```




    (96, 4)




```python
# information about the data
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 96 entries, 0 to 95
    Data columns (total 4 columns):
     #   Column                            Non-Null Count  Dtype  
    ---  ------                            --------------  -----  
     0   Country                           96 non-null     object 
     1   Year                              96 non-null     int64  
     2   Life expectancy at birth (years)  96 non-null     float64
     3   GDP                               96 non-null     float64
    dtypes: float64(2), int64(1), object(1)
    memory usage: 3.1+ KB
    

### Exploring the data to see the countries and years that are represented


```python
print(df.Country.unique())
```

    ['Chile' 'China' 'Germany' 'Mexico' 'United States of America' 'Zimbabwe']
    


```python
print(df.Country.unique())
```

    ['Chile' 'China' 'Germany' 'Mexico' 'United States of America' 'Zimbabwe']
    

### Cleaning Column Name

Looking over the data, there are inconsistencies with the column names. For example, the first two column names are one word each, while the third is five words long! Life expectancy at birth (years) is descriptive, which will be good for labeling the axis, but a little difficult to wrangle for coding the plot itself. The rename function is used to change the column name to lE meaning life expectancy.


```python
df = df.rename({"Life expectancy at birth (years)":"LEXBY"}, axis = "columns")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Year</th>
      <th>LEXBY</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chile</td>
      <td>2000</td>
      <td>77.3</td>
      <td>7.786093e+10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chile</td>
      <td>2001</td>
      <td>77.3</td>
      <td>7.097992e+10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chile</td>
      <td>2002</td>
      <td>77.8</td>
      <td>6.973681e+10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Chile</td>
      <td>2003</td>
      <td>77.9</td>
      <td>7.564346e+10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chile</td>
      <td>2004</td>
      <td>78.0</td>
      <td>9.921039e+10</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>LEXBY</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>96.000000</td>
      <td>96.000000</td>
      <td>9.600000e+01</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2007.500000</td>
      <td>72.789583</td>
      <td>3.880499e+12</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.633971</td>
      <td>10.672882</td>
      <td>5.197561e+12</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2000.000000</td>
      <td>44.300000</td>
      <td>4.415703e+09</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2003.750000</td>
      <td>74.475000</td>
      <td>1.733018e+11</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2007.500000</td>
      <td>76.750000</td>
      <td>1.280220e+12</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2011.250000</td>
      <td>78.900000</td>
      <td>4.067510e+12</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2015.000000</td>
      <td>81.000000</td>
      <td>1.810000e+13</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check missing values
print("Missing values per column:")
print(df.isna().sum(), "\n")
```

    Missing values per column:
    Country    0
    Year       0
    LEXBY      0
    GDP        0
    dtype: int64 
    
    

## Exploratory Plots


Visualizing data can often reveal patterns more effectively than tables or summary statistics. For example, the GDP distribution shown below is heavily right-skewed, with most values clustered toward the lower end. This shape is characteristic of a power-law distribution, a well-known pattern observed in many real-world phenomena. You can read more about power-law behavior here.


```python
plt.figure(figsize=(8,6))
sns.distplot(df.GDP, rug = True, kde=False)
plt.xlabel("GDP in Trillions of U.S. Dollars");
```


    
![png](output_25_0.png)
    


Next the distribution of LEXBY was examined. The distribution of LEXBY in the data is very left skewed where most of the values are on the right-hand side. This is almost the opposite of what was observed in the GDP column. A further look might also identify different modes or smaller groupings of distributions within the range.


```python
plt.figure(figsize=(8,6))
sns.distplot(df.LEXBY, rug = True, kde=False)
plt.xlabel("Life expectancy at birth (years)");
```


    
![png](output_27_0.png)
    


The previous plots did not break up the data by countries, so the next task will be to find the average LEXBY and GDP by country.


```python
dfMeans = df.drop("Year", axis = 1).groupby("Country").mean().reset_index()
```


```python
dfMeans
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>LEXBY</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chile</td>
      <td>78.94375</td>
      <td>1.697888e+11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>China</td>
      <td>74.26250</td>
      <td>4.957714e+12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Germany</td>
      <td>79.65625</td>
      <td>3.094776e+12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mexico</td>
      <td>75.71875</td>
      <td>9.766506e+11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>United States of America</td>
      <td>78.06250</td>
      <td>1.407500e+13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Zimbabwe</td>
      <td>50.09375</td>
      <td>9.062580e+09</td>
    </tr>
  </tbody>
</table>
</div>



With the data now grouped by country and the average values for LEXBY and GDP calculated, we can visualize these means using bar plots.

The first plot shows the mean Life Expectancy (LEXBY). All countries—except Zimbabwe—have average life expectancies in the mid-to-high 70s, which likely accounts for the skew observed in the earlier distribution.


```python
plt.figure(figsize=(8,6))
sns.barplot(x="LEXBY", y="Country", data=dfMeans)
plt.xlabel("Life expectancy at birth (years)");
```


    
![png](output_32_0.png)
    


Looking at the average GDP by country, the United States stands out with a value far higher than the others. In fact, Zimbabwe doesn’t appear on the bar plot at this scale, and Chile is barely visible. By comparison, China, Germany, and Mexico show GDP levels that are much closer to one another, though still well below that of the U.S.


```python
plt.figure(figsize=(8,6))
sns.barplot(x="GDP", y="Country", data=dfMeans)
plt.xlabel("GDP in Trillions of U.S. Dollars");
```


    
![png](output_34_0.png)
    


### Violin Plots

A useful way to compare data is by visualizing the distribution of each variable and examining the shapes for patterns.

A violin plot is especially helpful because it displays both the distribution shape and summary statistics, offering more context than a box plot alone. In the plots below, country appears on the x-axis, while the distributions of the numeric variables—GDP and LEXBY—are shown on the y-axis.

In the GDP violin plot (left), China and the United States exhibit relatively wide ranges, whereas Zimbabwe, Chile, and Mexico show much narrower spreads.

In the LEXBY violin plot, most countries have tight distributions, except for Zimbabwe, which spans a much larger range—from the high 30s to the high 60s—highlighting greater variability in life expectancy.


```python
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
axes[0] = sns.violinplot(ax=axes[0], x=df.GDP, y=df.Country)
axes[0].set_xlabel("GDP in Trillions of U.S. Dollars")
axes[1] = sns.violinplot(ax=axes[1], x=df.LEXBY, y=df.Country)
axes[1].set_xlabel("Life expectancy at birth (years)");
```


    
![png](output_37_0.png)
    


### Line Charts


Next the data will explore GDP and LEXBY over the years through line charts. Below the countries are separated by colors and one can see that the US and China have seen substantial gains between 2000-2015. China went from less than a quarter trillion dollars to one trillion dollars in the time span. The rest of the countries did not see increases in this magnitude.


```python
plt.figure(figsize=(8,6))
sns.lineplot(x=df.Year, y=df.GDP, hue=df.Country)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.ylabel("GDP in Trillions of U.S. Dollars");
```


    
![png](output_40_0.png)
    


Another element explored in greater depth was the use of faceted line charts by country. In these plots, each country is displayed with its own y-axis, making it much easier to compare the patterns of GDP over time without being constrained to a single scale.

This approach reveals that all countries have experienced growth since 2000. In the earlier combined chart, the increases for smaller economies appeared modest next to China and the United States. However, when viewed on individual scales, the upward trends for every country become much more apparent.


```python
graphGDP = sns.FacetGrid(df, col="Country", col_wrap=3,
                      hue = "Country", sharey = False)

graphGDP = (graphGDP.map(sns.lineplot,"Year","GDP")
         .add_legend()
         .set_axis_labels("Year","GDP in Trillions of U.S. Dollars"))

graphGDP;
```


    
![png](output_42_0.png)
    


The chart below highlights how life expectancy has progressed over the years. Although every country shows an upward trend, Zimbabwe stands out with the largest overall increase, particularly after a sharp drop around 2004.


```python
plt.figure(figsize=(8,6))
sns.lineplot(x=df.Year, y=df.LEXBY, hue=df.Country)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.ylabel("Life expectancy at birth (years)");

```


    
![png](output_44_0.png)
    


Similar to the earlier breakdown of GDP by country, the plot below separates life expectancy trends for each nation. Notably, both Chile and Mexico show dips in life expectancy around the same period—an observation that could merit further investigation. This type of visualization is valuable because many of these subtleties were obscured when the countries shared a single y-axis. It also reveals that what previously appeared to be smooth, linear changes were in fact more uneven for some countries.


```python
graphLEXBY = sns.FacetGrid(df, col="Country", col_wrap=3,
                      hue = "Country", sharey = False)

graphLEXBY = (graphLEXBY.map(sns.lineplot,"Year","LEXBY")
         .add_legend()
         .set_axis_labels("Year","Life expectancy at birth (years)"))

graphLEXBY;
```


    
![png](output_46_0.png)
    


### Scatter Plot

The next two charts examine the relationship between GDP and life expectancy (LEXBY). In the scatter plot below, Zimbabwe stands out: its GDP remains relatively flat while its life expectancy continues to rise. In contrast, the other countries show increases in life expectancy that generally track with rising GDP. The United States and China, in particular, display notably similar slopes in the relationship between GDP and life expectancy.


```python
sns.scatterplot(x=df.LEXBY, y=df.GDP, hue=df.Country).legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1);
```


    
![png](output_49_0.png)
    


As with the previous visualizations, each country is displayed in its own faceted scatter plot. Examining them individually shows that nations like the United States, Mexico, and Zimbabwe follow an almost linear relationship between GDP and life expectancy. China’s pattern leans toward a modest exponential curve, while Chile’s trend resembles a logarithmic shape. Despite these differences, the general pattern remains clear: higher GDP is associated with higher life expectancy


```python
graph = sns.FacetGrid(df, col="Country", col_wrap=3,
                      hue = "Country", sharey = False, sharex = False)
graph = (graph.map(sns.scatterplot,"LEXBY", "GDP")
         .add_legend()
         .set_axis_labels("Life expectancy at birth (years)", "GDP in Trillions of U.S. Dollars"));

```


    
![png](output_51_0.png)
    


### Conclusions



Even with a compact dataset, the analysis was able to uncover several meaningful insights about economic growth and public health across the six countries studied. The visualizations helped reveal patterns that would have been difficult to see from the raw numbers alone.

Key takeaways include:

Life expectancy trends:
All six countries show long-term improvements in life expectancy. Zimbabwe’s trajectory is especially striking, given how sharply it rebounds after its mid-2000s decline.

GDP patterns:
Economic output rises steadily for every country in the dataset, with China showing exceptionally strong growth compared with the others.

GDP–life expectancy relationship:
The scatter plots indicate a clear positive association: nations with higher GDP levels also tend to report longer life expectancy, though the shape of this relationship varies by country.

Typical life expectancy levels:
Most of the countries fall within the mid- to upper-70s range on average, whereas Zimbabwe consistently sits much lower.

Distribution shape:
Life expectancy values cluster toward the higher end, creating a distribution with a long tail on the lower side.






### Further Research



One promising direction for additional exploration is understanding the underlying forces behind China’s rapid economic expansion. External sources attribute this growth to factors such as large-scale industrial capacity, expanded access to investment capital, and the productive advantages of its sizable workforce. These explanations are consistent with the patterns observed in the dataset, suggesting that the visualizations are capturing real-world dynamics rather than statistical noise.
