# EXNO-6-DS-DATA VISUALIZATION USING SEABORN LIBRARY

# Aim:
  To Perform Data Visualization using seaborn python library for the given datas.

# EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# Algorithm:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

# Coding and Output:
```python
import seaborn as sns
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[3,6,2,7,1]
sns.lineplot(x=x,y=y)
```

![image](https://github.com/user-attachments/assets/44b66d67-0a8b-4c54-8a7a-b39bbf8de003)


```python
df=sns.load_dataset("tips")
df
```
![image](https://github.com/user-attachments/assets/0dd920c6-5a64-4bf5-baa8-0c2415b98cef)



```python
sns.lineplot(x="total_bill",y="tip",data=df,hue="sex",linestyle='solid',legend="auto")
```
![image](https://github.com/user-attachments/assets/743531e2-6fe1-4d95-ad99-5b2e48060c7a)


```python
import seaborn as sns
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y1=[3,5,2,6,1]
y2=[1,6,4,3,8]
y3=[5,2,7,1,4]
sns.lineplot(x=x,y=y1)
sns.lineplot(x=x,y=y2)
sns.lineplot(x=x,y=y3)
plt.xlabel('X Label')
plt.ylabel('Y Label')
```
![image](https://github.com/user-attachments/assets/8bfb8b41-ff83-487a-b43c-450a9e40cba6)



```python
import seaborn as sns
import matplotlib.pyplot as plt
tips=sns.load_dataset('tips')
avg_total_bill=tips.groupby('day')['total_bill'].mean()
avg_tips=tips.groupby('day')['tip'].mean()
plt.figure(figsize=(8,6))
p1=plt.bar(avg_total_bill.index, avg_total_bill, label='Total Bill')
p2=plt.bar(avg_tips.index, avg_tips, bottom=avg_total_bill, label='Tip')
plt.xlabel('Day of the Week')
plt.ylabel('Amount')
plt.title('Average Total Bill and Tip by Day')
plt.legend()
```
![image](https://github.com/user-attachments/assets/1344d865-4e85-4d08-97c2-a399e8379245)



```python
avg_total_bill=tips.groupby('day')['total_bill'].mean()
avg_tips=tips.groupby('day')['tip'].mean()
p1=plt.bar(avg_total_bill.index,avg_total_bill, label='Total Bill',width=0.4)
p2=plt.bar(avg_tips.index,avg_tips,bottom=avg_total_bill, label='Tip',width=0.4)
plt.xlabel('Time of Day')
plt.ylabel('Amount')
plt.title('Average Total Bill and Tip by Day')
plt.legend()
```

![image](https://github.com/user-attachments/assets/78a2ebf1-6b8e-4e45-a8fd-def7c0efe16e)


```python
years=range(2000,2012)
apples=[0.895,0.91,0.919,0.926,0.929,0.931,0.934,0.937,0.9375,0.9372,0.939,0.9392]
oranges=[0.962,0.941,0.930,0.923,0.918,0.908,0.907,0.904,0.901,0.898,0.9,0.896]
plt.bar(years, apples)
plt.bar(years, oranges, bottom=apples)
```
![image](https://github.com/user-attachments/assets/9a3b0559-c9b9-44a6-8692-6613242ad94d)



```python
import seaborn as sns
dt= sns.load_dataset('tips')
sns.barplot(x='day', y='total_bill',hue='sex',data=dt,palette='Set1')
plt.xlabel('Day of te week')
plt.ylabel('Total Bill')
plt.title('Total Bill by Day and Gender')
```
![image](https://github.com/user-attachments/assets/cdb57fdb-efa0-442c-bb6e-cebabd192546)


```python
import pandas as pd
tit=pd.read_csv("titanic_dataset.csv")
tit
```
![image](https://github.com/user-attachments/assets/9461e16a-1565-46e4-a471-94073d7b94f1)


```python
plt.figure(figsize=(8,5))
sns.barplot(x='Embarked',y='Fare',data=tit,palette='rainbow')
plt.title("Fare of Passenger by Embarked Town")
```
![image](https://github.com/user-attachments/assets/b8233ca2-2d31-4206-b597-69eef46250b2)


```python
plt.figure(figsize=(8,5))
sns.barplot(x='Embarked',y='Fare',data=tit,palette='rainbow',hue='Pclass')
plt.title("Fare of Passenger by Embarked Town, Divided by Class")

```
![image](https://github.com/user-attachments/assets/e4385ab9-7021-4e8d-8a77-1155ee82bc93)



```python
import seaborn as sns
tips= sns.load_dataset('tips')
sns.scatterplot(x='total_bill',y = 'tip',hue='sex',data=tips)
plt.ylabel('Tip Amount')
plt.xlabel('Total Bill')
plt.title('Scatter Plot of Total Bill vs. Tip Amount')
```
![image](https://github.com/user-attachments/assets/babc02d9-105b-4b3d-bd83-d282e7c09950)


```python
import seaborn as sns
import numpy as np
import pandas as pd
np.random.seed(1)
num_var = np.random.randn(1000)
num_var = pd.Series(num_var, name="Numerical Variable")
num_var
```
![image](https://github.com/user-attachments/assets/65a6e965-52ba-4fc0-9a7d-f2eef6fa2023)



```python
sns.histplot(data = num_var, kde=True)
```
![image](https://github.com/user-attachments/assets/47f4fcf7-325b-4b1b-8c84-7415a379f624)



```python
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df
```
![image](https://github.com/user-attachments/assets/3fe2a5cb-2600-4ae8-9081-d047b54e2a5f)



```python
sns.histplot(data = df,x = "Pclass",hue="Survived", kde=True)
```
![image](https://github.com/user-attachments/assets/86c7c5a6-b4f4-4e18-b42b-9d03aaf44cfc)


```python
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)
marks = np.random.normal(loc=70, scale=10, size=100)
marks
```
![image](https://github.com/user-attachments/assets/e9b96ab4-5b28-4e52-9e29-c0d54713b374)


```python
sns.histplot(data = marks, bins=10,kde=True,stat='count', cumulative=False, multiple='stack',element='bars',palette='Set1',shrink=0.7)
plt.xlabel('Marks')
plt.ylabel('Density')
plt.title('Histogram of Students Marks')
```
![image](https://github.com/user-attachments/assets/23164764-5655-46b0-96a7-37825f381892)


```python
import seaborn as sns
import pandas as pd
tips=sns.load_dataset('tips')
sns.boxplot(x=tips['day'],y=tips['total_bill'],hue=tips['sex'])
```
![image](https://github.com/user-attachments/assets/09227c50-d0d5-4bd2-b105-85817fae7a07)



```python
sns.boxplot(x="day",y="total_bill",hue="smoker",data=tips,linewidth=2,width=0.6,boxprops={"facecolor":"lightblue","edgecolor":"darkblue"},whiskerprops={"color":"black","linestyle":"--","linewidth":1.5},capprops={"color":"black","linestyle":"--","linewidth":1.5})
```
![image](https://github.com/user-attachments/assets/3246a3c5-2063-472e-b214-bdaea00b794b)



```python
sns.boxplot(x='Pclass',y='Age',data=df,palette='rainbow')
plt.title("Age by Passenger Class,Titanic")
```
![image](https://github.com/user-attachments/assets/d1f4fdf3-276f-4eaa-b3cf-b836ec816905)


```python
sns.violinplot(x='day',y='total_bill',hue='smoker',data=tips,linewidth=2,width=0.6)

plt.xlabel("Day of the week")
plt.ylabel("Total Bill")
plt.title("Violin Pot of Total Bill by Day and Smoker status")
```
![image](https://github.com/user-attachments/assets/d3b3021a-88d4-4495-98b3-cc750c9b266c)


```python
import seaborn as sns

sns.set_theme(style='whitegrid')
tip = sns.load_dataset('tips')
sns.violinplot(x='day', y='tip', data=tip, palette="hsv")
```
![image](https://github.com/user-attachments/assets/517cd298-03f0-4a28-8c8c-f35ecf07488d)



```python
sns.set_theme(style='whitegrid')
tip=sns.load_dataset('tips')
sns.violinplot(x=tip["total_bill"])
```
![image](https://github.com/user-attachments/assets/197ecc11-7060-43f6-93df-e22e60d62cad)


```python
import seaborn as sns

sns.set_theme(style='whitegrid')
tip = sns.load_dataset('tips')
sns.violinplot(x='tip', y='day', data=tip, palette="Spectral")
```
![image](https://github.com/user-attachments/assets/c9c57fa6-c87d-4b95-8ee7-b96e5bc7bbee)


```python
sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='stack', linewidth=3,palette='colorblind',alpha=0.8)
```
![image](https://github.com/user-attachments/assets/c8db41ba-5dc0-4046-9168-adea835be973)



```python
sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='fill',linewidth=3,palette='dark',alpha=0.8)
```
![image](https://github.com/user-attachments/assets/fc12bfad-d6db-4265-855a-4fbfbb6a3cc2)


```python
sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='layer',linewidth=3,palette='crest',alpha=0.8)
```
![image](https://github.com/user-attachments/assets/ec1a6c40-6541-4b14-82c0-221ed52af158)


```python
tips=sns.load_dataset("tips")
numeric_cols=tips.select_dtypes(include=np.number).columns
corr=tips[numeric_cols].corr()
sns.heatmap(corr,annot=True,cmap="plasma",linewidth=0.5)
```
![image](https://github.com/user-attachments/assets/9ca5d90f-2d0c-4ab6-b67b-a50df79b2815)



```python
data=np.random.randint(low=1,high=100,size=(10,10))

hm=sns.heatmap(data=data,annot=True)
```

![image](https://github.com/user-attachments/assets/16aa86a3-71ce-4353-a9bd-c641c3162eda)


```python
sns.heatmap(data=data)
```
![image](https://github.com/user-attachments/assets/5680796d-f1ec-42d5-847c-95759f1a1e28)



# Supermarket.csv data:
```
mart=pd.read_csv("supermarket.csv")
mart
```



```python
mart=mart[['Gender','Payment','Unit price','Quantity','Total','gross income']]
mart.head(10)
```
![image](https://github.com/user-attachments/assets/61a8be16-2231-4785-86a7-38609e31695f)


```python
sns.kdeplot(data=mart,x='Total')
```
![image](https://github.com/user-attachments/assets/799b04a1-c630-4c53-92a3-1e9385a0ae96)



```python
sns.kdeplot(data=mart,x='Unit price')
```
![image](https://github.com/user-attachments/assets/cb72c476-e514-4dae-aa8e-5b2e269da5ff)


```python
sns.kdeplot(data=mart)
```
![image](https://github.com/user-attachments/assets/805ea828-8b7b-47eb-92e6-9d1a7abfb377)



```python
sns.kdeplot(data=mart,x='Total',hue='Payment',multiple='stack')
```
![image](https://github.com/user-attachments/assets/f0b9f365-c2ad-49a5-850d-a85fae57951e)


python
```
sns.kdeplot(data=mart,x='Total',hue='Payment',multiple='stack',linewidth=5,palette='Dark2',alpha=0.5)
```
![image](https://github.com/user-attachments/assets/94b4639c-b9ca-44bc-bc33-8dccf3977095)




```python
sns.kdeplot(data=mart,x='Unit price',y='gross income')
```
![image](https://github.com/user-attachments/assets/d2c94e50-b851-43f5-bf3a-7edf24395bd3)























# Result:
Thus, We have performed Data Visualization using seaborn python library for the given datas.


### Name : ASWIN B
### Register Number : 212224110007
