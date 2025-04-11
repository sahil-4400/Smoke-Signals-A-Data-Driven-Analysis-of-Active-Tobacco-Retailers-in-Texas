import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as cb
from scipy import stats as st

df=pd.read_csv("E:\\lovely\\Document-4 GN 14\\INT375-DATA SCIENCE TOOLBOX PYTHON PROGRAMMING\\ca\\ca 2\\Active_Cigarette_Tobacco_Retailers.csv")
print(df)
print("############")



#knowing the data
print(df.head())
print("############")
print(df.describe())
print("############")
print(df.info())
print("############")
print(df.nunique())
print("############")
print(df.columns)
print("############")




#handling missing data
print(df.isnull().sum())
print("############")
print(df.dropna())
print("############")




#correlation
corre=df.corr(numeric_only=True)
print(corre)
print("############")
cb.heatmap(corre,cmap="coolwarm",annot=True,linewidths=0.5,fmt=".2f")
plt.xticks(rotation=45)
plt.show()





#covariance
cov=df.cov(numeric_only=True)
print(cov)
print("############")



#outliers
num=df.select_dtypes(include=['number']).columns
q1=df[num].quantile(0.25)#25% of your data
print(q1)
print("############")
q3=df[num].quantile(0.75)
print(q3)
print("############")
iqr=q3-q1
print(iqr)
print("##########")
lb=q1-1.5*iqr
print(lb)
print("############")
ub=q3+1.5*iqr
print(ub)
print("############")
out=(df[(df[num]<lb)|(df[num]>ub)]).sum()#sum of outliers in each column
print(out)
print("############")
cb.boxplot(df[num])
plt.xticks(rotation=45)
plt.show()
score=st.zscore(df[num],nan_policy='omit')#zscore is used to find the outliers in the data
out1=(abs(score>3)).sum(axis=0)#sum of outliers in each column
print(out1)
print("############")





#Objective 1: Number of Retailers by Permit Type (Bar Plot)
#Goal: Show how many businesses are registered as 'E-CIG RETAILER' vs' CIG/TOB RETAILER.'
cb.countplot(y="Permit Type",hue="Permit Type", data=df, palette="viridis", legend=False)
plt.title("Number of Retailers by Permit Type")
plt.xlabel("Count")
plt.ylabel("Permit Type")
plt.show()




#Objective 2: Top 10 Cities with Most Tobacco Retailers (Bar Plot)
#Goal: Identify cities with the highest number of licensed retailers.
top_cities = df['City'].value_counts().nlargest(10)
cb.barplot(x=top_cities.values, y=top_cities.index, hue=top_cities.index, palette="mako", legend=False)
plt.title("Top 10 Cities by Number of Tobacco Retailers")
plt.xlabel("Number of Retailers")
plt.ylabel("City")
plt.show()




#Objective 3: Permit Trends Over Time (Line Plot)
#Goal: Analyze how many permits started each year.
df['Permit Begin Date'] = pd.to_datetime(df['Permit Begin Date'], errors='coerce')
df['Permit Year'] = df['Permit Begin Date'].dt.year

permits_by_year = df['Permit Year'].value_counts().sort_index()
cb.lineplot(x=permits_by_year.index, y=permits_by_year.values, marker="o", color="teal")
plt.title("Number of Permits Issued per Year")
plt.xlabel("Year")
plt.ylabel("Number of Permits")
plt.show()




#Objective 4: Retailers by County (Top 10) (Bar Plot)
#Goal: Visualize which counties have the most tobacco retailers.
top_counties = df['County'].value_counts().nlargest(10)
cb.barplot(x=top_counties.values, y=top_counties.index, hue=top_counties.index, palette="pastel", legend=False)
plt.title("Top 10 Counties by Number of Retailers")
plt.xlabel("Number of Retailers")
plt.ylabel("County")
plt.show()




#Objective 5: Top 10 ZIP Codes by Share of Retailers (Pie Chart)
#Goal: Show the distribution of tobacco retailers among the top 10 ZIP codes by count, emphasizing their share of the total.
top_zips = df['Zip'].value_counts().nlargest(10)

plt.figure(figsize=(8, 8))
plt.pie(top_zips.values, labels=top_zips.index.astype(str), autopct='%1.1f%%', startangle=140, colors=cb.color_palette("Set3"))
plt.title("Top 10 ZIP Codes by Retailer Share\n\n")
plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
plt.show()

