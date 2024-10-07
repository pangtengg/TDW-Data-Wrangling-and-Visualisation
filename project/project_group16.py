#GROUP 16
#1211112369 ANIS SYIFAA' BT MOHD ZAFFARIN
#1211112304 KUEH PANG TENG
#1211112312 NUR INSYIRAH IMAN BT MOHD AZMAN
#1211111880 SOFIA BATRISYIA BT MOHAMAD FARIS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

#PART 3: DATA WRANGLING
# POINT 1: Load the dataset into a data frame using Pandas.
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Group_16\\project_group16\\Sleep_health_and_lifestyle_dataset.csv")

print("The Dataset of Sleep, Health and Lifestyle\n")
print(df)

# POINT 2: Explore the number of rows and columns, ranges of values, etc. 
rows, columns = df.shape
rows_columns = "\nThe dataset has %d rows and %d columns." % (rows, columns)
print(rows_columns)
print()
#display the first 5 rows by defaultdf.head()
#display the last 8 rows
df.tail(8)
#select specific multiple rows
print(df.iloc[[27,29]])
#select rows from specific columns, Sleep Disorder
print("\nRow from Sleep Disorder column:")
print(df['Sleep Disorder'][4])
#to display specific columns only
print("\nSpecific Columns:")
print(df[['BMI Category','Sleep Disorder']])
print()
#check data types
print(df.info())
#display the min and max values of specific column
print("\nThe min and max value of Daily Steps column:")
print(df['Daily Steps'].agg(['min', 'max']))

# POINT 3: Apply data wrangling techniques that you have learnt to handle missing, incorrect, and invalid data
#identifying missing data
print("\nThe missing values of each column:")
print(df.isnull().sum())
print('\nThe array in the Sleep Disorder column:')
print(df['Sleep Disorder'].unique())
#correcting missing data
df['Sleep Disorder'] = df['Sleep Disorder'].replace(['None'], np.nan)
print('\nSleep Disorder Updated Column:')
print(df['Sleep Disorder'])
#BMI category, normal weight to normal
print("\nThe BMI Category column:")
print(df['BMI Category'].iloc[[16, 17, 18]])
df['BMI Category'] = df['BMI Category'].replace(['Normal Weight'], 'Normal')
print("\nThe standardized BMI Category column:")
print(df['BMI Category'].iloc[[16, 17, 18]])
# Corrected use of duplicated() method
duplicates = df.duplicated(subset=['Gender', 'Age', 'Occupation', 'Sleep Duration', 
                                   'Quality of Sleep', 'Physical Activity Level',                                    'Stress Level', 'BMI Category', 'Blood Pressure', 
                                   'Heart Rate', 'Daily Steps', 'Sleep Disorder'], keep=False)
# Print the duplicated rows
print("\nThe duplicated rows:")
print(df[duplicates])
#drop the duplicate rows, and keep the first row
df_updated = df.drop_duplicates(subset=['Gender', 'Age', 'Occupation', 'Sleep Duration', 
                                   'Quality of Sleep', 'Physical Activity Level',                                    'Stress Level', 'BMI Category', 'Blood Pressure', 
                                   'Heart Rate', 'Daily Steps', 'Sleep Disorder'], keep='first')
print('\nThe updated dataframe:')
print(df_updated)
# POINT 4: Perform any additional steps (e.g., parsing dates, creating additional columns, merging multiple datasets, etc.)
# Creating a sleep efficiency score as a product of sleep duration and qualitydf_updated['Sleep Efficiency'] = ((df_updated['Sleep Duration'] * df_updated['Quality of Sleep']) / 100).apply(lambda x: '%.2f' % x) 
print("\nThe updated dataframe with new column:")
print(df_updated)

#PART 4: EDA
#calculating the range of numerical columns and format them to 2 significant figures
columns = ["Age","Sleep Duration", "Quality of Sleep", "Physical Activity Level", 
                       "Stress Level", "Heart Rate", "Daily Steps"]
maximum = df[columns].max()
minimum = df[columns].min()
range= maximum - minimum
formatted_range = range[columns].apply(lambda x: '%.2f' % x)
print(f"The range of each column is:\n{formatted_range.values}\n")

#calculating Q1 and Q3 of numerical columns and format them to 2 significant figures
Q1=df[columns].quantile(0.25)
Q3=df[columns].quantile(0.75)
formatted_Q1 = Q1[columns].apply(lambda x: '%.2f' % x)
print(f"The Q1 of each column is:\n{formatted_Q1.values}\n")
formatted_Q3 = Q3[columns].apply(lambda x: '%.2f' % x)
print(f"The Q3 of each column is:\n{formatted_Q3.values}\n")

#calculating the median of numerical columns and format them to 2 significant figures
median = df[columns].median()
formatted_median = median[columns].apply(lambda x: '%.2f' % x)
print(f"The median of each column is:\n{formatted_median.values}\n")

#calculating mode of numerical columns and format them to 2 significant figures
mode = df[columns].mode()
formatted_mode = mode[columns].apply(lambda x: '%.2f' % x)
print(f"The mode of each column is:\n{formatted_mode.values}\n")

# calculating means of numerical columns and format them to 2 significant figures
means = df[columns].mean()
formatted_means = means[columns].apply(lambda x: '%.2f' % x)
print(f"The means of each column is:\n{formatted_means.values}\n")

#calculating standard deviation of numerical columns and format them to 2 significant figures
std_dev=df[columns].std()
formatted_std_dev = std_dev[columns].apply(lambda x: '%.2f' % x)
print(f"The standard deviation of each column is:\n{formatted_std_dev.values}\n")

#calculating the skewness of numerical columns to check the distribution and format them to 2 significant figures
skewness = df[columns].skew()
formatted_skewness = skewness[columns].apply(lambda x: '%.2f' % x)
print(f"The skewness of each column is:\n{formatted_skewness.values}\n")


#PART 5: VISUALIZATION
#Histogram
df[columns].hist(figsize=(12, 10))
print()

#Bar plot to show distribution of BMI
plt.hist(df['BMI Category'], histtype='bar', ec='black')
plt.xlabel('BMI Category')
plt.ylabel('Frequency')
plt.show()
df['BMI Category'].value_counts()

#Bar plot for gender distribution
plt.hist(df['Gender'], histtype='bar', ec='black')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Gender Distribution in This Dataset')
plt.show()
df['Gender'].value_counts()

#Binning daily steps and presenting using bar plot
bins = [0, 5000, 7500, 10000]
labels = ['sedentary', 'low active', 'active']
df['Steps Binned']=pd.cut(df['Daily Steps'], bins=bins, labels=labels)

sns.countplot(x='Steps Binned', data=df)
plt.xlabel('Activity Level')
plt.ylabel('Number of People')
plt.title('Distribution of Daily Steps Categories')
plt.show()

#Heatmap to show correlation between 2 variables plotted on each axis
filtered = df.drop(['Person ID'], axis=1)  
cor = filtered.select_dtypes(include='number').corr()
plt.figure(figure=(6,3))
sns.heatmap(cor, annot=True)
plt.title('Correlation Heatmap')
plt.show()

#Boxplot to show sleep duration for male and female
plt.figure(figsize=(5, 4))
sns.boxplot(x='Gender', y='Sleep Duration', data=df)
plt.title('Box Plot of Sleep Duration by Gender')
plt.show()

#Scatter plot to show relationshi between sleep duration and sleep quality
sns.scatterplot(x='Sleep Duration',y='Quality of Sleep', data=df)
plt.title('Relationship Between Sleep Duration & Quality of Sleep')
plt.ylabel('Sleep Duration')
plt.xlabel('Quality of sleep')
plt.show()

#Scatterplot to show correlation between daily steps and physiczl activity level
sns.scatterplot(x='Physical Activity Level',y='Daily Steps', data=df_updated)
plt.title('Relationship Between Physical Activity Level and Daily Steps')
plt.ylabel('Physical Activity Level')
plt.xlabel('Daily Steps')
plt.show()

#Pie chart and count plot for occupation
occupation_counts=df['Occupation'].value_counts()

plt.bar(occupation_counts.index, occupation_counts.values, edgecolor='black')
plt.xticks(rotation=60, ha='right')
plt.xlabel('Occupation')
plt.ylabel('Frequency')
for idx, value in enumerate(occupation_counts.values):
    plt.text(idx, value, str(value), ha='center', va='bottom', fontsize=10)
plt.show()

plt.figure(figsize=(6, 6)) 
plt.pie(occupation_counts, labels=occupation_counts.index, autopct='%1.1f%%',  startangle=90)
plt.title('Occupation Distribution Pie Chart')
plt.show()

#DISCUSSION
# Does BMI influence ones’ sleep disorder?
sns.countplot(x='BMI Category', hue='Sleep Disorder', data=df)
plt.title('Relationship between BMI Categories and Sleep Disorder')
plt.show()

#How do BMI categories correlate with blood pressure and heart rate?
# Extract systolic value
df['Systolic_BP'] = df['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
heatmap_mean = df.groupby('BMI Category').agg({
    'Systolic_BP': 'mean',  # Mean of Systolic Blood Pressure
    'Heart Rate': 'mean'    # Mean of Heart Rate
}).reset_index()
print(heatmap_mean)

heatmap_mean['Systolic_BP'] = heatmap_mean['Systolic_BP'].round().astype(int)
heatmap_mean['Heart Rate'] = heatmap_mean['Heart Rate'].round().astype(int)

# heatmap
sns.heatmap(heatmap_mean.set_index('BMI Category').T, annot=True, fmt="d", cmap='coolwarm', cbar=True)
plt.title('Mean Systolic BP and Heart Rate by BMI Category')
plt.xlabel('BMI Category')
plt.ylabel('Mean Values')
plt.show()

#Are there any correlation between sleep efficiency and stress level?
sns.lineplot(x='Stress Level', y='Sleep Efficiency', data=df_updated)
plt.title('Relationship Between Sleep Efficiency and Stress Level')
plt.ylabel('Sleep Efficiency')
plt.xlabel('Stress Level')
plt.show()

#Which accoupation experiences the highest and lowest quality of sleep?
# dropdown menu for selecting an Occupation
input_dropdown = alt.binding_select(options=list(set(df.Occupation)))

selected_points = alt.selection_point(fields=['Occupation'], bind=input_dropdown, name='Select')

color = alt.condition(selected_points, alt.Color('Occupation:N'), alt.value('lightgray'))

alt.Chart(df).mark_circle().encode(
    x='Sleep Duration',
    y='Quality of Sleep',
    color=color,
    tooltip=['Gender','Age', 'Occupation:N', 'Sleep Duration:Q', 'Quality of Sleep:Q', 'Sleep Disorder']
).add_params(
    selected_points
).interactive()