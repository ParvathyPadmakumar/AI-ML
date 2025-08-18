# Burnout Breakdown

## Checking for missing values

--df.info(): To find missing values
The number of rows is found to be 3000 and by running df.info() each column has 3000 non-null values. Hence no missing values.

--df.describe(): to understand data dispersion

--df.isnull().sum().sort_values(ascending=False): the count of missing values per column.Here,0.

## Replace NaN with median for numeric data(Here,no change)

--The median is less affected by extreme values (outliers) compared to the mean for numeric data

## Replace with mean for categorical columns(Here,no change)

--The mode will replace with most seen value in the category for even distribution.

## Remove outliers(First,check boxplot then by IQR method)

--'PhysicalActivityHrs', 'CommuteTime', and 'TeamSize' have outliers in boxplot method
-- By IQR,no outliers were found and removed.

## Exploratory Data Analysis  

## Points to remember

-- warnings.filterwarnings("ignore"): This line suppresses any warning messages that might be generated during the execution of the code.

-- df.head(): This line displays the first 5 rows of the DataFrame df

-- The .mode() method can return multiple values if there's a tie, so [0] is used to select the first mode in case of multiple modes