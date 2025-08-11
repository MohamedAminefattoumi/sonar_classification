# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

#Importing the libraries
import pandas as pd 


#Importing the dataset 
dataset = pd.read_csv ('Sonar_data.csv' , header = None)

#Shape of data
print(f"Shape of data (rows,columns): {dataset.shape}")

#Description -> statistical measures of data
print("\n Statistical summary of the dataset: ")
print(dataset.describe ())

# Display the number of samples per class (column 60 is the target variable) with M --> Mine And R ---> Rock
print("\n Class distribution (target variable):")
print(dataset[60].value_counts())


# Calculate and display the mean of each feature grouped by the target class (column 60)
print("\n📈 Mean values of features grouped by class:")
print(dataset.groupby(60).mean())