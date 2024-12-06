
## Features and Performance Speed of Central Processing Units

I applied machine learning techniques to investigate the features of a central processing unit (CPU) and which feature affects the CPU performance the most. Below is my report.

***

## Introduction: What is a CPU and its Origins

Central Processing Units are really important and are the primary components of a computer that is responsible for executing instructions and performing calculations. The first central processing unit is developed in 1971 under the techonology of the Intel Company. Originally this was a task assigned by the Nippon Calculating Machine Corporation in 1969 to develop 12 custom chips for its new printing calculator. The first CPU had a set of four chips, a central processing unit chip, a supporting read-only memory chip (ROM) for the custom application programs, random-access memory (RAM) chip for processing data, and a final shift-register chip for the input/output (O/I) port. (See Reference [1])

Within a CPU, there is a control unit, an arithmetic and logic unit, andstorage (registers and memory). The central processing unit is the part of the CPU that helps assemble the execution of instructions. The ALU or arithemtic and logic unit is where all logical and arithemtic computations occur. The storage is seperated into two parts. A register where data is stored and a more common name to be known is the memory (RAM) where it is a collection of registers arranged and compact together to store a high number of data. All of these parts of the CPU affect the overall performance speed of the CPU. 



## Data
After learning about the basics and the origin of the CPU, 
   MYCT: machine cycle time in nanoseconds (integer)
   MMIN: minimum main memory in kilobytes (integer)
   MMAX: maximum main memory in kilobytes (integer)
   CACH: cache memory in kilobytes (integer)
   CHMIN: minimum channels in units (integer)
   CHMAX: maximum channels in units (integer)
   PRP: published relative performance (integer)
   ERP: estimated relative performance from the original article (integer)

Main memory (MMIN/MMAX) is the storage directly accessible by the central processing unit (CPU) for executing instructions and temporarily storing data. It serves as the workspace where active programs and data reside while the CPU processes them. (The RAM and the ROM)

Machine cycle time (MYCT) is the time taken by a computer's CPU to complete one basic operation or cycle of fetching, decoding, executing, and storing an instruction. This is measured in clock speed. 

Cache memory (CACH) is the location where frequently accessed data and instructions is stored. Reducing the time needed for the CPU to retrieve this information from slower main memory.

Channels (CHMIN/CHMAX) are computer pathways that facilitate the transfer of data between different components or devices within a computer system. 

After understanding the acryonyms of the data, we would continue to analyze the dataset of CPU from UCI Machine Learning Repository to see which feature affects the performance speed of the CPU the most or what feature is changed the most to increase the performance speed.

![image](https://github.com/user-attachments/assets/35aec05f-b34b-418d-a4a6-e0e66ac7907c)

*Figure 1. CPU Data uploaded as a CSV from the UCI Machine Learning Repository

After initalizing the data, I wanted primarily look for a comparison to of the estimated relative performance and the published relative performance to see if there was a huge difference. The resulting data gave me:
![image](https://github.com/user-attachments/assets/3134236d-fea2-45fe-81e4-fd9da30ac71a)

*Figure 2. Correlation between ERP and PRP

From the correlation value being 0.9664716584437556, we can tell that there is a strong postive correlation between the estimated and published relative performance. Therefore, the dataset analyzed is accurate. 
Now we would move on to modeling the datasets. 
## Modelling
The primary model that I have decided to analyze is a correlation matrix. The code that I used to get this matrix is:
```python
correlation_matrix = data.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()
```

The code results in a correlation matrix heatmap:


![image](https://github.com/user-attachments/assets/c94a4d94-fbcb-4fbb-96f5-77953f598a70)

*Figure 3 Correlation matrix heatmap

A heatmap where a hotter color (red) would symbolize a strong relationship between the two features of the CPU, where a colder color (blue) would symbolize a weak relationship.

Then I decided to do a collinearity analysis to see if any of the features are correlates to one another.  
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
data.drop('ERP', axis=1, inplace=True)
y = data['PRP']
numeric_data = data.select_dtypes(include=['float64', 'int64'])
X = numeric_data.drop('PRP', axis=1)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display the VIF values
print(vif_data)
```

![image](https://github.com/user-attachments/assets/e801a525-2f97-49d0-87ac-bca59ac59a6d)

*Figure 4. Collinearity Analysis

After seeing the correlations, I have decided to use three machine learning methods: Linear Regression, Random Foresting, and Decison Tree.

Linear Regression 
```python
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# Display feature importance (coefficients)
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coefficients.sort_values(by='Coefficient', ascending=False))
```
![image](https://github.com/user-attachments/assets/fe2bf2b0-5735-4961-85ae-c2f57a81de30)

*Figure 5 Coefficents in Linear Regression

Then a graphic description of the values. 

![image](https://github.com/user-attachments/assets/1c6012ed-f17e-4bed-a518-e63bfa5da85b)

*Figure 6 Linear Regression in graphic

Then moving onto using Random Foresting
```python
# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest Mean Squared Error (MSE):", rf_mse)
print("Random Forest R-squared (R²):", rf_r2)

# Feature Importance
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances from Random Forest:")
print(importance_df)
```

![image](https://github.com/user-attachments/assets/260236ee-a067-4906-ae76-9a74f2a20310)

*Figure 7 Calculated Values of Random Foresting

A graphic description
![image](https://github.com/user-attachments/assets/03caf84e-211a-42f6-af3d-f42b65fce014)

*Figure 8 Graph of Random Foresting

Then moving onto the last modeling: Decision Tree

```python
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)  # Adjust max_depth for better performance
dt_model.fit(X_train, y_train)

# Make predictions
dt_pred = dt_model.predict(X_test)

# Evaluate the model
dt_mse = mean_squared_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

print("Decision Tree Mean Squared Error (MSE):", dt_mse)
print("Decision Tree R-squared (R²):", dt_r2)

# Feature importance
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': dt_model.feature_importances_})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importances from Decision Tree:")
print(importance_df)
```


![image](https://github.com/user-attachments/assets/ea88276f-0711-4fec-9c11-9e8829095aaa)

*Figure 9 Calulation of Decision Tree

![image](https://github.com/user-attachments/assets/3a4ca586-a270-43b2-92e4-2b5edbc7caad)

*Figure 10 Plot of Decision Tree

![image](https://github.com/user-attachments/assets/da91c6f7-cae3-4440-b0c4-b5affb4e8a35)

*Figure 11 Visualization of Decision Tree


## Results

# DO Results based on Decision Tree

## Discussion

# Dicuss from toUpperCase78. (2020). intel-processors/Intel_Core_Processors_Analysis_Part1.ipynb at master · toUpperCase78/intel-processors. GitHub. https://github.com/toUpperCase78/intel-processors/blob/master/Intel_Core_Processors_Analysis_Part1.ipynb

How age affects and how this currently affects your orginial hypo.


## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

The work can be further extended where there would be newer CPUs to be calculated, a new feature is added to the CPU, how the CPU corresponds with the GPU for graphic depiction, or how the CPUs handle aritifical intelligence. As technology is being more advanced, CPUs and computing is necessary to complete all these tasks. 

## References

[1] Intel. (2010). The Story of the Intel® 4004. Intel. https://www.intel.com/content/www/us/en/history/museum-story-of-intel-4004.html

[2] Neupane, M. (2019, June 18). How does a CPU work? FreeCodeCamp.org. https://www.freecodecamp.org/news/how-does-a-cpu-work/

[3] toUpperCase78. (2020). intel-processors/Intel_Core_Processors_Analysis_Part1.ipynb at master · toUpperCase78/intel-processors. GitHub. https://github.com/toUpperCase78/intel-processors/blob/master/Intel_Core_Processors_Analysis_Part1.ipynb


[back](./)

