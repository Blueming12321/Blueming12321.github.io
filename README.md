George Huang

AOS C111: Introduction to Machine Learning for the Physical Sciences

Dr. Alexander Lozinski

December 6, 2024

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

From the correlation matrix (Figure 3), we can see that the MMIN and MMAX have the most correlation to CPU ERP and PRP, while the CACH, CHMN, CHMAX have moderate correlation. The MYCT have a low correlation to CPU ERP and PRP. To continue analyzing, a multicollinearity anaylsis is used to show that in fact that the MMIN and MMAX have a strong correlation to one another and both is considered the main memory which is closely correlated to the CPU ERP and PRP. I decided to try different forms of modeling to continue exploring the CPU features. Linear regression is the next model I developed. From Figure 5, I can see that through linear regression, the CHMAX actually has the highest coefficent of importance. Which was originally different from the result that I got from the correlation matrix. I decided to graph the Linear Regression to see whether or not the calculation is credible. From Figure 6, I can see that the Line of Best fit is only accurate for the CPUs with low predicted and low actual CPU performance. As CPU performance increased, the linear regression model fails to model the data. Therefore, I decided to use two other machine learning methods: Random Foresting and Decision Tree.

In the random foresting model (Figure 7), the mean squared error (MSE) is relatively low showing and the R-squared value indicating a 85.7% variance in the data. By analyzing these two features, the random foresting model is a credible model that can be used in our research. The results gave us that similar to the correlation matrix, the MMAX is the feature that is the most critical feature while CHMIN being of relative importance. I have decided to continue to analyze a decision tree model before making our conclusion. 

Continuing onto using a Decision Tree model, in (figure 9), we can see that the decision tree has a higher mean squared error (MSE) and lower R-squared data. Therefore, the decision tree model is not as accurately depicting the data as the random forest model. But the results of the primary feature of importance to the CPU of both models are the same, the MMAX. Therefore, from analyzing three models and a correlation matrix, I have been able to draw the conclusion that in low CPU performance speed models, the channels are the primary importance to dictate the performance speed and as the performance speed increases, the main memory becomes the most important feature in measuring CPU performance speed. Not saying that the cache or the machine cycle time is not important, it is just not as relavent to the CPU performance speed as the other features such as main memory or channels. 

## Discussion

After being able to analyze the data, I wanted to see how applicable this can be in our real life. I have decided to look into the age and how the age changes the features and how it affects the CPU performance speed. Therefore, finding a project completed by user toUpperCase78 on analyzing the age and performance speed of intel processors (https://github.com/toUpperCase78/intel-processors/blob/master/Intel_Core_Processors_Analysis_Part1.ipynb)[3].  By looking at the results from his/her research, we could see that as technology advances, the CPUs begin to have process at a faster speed and have more CPU cores to them. To handle more CPU cores, the main memory in the computer would start to increase. While this is only seen from older intel CPUs, from the current CPUs that have been released by the two giants: Intel and Advanced Micro Devices, the newer CPUs tend to contain higher amounts of CPU cores which require a higher amount of RAM allow the CPUs to perform at maximum speed.  [4] [5]

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
Main memory is the most important feature in measuring CPU speed.
As more modern CPUs are created, the number of CPU cores are increased which requires a higher amount of main memory which is a located of temporary storage of data for the CPU to fastily access to use in calculations. 

The work can be further extended where there would be newer CPUs to be calculated, a new feature is added to the CPU, how the CPU corresponds with the GPU for graphic depiction, or how the CPUs handle aritifical intelligence. As technology is being more advanced, CPUs and computing is necessary to complete all these tasks.

## References

[1] Intel. (2010). The Story of the Intel® 4004. Intel. https://www.intel.com/content/www/us/en/history/museum-story-of-intel-4004.html

[2] Neupane, M. (2019, June 18). How does a CPU work? FreeCodeCamp.org. https://www.freecodecamp.org/news/how-does-a-cpu-work/

[3] toUpperCase78. (2020). intel-processors/Intel_Core_Processors_Analysis_Part1.ipynb at master · toUpperCase78/intel-processors. GitHub. https://github.com/toUpperCase78/intel-processors/blob/master/Intel_Core_Processors_Analysis_Part1.ipynb

[4] Intel® CoreTM Processors - View Latest Generation Core Processors. (n.d.). Intel. https://www.intel.com/content/www/us/en/products/details/processors/core.html

[5] AMD RyzenTM Processors for Desktops. (2024, August 7). AMD. https://www.amd.com/en/products/processors/desktops/ryzen.html

Location of the dataset used: https://archive.ics.uci.edu/dataset/29/computer+hardware

