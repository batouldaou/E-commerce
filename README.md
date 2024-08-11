# E-commerce Customer Segmentation

## Project Overview
This project aims to segment e-commerce customers based on their purchasing behavior and demographic information using unsupervised learning techniques. The goal is to identify distinct customer groups to enable targeted marketing strategies.

## Dataset
The dataset contains e-commerce transaction data including customer demographics, purchase history, and product information.

## Preprocessing
- Missing values in `Description` were filled using the most frequent description for each `StockCode`.
- `CustomerID` with missing values were dropped.
- Duplicates were removed.
- New features like `TotalAmount` were created to enhance clustering.

## Modeling
- Binary encoding was used for `StockCode`.
- Label encoding was used for `Country`.
- The data was scaled using `StandardScaler`.
- K-Means algorithms was used for clustering.
- Model selection was done using the elbow method and silhouette score.

## Results
The optimal number of clusters was determined using the elbow method. Detailed analysis and profiling of these clusters revealed significant insights into customer behavior.

## How to Run
1. Clone the repository.
2. Install the needed libraries
3. Run the Jupyter Notebook: `jupyter notebook e_commerce_customer_segmentation.ipynb`.

## Conclusion
This project demonstrates the use of unsupervised learning for customer segmentation in e-commerce. Future work includes exploring more advanced feature engineering, clustering algorithms and further profiling of customer segments. Ideally it would be better to use silhouette score for different values of clusters but each time it took more than an hour for the large amount of data, therefore it wasnt implemeneted in a loop along with the inertia.

## Acknowledgments
Thanks to Kaggle for providing the dataset.
