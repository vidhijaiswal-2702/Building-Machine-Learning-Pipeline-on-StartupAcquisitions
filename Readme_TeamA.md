# Building Machine Learning Pipeline on Startup-Acquisition

The goal of this project is to predict the future status of a startup—whether it will remain operating, go public (IPO), be acquired, or shut down. Using a supervised machine learning approach, we will train a model based on historical data of startups that were acquired or closed, aiming to identify patterns that influence a startup's long-term outcome. This predictive model will help assess the likely trajectory of currently operating startups.  
We used CrunchBase as the source of the dataset. Crunchbase provides extensive data on startups, including their founding date, industry, funding history, and status (Operating, Acquired, IPO, Closed), which are crucial for training a predictive model.

Understanding the Dataset

The **Startup Acquisition Dataset** is a structured collection of historical data on startups, providing details that can predict the likelihood of a startup being acquired.  
Startups are crucial to the expansion of the economy. They move the economy by bringing fresh perspectives, encouraging innovation, and generating jobs. Every day, dozens of new businesses are launched, and venture capital has grown to represent a sizable asset class, with annual investments topping $100 billion in the US alone. Predicting a startup's growth enables investors to identify businesses with the potential for rapid growth, giving them an advantage over the competition.

* This dataset contains Crunchbase data from the year **2013** .

* Dataset contains a startup's financial information and is labeled with the company's status (IPO, Operating, Acquired, Closed). Dataset is extremely biased:

| IPO  | CLOSED  | ACQUIRED | OPERATING |
| :---- | :---- | :---- | :---- |
| 1.9% | 3.1% | 9.4% | 85.6% |

        
The dataset consists of **44** columns, from which a selection will serve as features for model training. From that 44 features, we selected some features that are most relevant and related to our target feature they are:

The selected features for model training include important characteristics and milestones of each startup: **name** (for identification), **category\_code** (indicating industry), and **status** (the target variable showing operational state like Operating, Acquired, IPO, or Closed). The **founded\_at** and **closed\_at** dates reveal the startup’s age and potential lifespan. **Country\_code** adds geographic context, while **first\_investment\_at** and **last\_investment\_at** mark the investment timeline, and **investment\_rounds** reflects the total number of funding rounds received, indicating investor interest. **Invested\_companies** shows the startup’s investment activities, while **first\_funding\_at** and **last\_funding\_at** identify the span of funding. **Funding\_rounds** and **funding\_total\_usd** give insight into financial backing, and **first\_milestone\_at**, **last\_milestone\_at**, and **milestones** track notable achievements. **Relationships** counts formal connections, and **ROI** (Return on Investment) reflects profitability. **Created\_at** and **updated\_at** timestamps ensure data currency but are not directly predictive. These features together offer a comprehensive view of the startup's growth trajectory, financial health, and market presence, which are crucial for acquisition prediction.


# Data Preprocessing:

This section details the key preprocessing steps applied to prepare the dataset for analysis.

### Data cleaning

Data cleaning refines raw data by:

1. **Checking NaN Values**: Identify the percentage of missing values in each column to assess impact.  
2. **Removing Duplicates**: Delete repeated rows to avoid skewing results.  
3. **Dropping High NaN Columns**: Remove columns mostly filled with NaN values to simplify the dataset.  
4. **Filtering Corrupted/Unnecessary Data**: Eliminate outliers and irrelevant data for accuracy.  
5. **Data Labeling**: Ensure categories and target variables are correctly assigned.

This process improves data quality for reliable analysis and modeling.

### Duplicate and Unique Entry Filtering:
Identified and removed duplicate entries based on company name and status to ensure unique records.
### Column Selection:
Dropped irrelevant columns (e.g., id, created_by, logo_url) to retain only relevant features.
### Location-Based Indicators:
Added top_3_cities (binary) to flag companies in the three most common cities, and usa_non_usa to differentiate U.S. companies from others.
### Handling Missing Values:
Removed columns with over 98% missing values and deleted rows missing critical information (e.g., status, country_code, category_code, founded_at).
### Outlier Removal:
Used interquartile range (IQR) to remove outliers in funding_total_usd and funding_rounds, improving data consistency.
### Date Processing:
Converted date columns to datetime format, extracted the year, and created year-based columns, then removed the original date columns.
### Feature Encoding:
Encoded status as numeric values (ipo = 1, acquired = 2, etc.) and added a binary target feature, isClosed, to indicate if a company is no longer active.
This preprocessing pipeline ensures the dataset is clean, consistent, and optimized for further analysis and modeling.

# Exploratory Data Analysis (EDA) on Company Dataset


**Dataset:**  
The dataset, loaded as `data`, includes various details on companies, such as funding rounds, founding dates, and statuses. The file is read from `cleaned_companies.csv`.

**Information of Dataset:**  
The dataset is inspected for null values and basic information, such as column types and memory usage, using `.isna()`, `.info()`, and `.describe()` functions. This preliminary check helps assess data quality and understand the initial data structure.

**Univariate Analysis:**  
- **Histograms**: For numerical columns, histograms display the frequency distribution of values.
- **Boxplots**: Boxplots are generated for numerical columns to highlight the presence of any outliers.
- **KDE and Histograms**: Kernel Density Plots, combined with histograms, provide a more detailed view of the distribution patterns within each numerical variable.
- **Status Distribution:** A pie chart is used to show the percentage distribution of company statuses within the dataset.
- **Pairplot of Numerical Features:** Selected numerical columns, including `founded_at`, `first_funding_at`, `last_funding_at`, and `funding_rounds`, are visualized to analyze their relationships.
- **Correlation Analysis:** Pearson and Spearman correlation matrices are calculated to understand the linear and ranked relationships between numerical features.

**Descriptive Statistics:**  
Descriptive statistics for numerical variables are provided, including metrics like mean, median, standard deviation, minimum, and maximum values.
| Statistic       | Value                     |
|-----------------|---------------------------|
| Count           | 43,138 entries            |
| Mean            | Values vary per attribute |
| Standard Deviation | Values vary per attribute |
| Min/Max         | Values vary per attribute |


**Correlation Plot of Numerical Variables:**  
A correlation matrix and pairwise plots are generated to understand the relationships between selected numerical variables, such as funding rounds and founding dates. These help in identifying any linear relationships or multicollinearity among variables.

**Visualization of Variables:**  
- **Pie Chart**: A pie chart visualizes the distribution of company statuses within the dataset.
- **Pair Plot**: Selected numerical columns are plotted against each other using a pair plot to observe any interactions or clustering tendencies among variables.
- **3D Scatter Plot:** A 3D scatter plot is created using features `founded_at`, `first_funding_at`, and `last_funding_at` to observe any interactions or clustering patterns.  
- **Principal Component Analysis (PCA):** Dimensionality reduction is performed with PCA to reduce the dataset to two components, and the explained variance ratio is calculated, indicating the proportion of variance captured by each principal component.



# Modeling
### Binary-Classification
- Logistic Regression is used for binary classification.
- 'isClosed' is the target feature.
### Multi-Classification
- Random Forest is used for the multi-classification.
- The output from the binary classification is used as the input for the 'isClosed' feature in multi-classification.
- 'status' is the target feature.


# Deployment
- Flask is used to develop a web application for the following project. 
- Views is where the server side processing takes place.
- All the user inputs are sent here, processed, fed into the developed machine learning pipeline and the output is generated
- This generated output is sent back to the user.




