import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RandomizedSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, VotingClassifier, StackingRegressor, GradientBoostingRegressor,  ExtraTreesRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor

# catboost import
from catboost import CatBoostRegressor

# xgboost imports
from xgboost import XGBRegressor, XGBRFRegressor
import xgboost as xgb

# lightgbm import
from lightgbm import LGBMRegressor

# Set theme for seaborn
sns.set_theme(style='white')

# Define RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Pandas display settings
pd.set_option('display.max_columns', None)
pd.options.display.max_rows = None
pd.set_option('display.max_colwidth', 200)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set seed for reproducibility
seed = 42
plt.style.use('ggplot')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#submission = pd.read_csv("solution_template.csv")

train.head()
print("Shape of training data: ", train.shape)
print("Shape of testing data: ", test.shape)

desc = pd.DataFrame(index = list(train))
desc['type'] = train.dtypes
desc['count'] = train.count()
desc['nunique'] = train.nunique()
desc['%unique'] = desc['nunique'] /len(train) * 100
desc['null'] = train.isnull().sum()
desc['%null'] = desc['null'] / len(train) * 100
desc = pd.concat([desc,train.describe().T.drop('count',axis=1)],axis=1)
desc.sort_values(by=['type','null']).style.background_gradient(axis=0)

# Drop columns with heavy missingness
df = train.drop(['metastatic_first_novel_treatment', 'metastatic_first_novel_treatment_type', 'patient_gender'], axis=1)

train['breast_cancer_diagnosis_code'].unique()

code_mapping = {
    '1749': 'C50919', 
    '1744': 'C50419', 
    '1741': 'C50119', 
    '1748': 'C50819', 
    '1743': 'C50319', 
    '1742': 'C50219', 
    '19881': 'C7981', 
    '1759': 'C50929'
}
train['breast_cancer_diagnosis_code'] = train['breast_cancer_diagnosis_code'].replace(code_mapping)

# Replace missing values
def mixed_imputation(df, group_col):
    """
    Impute missing values in a DataFrame using mean for numerical columns and mode for categorical columns, grouped by a specified column.
    Returns:
        DataFrame: DataFrame with missing values imputed.
    """
    for column in df.columns:
        if column != group_col:  # Exclude the group column
            # If the column is numerical, then mean imputation
            if df[column].dtype in [np.dtype('float_'), np.dtype('int_')]:
                mean_impute = df.groupby(group_col)[column].mean()
                df[column] = df[column].fillna(df[group_col].map(mean_impute))

            # If the column is categorical, apply mode imputation
            else :
                mode_impute = df.groupby(group_col)[column].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                df[column] = df[column].fillna(df[group_col].map(mode_impute))

    return df
# Fix bad zip
df['patient_state'] = np.where(df['patient_zip3'] == 630, 'MO', np.where(df['patient_zip3'] == 864, 'AZ', df['patient_state']))

# Male codes to female
df['breast_cancer_diagnosis_code'] = df['breast_cancer_diagnosis_code'].replace({
    'C50122':'C50112', 'C50221':'C50211', 'C50421':'C50411', 'C509':'C5091', 'C50922':'C50912'
})

# Recode categories in test data
test['breast_cancer_diagnosis_code'] = test['breast_cancer_diagnosis_code'].replace({'C5021':'C50219'})

# Population columns
pop_cols = df.loc[:, 'population':'veteran'].columns.to_list()

# Fix outliers
df.loc[df.patient_id == 441322, pop_cols] = df.loc[df.patient_id == 982003, pop_cols].values
df.loc[df.patient_id == 271422, pop_cols] = df.loc[df.patient_id == 271245, pop_cols].values
df.loc[df.patient_id == 714510, pop_cols] = df.loc[df.patient_id == 636245, pop_cols].values


# Subset population data
df_pop = df[['patient_zip3', 'patient_state'] + pop_cols].drop_duplicates().sort_values(by='patient_zip3').reset_index(drop=True)

# Impute missing values
df_pop = mixed_imputation(df=df_pop, group_col='patient_zip3')
df_pop = mixed_imputation(df=df_pop, group_col='patient_state')

print(df_pop.shape)
# Subset temperatures
avg_cols = df.columns[df.columns.str.startswith('Average')].tolist()
df_avg = df[['patient_zip3', 'patient_state'] + avg_cols].drop_duplicates().sort_values(by='patient_zip3').reset_index(drop=True)

print(df_avg.shape)
df_avg.head()
# Melt data
df_avg_melt = pd.melt(df_avg, id_vars=['patient_zip3', 'patient_state'])

# Extract month and convert it to datetime
df_avg_melt['month'] = df_avg_melt['variable'].apply(lambda x: x[len(x)-6:])
df_avg_melt['month'] = pd.to_datetime(df_avg_melt['month'], format='%b-%y')

# # Create growth from prior month
df_avg_melt.sort_values(by=['patient_zip3', 'patient_state', 'month'], inplace=True)

# Fill missingness - forward, then backwards for remaining
df_avg_melt['value'] = df_avg_melt.groupby(['patient_zip3', 'patient_state'])['value'].ffill()
df_avg_melt['value'] = df_avg_melt.groupby(['patient_zip3', 'patient_state'])['value'].bfill()
df_avg_melt.head()
# Reshape data
df_avgs = df_avg_melt.drop('month', axis=1).pivot(index=['patient_zip3', 'patient_state'],columns='variable', values='value').reset_index()[['patient_zip3', 'patient_state'] + avg_cols]
print(df_avgs.shape)
df_avgs.head()

df_full = df.drop(pop_cols + avg_cols, axis=1).merge(
    df_pop, how='left', on=['patient_zip3', 'patient_state']
).merge(
    df_avgs, how='left', on=['patient_zip3', 'patient_state']
)

# Impute payer - most frequent value at zip, then at state
df_full['payer_type'] = mixed_imputation(
    mixed_imputation(
        df_full[['patient_zip3', 'patient_state', 'payer_type']],
        group_col='patient_zip3'),
        group_col='patient_state')['payer_type'].values

# Impute race - most frequent value at zip, then at state
df_full['patient_race'] = mixed_imputation(
    mixed_imputation(df_full[['patient_zip3', 'patient_state', 'patient_race']], group_col='patient_zip3'),
    group_col='patient_state')['patient_race'].values

# Categorize variables
df_full['age_group'] = pd.cut(df_full['patient_age'], right=False, bins=[0, 30, 40, 50, 60, 70, 80, 90, np.inf], labels=[0,1,2,3,4,5,6,7]).astype(int)
df_full['icd_9'] = df_full['breast_cancer_diagnosis_code'].str.startswith('17').astype(int)

# Include bmi info
df_full['bmi_missing'] = df_full['bmi'].isna().astype(int)
df_full['bmi_recoded'] = np.where(df_full['bmi'].isna(), 0,
                                  np.where(df_full['bmi'] < 18.5, 1,
                                          np.where(df_full['bmi'] < 25, 2,
                                                 np.where(df_full['bmi'] < 30, 3, 4))))
df_full.columns = df_full.columns.str.replace(' ', '_').str.replace('-', '')

print(df_full.shape)
df_full.head()
df_test = test.drop(['metastatic_first_novel_treatment', 'metastatic_first_novel_treatment_type', 'patient_gender'], axis=1)
# - Transforms the new (testing) data in the same way

# Fix bad zip
df_test['patient_state'] = np.where(df_test['patient_zip3'] == 630, 'MO',
                                    np.where(df_test['patient_zip3'] == 864, 'AZ', df_test['patient_state']))

# Melt data
df_avg_melt_test = pd.melt(df_test[['patient_zip3', 'patient_state'] + avg_cols].drop_duplicates().sort_values(by='patient_zip3').reset_index(drop=True), id_vars=['patient_zip3', 'patient_state'])

# Extract month and convert it to datetime
df_avg_melt_test['month'] = df_avg_melt_test['variable'].apply(lambda x: x[len(x)-6:])
df_avg_melt_test['month'] = pd.to_datetime(df_avg_melt_test['month'], format='%b-%y')
df_avg_melt_test.sort_values(by=['patient_zip3', 'patient_state', 'month'], inplace=True)

# Fill missingness - forward, then backwards for remaining
df_avg_melt_test['value'] = df_avg_melt_test.groupby(['patient_zip3', 'patient_state'])['value'].ffill()
df_avg_melt_test['value'] = df_avg_melt_test.groupby(['patient_zip3', 'patient_state'])['value'].bfill()

# Reshape data
df_avgs_test = df_avg_melt_test.drop('month', axis=1).pivot(index=['patient_zip3', 'patient_state'],columns='variable', values='value').reset_index()[['patient_zip3', 'patient_state'] + avg_cols]

# Bring all necessary data together
df_test_full = df_test.drop(avg_cols, axis=1).merge(
    df_avgs_test, how='left', on=['patient_zip3', 'patient_state']
)

# Categorize variables
df_test_full['age_group'] = pd.cut(df_test_full['patient_age'], right=False, bins=[0, 30, 40, 50, 60, 70, 80, 90, np.inf], labels=[0,1,2,3,4,5,6,7]).astype(int)
df_test_full['icd_9'] = df_test_full['breast_cancer_diagnosis_code'].str.startswith('17').astype(int)

# Include bmi info
df_test_full['bmi_missing'] = df_test_full['bmi'].isna().astype(int)
df_test_full['bmi_recoded'] = np.where(df_test_full['bmi'].isna(), 0,
                                  np.where(df_test_full['bmi'] < 18.5, 1,
                                          np.where(df_test_full['bmi'] < 25, 2,
                                                 np.where(df_test_full['bmi'] < 30, 3, 4))))

# Impute payer - most frequent value at zip from training
payer_zip = df_full.groupby('patient_zip3')['payer_type'].apply(lambda x: x.value_counts().index[0]).reset_index().set_index('patient_zip3')['payer_type'].to_dict()
df_test_full['payer_type'] = df_test_full['payer_type'].fillna(df_test_full['patient_zip3'].map(payer_zip)).fillna('COMMERCIAL')


df_test_full.columns = df_test_full.columns.str.replace(' ', '_').str.replace('-', '')

print(df_test_full.shape)
df_test_full.head()
df_test_full.to_csv('Final_submission.csv', index=False)
df_test_full.head()

mean_value = df['metastatic_diagnosis_period'].mean()
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['metastatic_diagnosis_period'], kde=True, color='skyblue', ax=ax)
ax.axvline(mean_value, linestyle = '-', color = 'red')
ax.text(mean_value, float(ax.get_ylim()[1]) * 0.55, ' mean = ' + str(round(mean_value, 2)), fontsize=12)

# Set title and labels
ax.set_title('Distribution of Metastatic Diagnosis Period')
ax.set_xlabel('Metastatic Diagnosis Period')
ax.set_ylabel('Frequency')

# Show the plot
plt.show()

def plot_categorical_columns(df, cat_cols):
    """
    Plot pie and bar charts for each categorical column in the DataFrame.
    """
    for col in cat_cols:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot pie chart
        axes[0].pie(df[col].value_counts(), labels=df[col].value_counts().index, autopct='%1.1f%%',
                    wedgeprops=dict(width=0.3, edgecolor='w'), startangle=90, colors=sns.color_palette("Set2", len(df[col].unique())))
        
        # Plot bar chart
        sns.barplot(x=df[col].value_counts().index, y=df[col].value_counts().values,
                    width=0.5, ax=axes[1], palette=sns.color_palette("Set2", len(df[col].unique())))
        
        for bars in axes[1].containers:
            axes[1].bar_label(bars)
        
        axes[1].set_title(f" {col}")
        axes[1].set_xlabel(col)
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# plot_categorical_columns(df, cat_cols)
submission.to_csv('Final_submission.csv', index=False)
submission.head()



