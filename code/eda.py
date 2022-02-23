"""EDA for the UCI heart disease dataset."""

# Import libraries.
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import warnings

# Suppress warnings.
warnings.filterwarnings("ignore")

# Function containing EDA, runs if script is executed.
def main():
    """EDA for UCI heart disease dataset."""

    # Load data.
    heart_data = pd.read_csv('../datasets/heart.csv')

    # Display first five rows.
    heart_data.head()

    # Display null values and data types.
    heart_data.info()
    
    # Descriptive statistics for dataset.
    heart_data.describe()

    # Null values for ca.
    heart_data[heart_data.ca==4]

    # Percent null for ca.
    print('\nPercentage of ca null: {}%'.format(
        (len(heart_data[heart_data.ca==4])/len(heart_data.ca))*100))

    # Null values for thal.
    heart_data[heart_data.thal==0]

    # Percent null for thal.
    print('\nPercentage of thal null: {}%'.format(
        (len(heart_data[heart_data.thal==0])/len(heart_data.thal))*100))

    # Define array with numerical feature names.
    numerical = [
        'age',
        'trestbps',
        'chol',
        'thalach',
        'oldpeak',
    ]

    # Define array with categorical feature names.
    categorical = [
        'sex',
        'cp',
        'fbs',
        'restecg',
        'exang',
        'slope',
        'ca',
        'thal',
        'target'
    ]

    # Display outliers greater than or equal to 3 std from mean.
    heart_data[np.abs(stats.zscore(heart_data)) >= 3]

    # Display outliers greater than or equal to 4 std from mean.
    heart_data[np.abs(stats.zscore(heart_data)) >= 4]

    # Display outliers greater than or equal to 6 std from mean.
    heart_data[np.abs(stats.zscore(heart_data)) >= 6]

    # Create boxplot for oldpeak to visualize outlier.
    sns.boxplot(x=heart_data['oldpeak'])
    plt.xlabel('oldpeak')
    plt.savefig('../output/eda/boxplot_oldpeak.jpg', bbox_inches='tight')
    plt.show()

    # Create boxplot for chol to visualize outlier.
    sns.boxplot(x=heart_data['chol'])
    plt.xlabel('chol')
    plt.savefig('../output/eda/boxplot_chol.jpg', bbox_inches='tight')
    plt.show()

    # Create histograms for numerical variables.
    for i in heart_data[numerical].columns:
        plt.hist(heart_data[numerical][i])
        plt.xticks()
        plt.xlabel(i)
        plt.ylabel('number of people')
        plt.savefig('../output/eda/hist_{}.jpg'.format(i), bbox_inches='tight')
        plt.show()
    
    # Create barplots for categorical variables.
    for i in heart_data[categorical].columns:
        sns.barplot(x=heart_data[categorical][i].value_counts().index,
                    y=heart_data[categorical][i].value_counts())
        plt.xlabel(i)
        plt.ylabel('number of people')
        plt.savefig('../output/eda/barplot_{}.jpg'.format(i), bbox_inches='tight')
        plt.show()
    
    # Heart disease (target) percentage of dataset.
    print("\nHeart disease proportions:\n{}".format(
        heart_data["target"].value_counts()/len(heart_data)))
    
    # Create heatmap of Pearson correlation coefficient using numerical
    # variables.
    plt.figure(figsize=(16, 6))
    sns.heatmap(heart_data[numerical].corr(method='pearson'), annot=True)
    plt.title(
        'Pearson correlation coefficient heatmap for numerical variables',
        fontdict={'fontsize':12}, pad=12)
    plt.savefig('../output/eda/heatmap_pearson_corr.jpg', bbox_inches='tight')
    plt.show()

    # Create scatterplot matrix with hue as target.
    sns.pairplot(heart_data, hue='target')
    plt.savefig('../output/eda/pairplot.jpg', bbox_inches='tight')
    plt.show()

    # Create regression plot for thalach and age.
    sns.lmplot(x='age', y='thalach', data=heart_data)
    # Settings to display all markers.
    xticks, xticklabels = plt.xticks()
    xmin = (3*xticks[0] - xticks[1])/2.
    xmax = (3*xticks[-1] - xticks[-2])/2.
    plt.xlim(xmin, xmax)
    plt.xticks(xticks)
    plt.savefig('../output/eda/lmplot_age_thalach.jpg', bbox_inches='tight')
    plt.show()

    # Create regression plot for thalach and age with target hue.
    sns.lmplot(x='age', y='thalach', hue='target', data=heart_data)
    # Settings to display all markers.
    xticks, xticklabels = plt.xticks()
    xmin = (8*xticks[0] - xticks[1])/2.
    xmax = (3*xticks[-1] - xticks[-2])/2.
    plt.xlim(xmin, xmax)
    plt.xticks(xticks)
    plt.savefig('../output/eda/lmplot_age_thalach_target.jpg', bbox_inches='tight')
    plt.show()

    # Create violinplot for age and target.
    sns.violinplot(x='target', y='age', data=heart_data)
    plt.savefig('../output/eda/violinplot_target_age.jpg', bbox_inches='tight')
    plt.show()
    
    # Create histogram for cp with exang as hue.
    sns.histplot(discrete=True,x="cp", hue="exang", data=heart_data,
                 stat="count", multiple="stack")
    plt.ylabel('number of people')
    plt.savefig('../output/eda/histplot_cp_exang.jpg', bbox_inches='tight')
    plt.show()
    
    # Create histogram for cp with target as hue.
    sns.histplot(discrete=True, x="cp", hue="target", data=heart_data,
                 stat="count", multiple="stack")
    plt.ylabel('number of people')
    plt.xticks(ticks=[0,1,2,3])
    plt.savefig('../output/eda/histplot_cp_target.jpg', bbox_inches='tight')
    plt.show()
    
    # Create a violinplot for thalach and exang.
    sns.violinplot(x='exang', y='thalach', data=heart_data)
    plt.savefig('../output/eda/violinplot_exang_thalach.jpg', bbox_inches='tight')
    plt.show()
    
    # Create a violinplot for thalach and exang with target as hue.
    sns.violinplot(x='exang', y='thalach', data=heart_data, hue='target')
    plt.savefig('../output/eda/violinplot_exang_thalach_target.jpg',
                bbox_inches='tight')
    plt.show()
    
    # Create a swarmplot for thalach and exang with target as hue.
    sns.swarmplot(y=heart_data['thalach'], x=heart_data['exang'],
                  hue=heart_data['target'])
    plt.savefig('../output/eda/swarmplot_thalach_exang.jpg', bbox_inches='tight')
    plt.show()
    
    # Create vioplinplot for thalach and target.
    sns.violinplot(x='target', y='thalach', data=heart_data)
    plt.savefig('../output/eda/violinplot_target_thalach.jpg', bbox_inches='tight')
    plt.show()
    
    # Create histogram for exang with target as hue.
    sns.histplot(discrete=True, x="exang", hue="target", data=heart_data,
                 stat="count", multiple="stack")
    plt.ylabel('number of people')
    plt.xticks(ticks=[0,1])
    plt.savefig('../output/eda/histplot_exang_target.jpg', bbox_inches='tight')
    plt.show()
    
    # Create violinplot for oldpeak and slope.
    sns.violinplot(x='slope', y='oldpeak', data=heart_data)
    plt.savefig('../output/eda/violinplot_slope_oldpeak.jpg', bbox_inches='tight')
    plt.show()
    
    # Create violin plot for oldpeak and target.
    sns.violinplot(x='target', y='oldpeak', data=heart_data)
    plt.savefig('../output/eda/violinplot_target_oldpeak.jpg', bbox_inches='tight')
    plt.show()
    
    # Create histogram for slope with target as hue.
    sns.histplot(discrete=True, x="slope", hue="target", data=heart_data,
                 stat="count", multiple="stack")
    plt.ylabel('number of people')
    plt.xticks(ticks=[0,1,2])
    plt.savefig('../output/eda/histplot_slope_target.jpg', bbox_inches='tight')
    plt.show()
    
    # Create histogram for ca with target as hue.
    sns.histplot(discrete=True, x="ca", hue="target", data=heart_data,
                 stat="count", multiple="stack")
    plt.ylabel('number of people')
    plt.xticks(ticks=[0,1,2,3,4])
    plt.savefig('../output/eda/histplot_ca_target.jpg', bbox_inches='tight')
    plt.show()
    
    # Create histogram for thal with target as hue
    sns.histplot(discrete=True, x="thal", hue="target", data=heart_data,
                 stat="count", multiple="stack")
    plt.ylabel('number of people')
    plt.xticks(ticks=[0,1,2,3])
    plt.savefig('../output/eda/histplot_thal_target.jpg', bbox_inches='tight')
    plt.show()
    
    # Create a feature called "risk factors" using common risk factors for
    # heart disease which includes being male and at least 50 years old, being
    # female and at least 45 years old, having resting blood pressure over
    # 130 mmHg, having cholesterol of at least 240 mg/dL, and having fasting
    # blood sugar greater than 120 mg/dL.
    # Feature will be a count of how many risk factors are found.
    
    # Locate age/sex risk.
    age_sex_risk = heart_data.loc[(heart_data.sex == 0) &
                                  (heart_data.age >= 50) |
                                  (heart_data.sex == 1) &
                                  (heart_data.age >= 45)]

    # Locate resting blood pressure risk.
    high_blood_pressure_risk = heart_data.loc[heart_data.trestbps >= 130]

    # Locate cholesterol risk.
    high_cholesterol_risk = heart_data.loc[heart_data.chol >= 240]

    # Locate fasting blood surgar risk.
    diabetes_risk = heart_data.loc[heart_data.fbs == 1]

    # Create an array with all indices with risks.
    risk_factors_indices = np.concatenate((age_sex_risk.index,
                                           high_blood_pressure_risk.index,
                                           high_cholesterol_risk.index,
                                           diabetes_risk.index))

    # Create an array with counts of risks for all rows.
    risk_factor_counts = np.bincount(risk_factors_indices)

    # Convert risk factor counts to dataframe.
    risk_factors = pd.DataFrame(risk_factor_counts)

    # Create coloumn in dataframe using feature.
    risk_factors['risk factors'] = risk_factors
    
    # Copy target to new dataframe with risk factor count feature.
    risk_factors['target'] = heart_data['target'].copy()

    # Create histogram of risk factors with target as hue.
    sns.histplot(discrete=True, x="risk factors", hue="target",
                 data=risk_factors, stat="count", multiple="stack")
    plt.ylabel('number of people')
    plt.xticks(ticks=[0,1,2,3,4])
    plt.savefig('../output/eda/histplot_riskfactors_target.jpg',
                bbox_inches='tight')
    plt.show()
        
if __name__ == "__main__":
    main()
