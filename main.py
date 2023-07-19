import math
import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mutual_votes_csv = os.path.join(os.getcwd(), "mutual_votes_data.csv")
abnormal_votes_csv = os.path.join(os.getcwd(), "abnormal_votes_data.csv")
points_csv = os.path.join(os.getcwd(), "points.csv")
COUNTRY_POINTS_BANK = sum([1, 2, 3, 4, 5, 6, 7, 8, 10, 12])


def get_unordered_pairs(lst):
    pairs = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            pair = (lst[i], lst[j])
            pairs.append(pair)
    return pairs


def mount_data(abnormal_size):
    # load eurovision dataframe
    print("Mounting data...")
    if os.path.exists(mutual_votes_csv) and os.path.exists(abnormal_votes_csv):
        return pd.read_csv(mutual_votes_csv), pd.read_csv(abnormal_votes_csv)
    eurovision_df = pd.read_csv("eurovision.csv")

    # filter all eurovision scoring metric where a country voted for herself
    eurovision_df = eurovision_df[eurovision_df['Duplicate'] != 'x']
    eurovision_df = eurovision_df.drop('Duplicate', axis=1)

    # filter semi-final rows
    eurovision_df = eurovision_df[eurovision_df["(semi-) final"] == 'f']

    # remove years with televoting (4 years in total)
    # and years when countries did not receive any votes
    televote_years = set(eurovision_df[(eurovision_df["Jury or Televoting"] == "T")]["Year"].unique())
    diff_from_to_years = set()
    for year, euro in eurovision_df.groupby('Year'):
        if set(euro['From country'].unique()) != set(euro['To country'].unique()):
            diff_from_to_years.add(year)
    filter_years = televote_years.union(diff_from_to_years)
    eurovision_df = eurovision_df[~eurovision_df["Year"].isin(filter_years)]

    # fix Points column appearance (strip trailing spaces from column title)
    points_col_title = 'Points      '
    eurovision_df = eurovision_df.rename(columns={points_col_title: points_col_title.strip()})

    # extract eurovision relevant countries + years
    euro_countries = eurovision_df["From country"].unique()
    euro_years = eurovision_df["Year"].unique()

    for year in euro_years:
        # create points (based on received points)
        # per country over years DataFrame
        countries_in_year = eurovision_df[eurovision_df["Year"] == year]["From country"].unique()
        points_per_country_in_year = eurovision_df[eurovision_df["Year"] == year].groupby(['To country'])[
                                         'Points'].sum() / (countries_in_year.shape[0] - 1)

        for country in countries_in_year:
            eurovision_df.loc[eurovision_df[(eurovision_df["Year"] == year) &
                                            (eurovision_df["To country"] == country)].index, "FVR"] = \
                eurovision_df[(eurovision_df["Year"] == year) & (eurovision_df["To country"] == country)]["Points"] - \
                points_per_country_in_year[country]

    # for each year,
    # create sum of FVRs over the years table for each from/to country pairs
    # where x-axis describes From --> to FVRs and y axis describes to --> From FVRs
    all_pairs = get_unordered_pairs(euro_countries)
    all_pairs_fvr = DataFrame(index=np.arange(len(all_pairs)),
                              columns=["From", "To", "FVR from to", "FVR to from"])
    for idx, (country_a, country_b) in enumerate(all_pairs):
        if country_a != country_b:
            all_from_to = eurovision_df[(eurovision_df["From country"] == country_a) &
                                        (eurovision_df["To country"] == country_b)]
            all_to_from = eurovision_df[(eurovision_df["To country"] == country_a) &
                                        (eurovision_df["From country"] == country_b)]
            all_pairs_fvr.loc[idx] = {
                "From": country_a,
                "To": country_b,
                "FVR from to": all_from_to["FVR"].sum(),
                "FVR to from": all_to_from["FVR"].sum(),
            }

    # remove all rows where voting values are 0
    all_pairs_fvr = all_pairs_fvr[(all_pairs_fvr["FVR from to"] != 0) & (all_pairs_fvr["FVR to from"] != 0)]
    all_pairs_fvr.reset_index(inplace=True)

    # add distance column from common voting behaviour (distance from (0,0) - the mean behaviour)
    all_pairs_fvr["Distance"] = np.sqrt(pd.Series(all_pairs_fvr["FVR from to"], dtype="float64") ** 2
                                        + pd.Series(all_pairs_fvr["FVR to from"], dtype="float64") ** 2)

    # sort all pairs FVR values according to the abnormal behviour rate of mutual votings
    abnormal_pairs = all_pairs_fvr.sort_values(by="Distance", ascending=False).head(abnormal_size)
    abnormal_pairs.reset_index(inplace=True)

    # save all euro countries pairs mutual vote rates dataframe
    all_pairs_fvr.to_csv(mutual_votes_csv)
    abnormal_pairs.to_csv(abnormal_votes_csv)

    return all_pairs_fvr, abnormal_pairs


def analyze_regression(fvr_df: DataFrame):
    print("\n\n--- LINEAR REGRESSION ---")
    # extract x, y axis
    from_to = fvr_df["FVR from to"].to_numpy().reshape((fvr_df["FVR from to"].shape[0], 1))
    to_from = fvr_df["FVR to from"].to_numpy().reshape((fvr_df["FVR to from"].shape[0], 1))

    # ---build regression model for analyzed data---
    # scatter data
    plt.scatter(from_to, to_from)

    # linear regression
    lin_model = LinearRegression()
    lin_model.fit(from_to, to_from)
    lin_pred = lin_model.predict(from_to)

    # calculate accuracy of predictions using MSE method
    mse = mean_squared_error(to_from, lin_pred)

    # visualize linear regression model
    plt.plot(from_to, lin_pred, color='red', label=f'Linear regression (MSE={mse:.2f})')
    plt.xlabel('FVR from to')
    plt.ylabel('FVR to from')
    plt.title('Linear Regression Results on FVR')
    plt.legend()
    plt.show()


def analyze_heatmap(fvr_df: DataFrame, corr=False):
    print("\n\n--- HEATMAP ---")
    # using correlation matrix
    if corr:
        # Select the relevant columns for correlation analysis
        columns_of_interest = ['FVR from to', 'FVR to from']

        # Calculate the correlation matrix
        correlation_matrix = fvr_df[columns_of_interest].corr()

        # Visualize the correlation matrix as a heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)

    else:
        # extract x, y axis
        from_to = fvr_df["FVR from to"].to_numpy().reshape((fvr_df["FVR from to"].shape[0], 1))
        to_from = fvr_df["FVR to from"].to_numpy().reshape((fvr_df["FVR to from"].shape[0], 1))
        from_to = np.asarray(from_to)[:, 0]
        to_from = np.asarray(to_from)[:, 0]

        # Create a heatmap of the scatter plot
        heatmap, x_edges, y_edges = np.histogram2d(from_to, to_from, bins=20)
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot')
        plt.colorbar(label='Counts')

    plt.xlabel('FVR: country A --> country B')
    plt.ylabel('FVR: country B --> country A')
    title = 'Heatmap of FVR mutuality values'
    if corr:
        title += " (Correlation Matrix)"
    plt.title(title)

    plt.show()


def analyze_clustering(fvr_df: DataFrame, num_clusters: int):
    print("\n\n--- CLUSTERING ---")
    # Select the relevant columns for clustering
    clustering_data = fvr_df[['FVR from to', 'FVR to from']]

    # Perform feature scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(scaled_data)

    # Add the cluster labels to the original dataset
    fvr_df['Cluster'] = kmeans.labels_

    # Analyze the cluster centers
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=['X', 'Y'])
    print("Cluster Centers:")
    print(cluster_centers_df)

    # Display the cluster assignments for each country
    print("\nCluster Assignments:")
    print(fvr_df[['From', 'To', 'Cluster']])

    # Plot the clustering results
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', c='red', s=100, label='Cluster Centers')
    plt.xlabel('FVR from to')
    plt.ylabel('FVR to from')
    plt.title('Clustering Results on FVR')
    plt.legend()
    plt.show()


def analyze_random_forest(fvr_df: DataFrame, num_iter: int):
    print("\n\n--- RANDOM FOREST ---")
    # Select the features and target variable for Random Forest
    features = ['FVR from to', 'FVR to from']
    target = 'Cluster'

    accuracy_sum = 0
    for iter in range(num_iter):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(fvr_df[features], fvr_df[target],
                                                            test_size=0.2,
                                                            random_state=42)

        # init random forest model
        rf = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)

        # predict values over learned dataset
        rf_pred = rf.predict(X_test)

        # accumulate results
        accuracy_sum += accuracy_score(y_test, rf_pred)
        print(f"confusion matrix in iteration {iter}:\n{confusion_matrix(y_test, rf_pred)}")

    print(f"\naverage accuracy score over {num_iter} iterations: {(accuracy_sum / num_iter) * 100:.2f}%")


def data_distribution(fvr_df: DataFrame):
    fvr_diff = abs(fvr_df["FVR from to"] - fvr_df["FVR to from"])
    # Create a histogram of the Series values
    fvr_diff.hist()

    # Set labels and title
    plt.xlabel('FVR diff')
    plt.ylabel('Frequency')
    plt.title('Histogram of FVR diff (mutuality of FVR values)')

    plt.show()


def main():
    num_abnormal = 12
    # mount regression analysis dataframes
    all_pairs_fvr, abnormal_pairs = mount_data(abnormal_size=num_abnormal)

    # show data distribution
    data_distribution(fvr_df=all_pairs_fvr)

    # analyze regression
    analyze_regression(fvr_df=all_pairs_fvr)

    # show heatmap (correlation + non-correlation)
    analyze_heatmap(fvr_df=all_pairs_fvr, corr=True)
    analyze_heatmap(fvr_df=all_pairs_fvr)

    # show clustering
    analyze_clustering(fvr_df=all_pairs_fvr, num_clusters=3)

    # show random forest (validating clustering success)
    analyze_random_forest(fvr_df=all_pairs_fvr, num_iter=20)

    # present outlier country pairs which their mutual voting
    # pattern was not according to mean behaviour (voting values were not
    # close to the mean voting values)
    print(f"\n\nTop {num_abnormal} outlier countries pairs:\n{abnormal_pairs[['From', 'To', 'FVR from to', 'FVR to from', 'Distance']]}")


if __name__ == '__main__':
    main()
