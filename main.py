import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

mutual_votes_csv = os.path.join(os.getcwd(), "mutual_votes_data.csv")
points_csv = os.path.join(os.getcwd(), "points.csv")
COUNTRY_POINTS_BANK = sum([1, 2, 3, 4, 5, 6, 7, 8, 10, 12])


def mount_data():
    # mount data from path if already analyzed
    if os.path.exists(mutual_votes_csv) and os.path.exists(points_csv):
        points_df = pd.read_csv(points_csv)
        points_df = points_df.set_index(points_df.columns[0])
        mutual_votes_df = pd.read_csv(mutual_votes_csv)
        mutual_votes_df = mutual_votes_df.set_index(mutual_votes_df.columns[0])
        return points_df, mutual_votes_df

    # load eurovision dataframe
    print("Mounting data...")
    eurovision_df = pd.read_csv("eurovision.csv")

    # filter all eurovision scoring metric where a country voted for herself
    eurovision_df = eurovision_df[eurovision_df['Duplicate'] != 'x']

    # filter semi-final rows
    eurovision_df = eurovision_df[eurovision_df["(semi-) final"] == 'f']

    # remove years with televoting (4 years in total) TODO: maybe add them back ?
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

    # create regression dataframes
    euro_countries = eurovision_df["From country"].unique()
    euro_years = eurovision_df["Year"].unique()
    points_df = DataFrame(index=euro_countries, columns=euro_years)
    mutual_votes_df = DataFrame(index=euro_countries, columns=euro_years)
    for year in euro_years:
        # create points (based on received points)
        # per country over years DataFrame
        points_per_country_in_year = eurovision_df[eurovision_df["Year"] == year].groupby(['To country'])[
            'Points'].sum()
        points_df.loc[:, year] = points_per_country_in_year

        # create mutual vote rate (i.e. mean of diff between points given to points earned)
        # dataframe per country over years DataFrame
        vote_cols = ['To country', 'Points']
        for from_country, points_given in \
                eurovision_df[eurovision_df["Year"] == year].groupby(['From country'])[vote_cols]:
            mutual_votes_diff = []
            for index, voted_country_row in points_given[points_given['Points'] > 0].iterrows():
                voted_country_name = voted_country_row['To country']
                points_voted = voted_country_row['Points']
                points_earned = eurovision_df[(eurovision_df["Year"] == year) &
                                              (eurovision_df['From country'] == voted_country_name) &
                                              (eurovision_df['To country'] == from_country)]['Points'].tolist()[0]
                mutual_votes_diff.append(abs(points_voted - points_earned))

            mutual_votes_diff_rate = (sum(mutual_votes_diff) / len(mutual_votes_diff))
            mutual_votes_df.loc[from_country, year] = mutual_votes_diff_rate

    # fix voting rate to be the diff from max voting diff rate recorded
    max_mutual_votes_diff_rate = np.max(mutual_votes_df.stack().to_numpy())
    mutual_votes_df = mutual_votes_df.applymap(lambda x: max_mutual_votes_diff_rate - x)

    # save analyzed regression dataframes
    points_df.to_csv(points_csv)
    mutual_votes_df.to_csv(mutual_votes_csv)

    # Visualize data
    print(f"Points earned DataFrame:\n{points_df}\n\n"
          f"Mutual votes rate DataFrame:\n{mutual_votes_df}")

    return points_df, mutual_votes_df


def data_distribution(data_dist: DataFrame, name: str):
    # extract all values
    data_raw = data_dist.stack().to_numpy()

    # create histogram
    plt.hist(data_raw, bins=50)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {name} in eurovision over years')
    plt.show()


def analyze_regression(mutual_votes_df: DataFrame, points_df: DataFrame):
    # extract regression x, y axis
    mutual_votes_values = mutual_votes_df.stack().to_numpy()
    points_values = points_df.stack().to_numpy()
    mutual_votes_values = mutual_votes_values.reshape((mutual_votes_values.shape[0], 1))
    points_values = points_values.reshape((points_values.shape[0], 1))

    # visualize data distribution
    plt.scatter(x=mutual_votes_values, y=points_values, color='blue', label='Data points')
    plt.title(f"Eurovision country rank X mutual voting rate")
    plt.xlabel(f"Country mutual voting rate with voted countries")
    plt.ylabel("Country rank")

    # ---build regression model for analyzed data---
    # Linear regression
    lin_model = LinearRegression()
    lin_model.fit(mutual_votes_values, points_values)
    lin_pred = lin_model.predict(mutual_votes_values)

    # calculate accuracy of predictions using MSE method
    mse = mean_squared_error(mutual_votes_values, lin_pred)

    # visualize linear & regression model
    plt.plot(mutual_votes_values, lin_pred, color='red', label=f'Linear regression (MSE={mse:.2f})')
    plt.legend()
    plt.show()



    """
    Conclusion:
    -----------
    
    """


def main():
    # mount regression analysis dataframes
    points_df, mutual_votes_df = mount_data()

    # show data distribution of both parameters (points vs. mutual voting rate)
    data_distribution(data_dist=points_df, name="Points")
    data_distribution(data_dist=mutual_votes_df, name="Mutual voting rate")

    # analyze regression of ranks per mutual voting rate
    analyze_regression(mutual_votes_df=mutual_votes_df,
                       points_df=points_df)


if __name__ == '__main__':
    main()
