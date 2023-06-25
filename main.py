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
    # if os.path.exists(mutual_votes_csv) and os.path.exists(points_csv):
    #     points_df = pd.read_csv(points_csv)
    #     points_df = points_df.set_index(points_df.columns[0])
    #     return points_df

    # load eurovision dataframe
    print("Mounting data...")
    eurovision_df = pd.read_csv("eurovision.csv")

    # filter all eurovision scoring metric where a country voted for herself
    eurovision_df = eurovision_df[eurovision_df['Duplicate'] != 'x']
    eurovision_df = eurovision_df.drop('Duplicate' ,axis=1)

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

    for year in euro_years:
        # create points (based on received points)
        # per country over years DataFrame
        points_per_country_in_year = eurovision_df[eurovision_df["Year"] == year].groupby(['To country'])[
            'Points'].sum()

        # normalize columns (points each country received per year)
        # by reducing median value from all
        all_countries_in_year = eurovision_df[eurovision_df["Year"] == year]["From country"].unique()
        num_countries_in_year = all_countries_in_year.shape[0]
        median_country_in_year = points_per_country_in_year.median() / num_countries_in_year
        print(f"A mid table country in {year} eurovision would receive aprrox. {median_country_in_year} per vote.")
        # points_df.loc[:, year] -= median_country_in_year

        # For all voting that appea
        eurovision_df.loc[eurovision_df[eurovision_df["Year"] == year].index, "FVR"] = eurovision_df[eurovision_df["Year"] == year]["Points"] / median_country_in_year

        first_country_fvr = eurovision_df[(eurovision_df["To country"] == all_countries_in_year[0]) &
                                            (eurovision_df["Year"] == year)]["FVR"]


        # plt.scatter(all_countries_in_year[1:], first_country_fvr)
        # plt.title(f"FVR to {all_countries_in_year[0]} in year {year}, median={median_country_in_year}")
        # plt.show()

    print(eurovision_df[eurovision_df["Year"] == 1975][["From country", "To country", "Year", "Points", "FVR"]])
    return 5


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
    points_df = mount_data()

    # show data distribution of both parameters (points vs. mutual voting rate)
    # data_distribution(dabta_dist=points_df, name="Points")


if __name__ == '__main__':
    main()
