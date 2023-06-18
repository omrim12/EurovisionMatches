import os
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

COUNTRY_POINTS_BANK = sum([1, 2, 3, 4, 5, 6, 7, 8, 10, 12])


def mount_data(top_n: int):
    # load eurovision dataframe
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
    topn_vote_df = DataFrame(index=euro_countries, columns=euro_years)
    for year in euro_years:
        # create points (based on received points)
        # per country over years DataFrame
        points_per_country_in_year = eurovision_df[eurovision_df["Year"] == year].groupby(['To country'])[
            'Points'].sum()
        points_df.loc[:, year] = points_per_country_in_year

        # extract top N countries in euro this year
        _top_n = points_per_country_in_year.sort_values().tail(top_n).index[::-1]

        # TODO: might need to analyze a different feature
        # create top N votings rates per country over years dataframe.
        # (i.e. points given to top N countries divided in total points given)
        vote_cols = ['To country', 'Points']
        for from_country, points_given in \
                eurovision_df[eurovision_df["Year"] == year].groupby(['From country'])[vote_cols]:
            topn_voted_countries = list(
                set(points_given[points_given["Points"] > 0]['To country']).intersection(set(_top_n)))
            topn_votes_rate = sum(
                points_given[points_given["To country"].isin(topn_voted_countries)]['Points']) / COUNTRY_POINTS_BANK
            topn_vote_df.loc[from_country, year] = topn_votes_rate

    return points_df, topn_vote_df


def analyze_regression(topn_vote_df: DataFrame, points_df: DataFrame, top_n: int):
    # extract regression x, y axis
    topn_votes_values = topn_vote_df.stack().to_numpy()
    points_values = points_df.stack().to_numpy()
    topn_votes_values = topn_votes_values.reshape((topn_votes_values.shape[0], 1))
    points_values = points_values.reshape((points_values.shape[0], 1))

    # visualize data distribution
    plt.scatter(x=topn_votes_values, y=points_values, color='blue', label='Data points')
    plt.title(f"Eurovision country rank X country votes to top {top_n} rate")
    plt.xlabel(f"Country votes to top {top_n} rate")
    plt.ylabel("Country rank")

    # ---build regression model for analyzed data---
    # Linear regression
    lin_model = LinearRegression()
    lin_model.fit(topn_votes_values, points_values)
    lin_pred = lin_model.predict(topn_votes_values)

    # visualize linear & regression model
    plt.plot(topn_votes_values, lin_pred, color='red', label='Linear regression')
    plt.draw()
    plt.pause(interval=0.01)
    plt.clf()

    """
    Conclusion:
    -----------
    No correlation between voting for top N rate to actual self rank?
    """


def main():
    for top_n in range(3, 20):
        # mount regression analysis dataframes
        points_df, topn_vote_df = mount_data(top_n=top_n)

        # analyze regression of ranks per voting for top N rate
        analyze_regression(topn_vote_df=topn_vote_df,
                           points_df=points_df,
                           top_n=top_n)


if __name__ == '__main__':
    main()
