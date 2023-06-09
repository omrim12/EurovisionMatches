import os
import pandas as pd
from pandas import DataFrame
import itertools as it
import matplotlib.pyplot as plt
import numpy as np

regression_data_filename = os.path.join(os.getcwd(), "regression_data.csv")


def mount_data(regression_csv_path: str) -> (pd.DataFrame, int, int):
    YEAR_LEN = 4

    # load dataframes
    matches_df = pd.read_csv("results.csv")
    eurovision_df = pd.read_csv("eurovision.csv")

    # get years min + max borders
    min_year = eurovision_df['Year'].min()
    max_year = eurovision_df['Year'].max()

    if os.path.exists(regression_data_filename):
        return pd.read_csv(regression_csv_path), min_year, max_year

    # modify matches dates in dataframes to hold only year value
    matches_df['date'] = matches_df['date'].str[:YEAR_LEN]

    # filter tournament type to be friendly only in matches dataframe
    matches_df = matches_df[matches_df["tournament"] == "Friendly"]

    # filter all eurovision scoring metric where a country voted for herself
    eurovision_df = eurovision_df[eurovision_df['From country'] != eurovision_df['To country']]

    # fix Points column appearance (strip trailing spaces from column title)
    points_col_title = 'Points      '
    eurovision_df = eurovision_df.rename(columns={points_col_title: points_col_title.strip()})

    # extract all relevant years from matches dataframe
    matches_df = matches_df[(matches_df['date'] >= str(min_year)) &
                            (matches_df['date'] <= str(max_year))]

    # get all participating eurovision countries (received and supplied votes)
    euro_countries = list(set(eurovision_df['To country']).intersection(set(eurovision_df['From country'])))

    # clean matches and eurovision dataframes to hold only countries participating in eurovision and matches
    matches_df = matches_df[(matches_df['away_team'].isin(euro_countries)) &
                            (matches_df['home_team'].isin(euro_countries))]
    eurovision_df = eurovision_df[(eurovision_df['From country'].isin(matches_df['away_team'])) |
                                  (eurovision_df['From country'].isin(matches_df['home_team']))]

    # create all pairs of countries who participated both in a mutual vote and a match
    euro_pairs = set(it.product(eurovision_df['From country'].unique(), eurovision_df['To country'].unique()))
    team_pairs = set(it.product(matches_df['away_team'].unique(), matches_df['home_team'].unique()))

    # TODO: check if country duplicates + a == b case can be done with pandas methods
    # filtering duplicates and same country cases
    country_pairs = []
    for (a, b) in list(euro_pairs.intersection(team_pairs)):
        if (b, a) not in country_pairs and a != b:
            country_pairs.append((a, b))

    # Create regression DataFrame with x, y domains values
    regression_df = DataFrame(country_pairs, columns=["Country_A", "Country_B"])
    for year in range(min_year, max_year + 1):
        matches_col_name = f"{str(year)}_matches"
        votes_col_name = f"{str(year)}_votes"
        matches_in_year = pd.Series(name=matches_col_name, index=range(len(country_pairs)))
        votes_in_year = pd.Series(name=votes_col_name, index=range(len(country_pairs)))
        for country_a, country_b in country_pairs:
            # count number of matches between country a and b in current year
            matches_a_b = matches_df[(((matches_df['away_team'] == country_a) &
                                       (matches_df['home_team'] == country_b)) |
                                      ((matches_df['away_team'] == country_b) &
                                       (matches_df['home_team'] == country_a))) &
                                     (matches_df['date'] == str(year))
                                     ]
            num_matches_a_b_year = len(matches_a_b[matches_a_b['date'] == str(year)])

            # count total voting points between country a and b in current year
            euro_a_b = eurovision_df[((eurovision_df['From country'] == country_a) &
                                      (eurovision_df['To country'] == country_b)) |
                                     ((eurovision_df['From country'] == country_b) &
                                      (eurovision_df['To country'] == country_a))]
            votes_euro_a_b_year = sum(euro_a_b[euro_a_b['Year'] == year]['Points'])

            # create (m, v) point (m = friendly matches, v = sum of mutual votes) for country a and b in current year
            country_pair_index = country_pairs.index((country_a, country_b))
            matches_in_year[country_pair_index] = num_matches_a_b_year
            votes_in_year[country_pair_index] = votes_euro_a_b_year

        regression_df[matches_col_name] = matches_in_year
        regression_df[votes_col_name] = votes_in_year

    # save regression dataframe
    regression_df.to_csv(regression_data_filename)

    return regression_df, min_year, max_year


def examine_regression(regression_df: pd.DataFrame, min_year, max_year):
    # TODO: visualize data with matplotlib
    # Assuming your DataFrame is named 'df'
    # Extract the columns for years, votes, and matches
    years = regression_df.columns[3::2].str[:4]
    votes = regression_df.iloc[:, 4::2].melt()['value']
    matches = regression_df.iloc[:, 3::2].melt()['value']

    # Visualize data
    print(years.shape)
    print(votes.shape)
    print(matches.shape)

    # TODO: fix scatter
    plt.scatter(years, matches, label='Matches')
    plt.scatter(years, votes, label='Votes')

    # Set plot title and labels
    plt.title('Relationship between Votes and Matches over the Years')
    plt.xlabel('Year')
    plt.ylabel('Count')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

    # TODO: calculate regression curve to visualize correlation

    # TODO: visualize correlation and make conclusions


def main():
    # mount regression of matches X eurovision votes dataframe
    regression_df, min_year, max_year = mount_data(regression_csv_path=regression_data_filename)

    # analyze regression between matches X eurovision votes
    examine_regression(regression_df=regression_df, min_year=min_year, max_year=max_year)


if __name__ == '__main__':
    main()
