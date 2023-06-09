import os
import pandas as pd
from pandas import DataFrame
import itertools as it
import matplotlib.pyplot as plt


def main():
    YEAR_LEN = 4
    APPEAR_THRESH = 10
    # load dataframes
    matches_df = pd.read_csv("results.csv")
    eurovision_df = pd.read_csv("eurovision.csv")

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
    min_year = eurovision_df['Year'].min()
    max_year = eurovision_df['Year'].max()
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
        matches_euro_data_in_year = pd.Series()
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
            matches_euro_data_in_year[country_pairs.index((country_a, country_b))] = (num_matches_a_b_year,
                                                                                      votes_euro_a_b_year)
        regression_df[str(year)] = matches_euro_data_in_year

    # save regression dataframe
    regression_df.to_csv(os.path.join(os.getcwd(), "regression_data.csv"))

    # TODO: visualize data with matplotlib
    all_matches_votes = []

    # TODO: calculate regression curve to visualize correlation

    # TODO: visualize correlation and make conclusions


if __name__ == '__main__':
    main()
