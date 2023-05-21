import pandas as pd
from pandas import DataFrame
import itertools as it

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

    # extract all relevant years from matches dataframe
    matches_df = matches_df[(matches_df['date'] >= str(eurovision_df['Year'].min())) &
                            (matches_df['date'] <= str(eurovision_df['Year'].max()))]

    # get all participating eurovision countries
    euro_countries = list(set(eurovision_df['To country']).intersection(set(eurovision_df['From country'])))

    # clean matches and eurovision dataframes to hold only countries participating in eurovision and matches
    # appearances = {country: len(list(set(eurovision_df[eurovision_df['From country'] == country]['Year'])))
    #                for country in euro_countries}
    # appearances = dict(filter(lambda elem: elem[1] >= APPEAR_THRESH, appearances.items()))
    matches_df = matches_df[(matches_df['away_team'].isin(euro_countries)) &
                            (matches_df['home_team'].isin(euro_countries))]
    eurovision_df = eurovision_df[eurovision_df['From country'].isin(matches_df['away_team'])]

    # create all pairs of countries who participated both in a mutual vote and a match
    euro_pairs = set(it.product(eurovision_df['From country'].unique(), eurovision_df['To country'].unique()))
    team_pairs = set(it.product(matches_df['away_team'].unique(), matches_df['home_team'].unique()))
    country_pairs = list(euro_pairs.intersection(team_pairs))

    # Create regression DataFrame with x, y domains values
    regression_df = DataFrame(country_pairs, columns=["Country_A", "Country_B"])
    print(regression_df)



    # TODO: for each year - count friendly matches between country x and country y and
    # TODO: total voting points between country x and country y.
    # TODO: mark matches as x param and


if __name__ == '__main__':
    main()
