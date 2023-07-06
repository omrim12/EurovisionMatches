import os
import numpy as np
import pandas as pd
from scipy import stats
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

mutual_votes_csv = os.path.join(os.getcwd(), "mutual_votes_data.csv")
points_csv = os.path.join(os.getcwd(), "points.csv")
COUNTRY_POINTS_BANK = sum([1, 2, 3, 4, 5, 6, 7, 8, 10, 12])


def get_unordered_pairs(lst):
    pairs = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            pair = (lst[i], lst[j])
            pairs.append(pair)
    return pairs


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
    eurovision_df = eurovision_df.drop('Duplicate', axis=1)

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

        # For all voting that appea
        eurovision_df.loc[eurovision_df[eurovision_df["Year"] == year].index, "FVR"] = \
            eurovision_df[eurovision_df["Year"] == year]["Points"] / median_country_in_year

    # for each year,
    # create sum of FVRs table for each from/to country pairs
    # where x-axis describes From --> to FVRs and y axis describes to --> From FVRs
    all_pairs = get_unordered_pairs(euro_countries)
    all_pairs_fvr = DataFrame(index=np.arange(len(all_pairs)), columns=["From", "To", "FVR from to", "FVR to from"])
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
                "FVR to from": all_to_from["FVR"].sum()
            }

    plt.show()

    return all_pairs_fvr


def analyze_regression(all_pairs_fvr: DataFrame):
    # extract x, y axis
    from_to = all_pairs_fvr["FVR from to"].to_numpy().reshape((all_pairs_fvr["FVR from to"].shape[0], 1))
    to_from = all_pairs_fvr["FVR to from"].to_numpy().reshape((all_pairs_fvr["FVR to from"].shape[0], 1))

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
    plt.legend()
    plt.show()

    """
    Conclusion:
    -----------
    """


def analyze_heatmap(all_pairs_fvr: DataFrame):
    # extract x, y axis
    from_to = all_pairs_fvr["FVR from to"].to_numpy().reshape((all_pairs_fvr["FVR from to"].shape[0], 1))
    to_from = all_pairs_fvr["FVR to from"].to_numpy().reshape((all_pairs_fvr["FVR to from"].shape[0], 1))
    from_to = np.asarray(from_to)[:, 0]
    to_from = np.asarray(to_from)[:, 0]

    # # replace NaN values with zeros
    # fvr_2d[np.where(np.isinf(fvr_2d))] = 0

    # Create a heatmap of the scatter plot
    heatmap, x_edges, y_edges = np.histogram2d(from_to, to_from, bins=20)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot')
    plt.colorbar(label='Counts')
    plt.xlabel('FVR: country A --> country B')
    plt.ylabel('FVR: country B --> country A')
    plt.title('Heatmap of FVR mutuality values')
    plt.show()

    plt.scatter(from_to, to_from)
    plt.show()


def main():
    # mount regression analysis dataframes
    all_pairs_fvr = mount_data()

    # analyze regression
    # analyze_regression(all_pairs_fvr=all_pairs_fvr)

    # show heatmap
    analyze_heatmap(all_pairs_fvr=all_pairs_fvr)


if __name__ == '__main__':
    main()
