# TODO: Add final feature engineering functions here


def has_feature(df, feat):
    """Adds a boolean feature if a feature exists or now"""
    return df[feat].notna().astype(int)


def group_by_feature(train_df, test_df, group, agg_feat, transform, name):
    """Adds a new aggregated feature based on a categorical variable"""
    train_df_new = train_df.copy()
    test_df_new = test_df.copy()

    fill_values = train_df_new.groupby(group).aggregate(transform)[agg_feat].to_dict()

    train_df_new[name] = train_df_new[group].map(fill_values)
    test_df_new[name] = test_df_new[group].map(fill_values)

    return train_df_new, test_df_new


def bin_days_below_above(df):
    """Bins the days below and above features into 4 bins"""
    df_eng = df.copy()

    df_eng["freezing_days"] = df_eng["days_below_0F"] + df_eng["days_below_10F"]
    df_eng["cold_days"] = df_eng["days_below_20F"] + df_eng["days_below_30F"]
    df_eng["warm_days"] = df_eng["days_above_80F"] + df_eng["days_above_90F"]
    df_eng["hot_days"] = df_eng["days_above_100F"] + df_eng["days_above_110F"]

    return df_eng


def seasonal_temps(df, stat):
    """Add seasonal temps for a given stat (e.g. "avg_temp")"""
    df_eng = df.copy()

    df_eng[f"winter_{stat}"] = (
        df_eng[f"december_{stat}"]
        + df_eng[f"january_{stat}"]
        + df_eng[f"february_{stat}"]
    ) / 3

    df_eng[f"spring_{stat}"] = (
        df_eng[f"march_{stat}"] + df_eng[f"april_{stat}"] + df_eng[f"june_{stat}"]
    ) / 3

    df_eng[f"summer_{stat}"] = (
        df_eng[f"june_{stat}"] + df_eng[f"july_{stat}"] + df_eng[f"august_{stat}"]
    ) / 3

    df_eng[f"autumn_{stat}"] = (
        df_eng[f"september_{stat}"]
        + df_eng[f"october_{stat}"]
        + df_eng[f"november_{stat}"]
    ) / 3

    return df_eng


def feature_engineer(train_df, test_df):
    """Feature engineering function for the WiDS 2022 kaggle competition.

    Note: The feature engineering here was done in a "quick and dirty" method
    in pandas, as speed is key in Kaggle competitions.  In real life ML, I would
    not do it this way, but rather, with custom sklearn Transformers, which is
    much cleaner.
    """
    train_df_eng = train_df.copy()
    test_df_eng = test_df.copy()

    # whether or not a building has a fog detector
    train_df_eng["has_fog_detector"] = has_feature(train_df_eng, "days_with_fog")
    test_df_eng["has_fog_detector"] = has_feature(test_df_eng, "days_with_fog")

    # whether or not a building has a wind detector
    train_df_eng["has_wind_detector"] = has_feature(train_df_eng, "max_wind_speed")
    test_df_eng["has_wind_detector"] = has_feature(test_df_eng, "max_wind_speed")

    # bin days above/below temperature
    train_df_eng = bin_days_below_above(train_df_eng)
    test_df_eng = bin_days_below_above(test_df_eng)

    # seasonal avg temps
    train_df_eng = seasonal_temps(train_df_eng, "avg_temp")
    test_df_eng = seasonal_temps(test_df_eng, "avg_temp")

    # aggregate features
    agg_feats = ["energy_star_rating", "floor_area", "ELEVATION"]

    for agg_feat in agg_feats:
        name = "mean_" + agg_feat
        train_df_eng, test_df_eng = group_by_feature(
            train_df_eng, test_df_eng, "facility_type", agg_feat, "mean", name
        )

    # whether or not energy star is better than mean for facility
    train_df_eng["e_star_better_than_mean"] = (
        train_df_eng["energy_star_rating"] > train_df_eng["mean_energy_star_rating"]
    ).astype(int)
    test_df_eng["e_star_better_than_mean"] = (
        test_df_eng["energy_star_rating"] > test_df_eng["mean_energy_star_rating"]
    ).astype(int)

    # total snow and rain
    train_df_eng["snow_rain_inches"] = (
        train_df_eng["snowfall_inches"] + train_df_eng["precipitation_inches"]
    )
    test_df_eng["snow_rain_inches"] = (
        test_df_eng["snowfall_inches"] + test_df_eng["precipitation_inches"]
    )

    # total degree days
    train_df_eng["degree_days"] = (
        train_df_eng["cooling_degree_days"] + train_df_eng["heating_degree_days"]
    )
    test_df_eng["degree_days"] = (
        test_df_eng["cooling_degree_days"] + test_df_eng["heating_degree_days"]
    )

    # floor area interaction with e-star
    train_df_eng["e_star_floor_area"] = (
        train_df_eng["floor_area"] * train_df_eng["energy_star_rating"] + 1
    )
    test_df_eng["e_star_floor_area"] = (
        test_df_eng["floor_area"] * test_df_eng["energy_star_rating"] + 1
    )

    # elevation interaction with e-star
    train_df_eng["e_star_elevation"] = (
        train_df_eng["ELEVATION"] * train_df_eng["energy_star_rating"] + 1
    )
    test_df_eng["e_star_elevation"] = (
        test_df_eng["ELEVATION"] * test_df_eng["energy_star_rating"] + 1
    )

    # year_built interaction with e-star
    train_df_eng["e_star_year_built"] = (
        train_df_eng["year_built"] * train_df_eng["energy_star_rating"] + 1
    )
    test_df_eng["e_star_year_built"] = (
        test_df_eng["year_built"] * test_df_eng["energy_star_rating"] + 1
    )

    # cooling degree days interaction with energy star
    train_df_eng["cooling_e_star"] = (
        train_df_eng["cooling_degree_days"] * train_df_eng["energy_star_rating"] + 1
    )
    test_df_eng["cooling_e_star"] = (
        test_df_eng["cooling_degree_days"] * test_df_eng["energy_star_rating"] + 1
    )

    # heating degree days interaction with energy star
    train_df_eng["heating_e_star"] = (
        train_df_eng["heating_degree_days"] * train_df_eng["energy_star_rating"] + 1
    )
    test_df_eng["heating_e_star"] = (
        test_df_eng["heating_degree_days"] * test_df_eng["energy_star_rating"] + 1
    )

    # floor area interaction with year built
    train_df_eng["floor_area_year_built"] = (
        train_df_eng["floor_area"] * train_df_eng["year_built"] + 1
    )
    test_df_eng["floor_area_year_built"] = (
        test_df_eng["floor_area"] * test_df_eng["year_built"] + 1
    )

    # concatenated state and facility type and floor area
    train_df_eng["facility_floor"] = (
        train_df_eng["facility_type"] + "_" + train_df_eng["floor_area"].astype(str)
    )
    test_df_eng["facility_floor"] = (
        test_df_eng["facility_type"] + "_" + test_df_eng["floor_area"].astype(str)
    )

    # concatenated state, facility, year built
    train_df_eng["facility_year"] = train_df_eng["facility_type"] + train_df_eng[
        "year_built"
    ].astype(str)
    test_df_eng["facility_year"] = test_df_eng["facility_type"] + test_df_eng[
        "year_built"
    ].astype(str)

    # concatenated state and facility type and floor area and year
    train_df_eng["facility_floor_year"] = (
        train_df_eng["facility_floor"] + "_" + train_df_eng["year_built"].astype(str)
    )
    test_df_eng["facility_floor_year"] = (
        test_df_eng["facility_floor"] + "_" + test_df_eng["year_built"].astype(str)
    )

    # grouped mean site_eui
    groups = ["facility_floor", "facility_year", "facility_floor_year"]

    for group in groups:
        name = "median_" + group + "_site_eui"
        train_df_eng, test_df_eng = group_by_feature(
            train_df_eng, test_df_eng, group, "site_eui", "median", name
        )

    return train_df_eng, test_df_eng


def feature_engineer_multiple(train_dfs, test_dfs):
    """Feature engineer multiple dataframes at once"""
    train_dfs_eng = {}
    test_dfs_eng = {}

    for (name1, train_df), (name2, test_df) in zip(train_dfs.items(), test_dfs.items()):
        assert name1 == name2
        train_dfs_eng[name1], test_dfs_eng[name2] = feature_engineer(train_df, test_df)

    return train_dfs_eng, test_dfs_eng


def add_cluster_labels(model, ct, train_df, test_df, target):
    """Adds cluster labels as a feature"""
    train_df_cluster = train_df.copy()
    test_df_cluster = test_df.copy()

    X_cluster = ct.fit_transform(train_df.drop(columns=[target]))
    test_cluster = ct.transform(test_df)

    model.fit(X_cluster)

    train_df_cluster["cluster_label"] = model.labels_
    test_df_cluster["cluster_label"] = model.predict(test_cluster)

    return train_df_cluster, test_df_cluster
