import os
import pandas as pd


def read_in_data(base_path: str,
                 user_names: dict) -> dict:
    """
    This function gets the saved model predictions from a specific folder containing .csv files. In the base_path
    directory we expect one .csv file per user, which contains a list of the top 50 restaurants for the user, as
    predicted by one model.

    :param base_path: (str) the path to the directory where the data for each user is stored for a specific model
    :param user_names: (dict) a dictionary of usernames for which we want model predictions
    :return: (dict) a dictionary containing a pandas DataFrame for each user that had a prediction file in the
    base_path directory
    """
    dir_list = os.listdir(base_path)
    user_data_dict = {}
    for file in dir_list:
        for i in range(0, len(user_names)):
            if file.find(user_names[i]) != -1:
                full_path = base_path + file
                data = pd.read_csv(full_path)
                user_data_dict[user_names[i]] = data
                break

    return user_data_dict


def standardize_df_names(dfx: pd.DataFrame,
                         base_set: str = 'rf') -> pd.DataFrame:
    """
    This function standardizes the input from the .csv received from each model into a standard pandas DataFrame
    structure (i.e. the same columns, and column names)

    :param dfx: (pd.DataFrame) the data frame as originally read in from the raw .csv file
    :param base_set: (str) which ensemble model did the dfx come from
    :return: (pd.DataFrame) a standardized dataframe
    """
    if base_set == "user":
        # user_cluster_df_names = ['business_id','name','address','city','state','latitude','longitude','ranking']
        bus_ids = list(dfx.iloc[:, 0])
        final_df = pd.DataFrame({'business_id': bus_ids})
        final_df['name'] = dfx['name']
        final_df['ranking'] = dfx['ranking']

    elif base_set == "rf":
        # rf_df_names = ['Unnamed: 0','name','address','city','state','latitude','longitude','Predicted_Rating']
        bus_ids = list(dfx.iloc[:, 0])
        final_df = pd.DataFrame({'business_id': bus_ids})
        final_df['name'] = dfx['name']
        rf_rank = list(dfx.index)
        rf_rank = [x + 1 for x in rf_rank]
        final_df['ranking'] = rf_rank

    elif base_set == "bus":
        # bus_df_names = ['','business_id','name','address','city','state','latitude','longitude','ranking']
        bus_ids = list(dfx['business_id'])
        final_df = pd.DataFrame({'business_id': bus_ids})
        final_df['name'] = dfx['name']
        final_df['ranking'] = dfx['ranking']

    else:
        raise ValueError(f'The supplied base_set, {base_set}, is not one of the available options. '
                         f'The options are: user, rf, and bus')

    return final_df


def get_data_DataFrames(user_names_dict: dict) -> (dict, dict, dict):
    """
    This function encapsulates the function calls necessary to read in the data for each user, and each model that we
    want ensemble predictions for

    :param: user_names_dict: (dict) a dictionary containing each user id we want ensemble predictions for. The
    keys are expected to be integer beginning at 0, and the values are the usernames
    :return: (dict, dict, dict) one dictionary containing all the user model predictions, one dictionary
    containing all the business model predictions, one dictionary containing all the random forrest predictions
    """

    user_cluster_base_path = "/home/taylor/Duke/ECE 682/ECE_682_Final_Project/Analysis/data/user_cluster_rankings/user_cluster_rankings/"
    user_cluster_preds = read_in_data(base_path=user_cluster_base_path, user_names=user_names_dict)

    bus_cluster_base_path = "/home/taylor/Duke/ECE 682/ECE_682_Final_Project/Analysis/data/final_user_recs_BC/final_user_recs/"
    bus_cluster_preds = read_in_data(base_path=bus_cluster_base_path, user_names=user_names_dict)

    rf_file = "/home/taylor/Duke/ECE 682/ECE_682_Final_Project/Analysis/data/RF-closest-rankings/"
    rf_preds = read_in_data(base_path=rf_file, user_names=user_names_dict)

    return user_cluster_preds, bus_cluster_preds, rf_preds


def get_data_for_user(user: int,
                      user_cluster_preds: dict,
                      bus_cluster_preds: dict,
                      rf_preds: dict) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    This function encapsulates the calls necessary to extract and standardize the predictions for one specific user
    from each model. It outputs one standardized pandas Data Frame containing predictions for each model in the ensemble

    :param user: (int) the index of the user which we want predictions for
    :param user_cluster_preds: (dict) a dictionary contain the user-cluster model predictions for each user
    :param bus_cluster_preds: (dict) a dictionary contain the business-cluster model predictions for each user
    :param rf_preds: (dict) a dictionary contain the random forrest model predictions for each user
    :return: (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """

    user_dat = user_cluster_preds[user]
    user_df = standardize_df_names(dfx=user_dat, base_set='user')

    bus_dat = bus_cluster_preds[user]
    bus_df = standardize_df_names(dfx=bus_dat, base_set='bus')

    rf_df = None
    try:
        rf_dat = rf_preds[user]
        rf_df = standardize_df_names(dfx=rf_dat, base_set='rf')
    except:
        rf_df = None

    return user_df, bus_df, rf_df


def get_df_rank(df: pd.DataFrame,
                bus_id: str,
                num_businesses_ranked: int = 50) -> int:
    """
    Determine the ranking given to a specific business by a specific model. If the business does not show up in the
    rankings, it is assumed to be ranked at least 1 beyond the last business listed. Example if 50 businesses are
    listed, and the supplied business is not found, a rank of 51 will be returned.

    :param df: (pd.DataFrame) a dataframe containing the business rankings for a specific model
    :param bus_id: (str) the business id to get the ranking for
    :param num_businesses_ranked: (int) the number of businesses ranked by the underlying model (default is 50)
    :return: (int) the ranking of the business for this specific model
    """
    try:
        rank = df.loc[(df['business_id'] == bus_id), 'ranking']
        if len(rank) == 0:
            rank = num_businesses_ranked + 1
        else:
            rank = int(list(rank)[0])
    except:
        rank = num_businesses_ranked + 1

    return rank


def get_ensemble_rank_for_business_id(bus_id: str,
                                      user_df: pd.DataFrame,
                                      bus_df: pd.DataFrame,
                                      rf_df: pd.DataFrame,
                                      ensemble_method: str = "lowest-sum") -> int:
    """
    This function get computes the ensemble rank for a specific business from the ranking supplied the individual models

    :param bus_id: (str) the id of the business to get the ensemble ranking for
    :param user_df: (pd.DataFrame) a dataframe a restaurant rankings for a specific user as predicted by
    the user-cluster model
    :param bus_df: (pd.DataFrame) a dataframe a restaurant rankings for a specific user as predicted by
    the business-cluster model
    :param rf_df: (pd.DataFrame)  a dataframe a restaurant rankings for a specific user as predicted by
    the random forest model
    :param ensemble_method: (str) how should the individual model's results be combined. There are two recognized
    options: "lowest-sum", and "best-vote"
    :return: (int) The ensemble derived ranking for this specific business
    """
    user_rank = get_df_rank(user_df, bus_id)
    bus_rank = get_df_rank(bus_df, bus_id)

    total_rank = user_rank + bus_rank

    if rf_df is not None:
        rf_rank = get_df_rank(rf_df, bus_id)
        total_rank = total_rank + rf_rank
    else:
        rf_rank = None

    if ensemble_method == 'best-vote':
        if rf_df is not None:
            total_rank = min([user_rank, bus_rank, rf_rank])
        else:
            total_rank = min([user_rank, bus_rank])
    elif ensemble_method != "lowest-sum":
        raise ValueError(f'The supplied ensemble method, {ensemble_method}, is not supported. '
                         f'The supported options are: best-vote, and lowest-sum')

    return total_rank


def ensemble_ordered_user_resturant_ranking(user_df: pd.DataFrame,
                                            bus_df: pd.DataFrame,
                                            rf_df: pd.DataFrame,
                                            ensemble_method: str = "lowest-sum") -> pd.DataFrame:
    """
    This method combines the predictions of all underlying models in the ensemble to generate a final prediction for a
    specific user

    :param user_df: (pd.DataFrame) a dataframe a restaurant rankings for a specific user as predicted by the
    user-cluster model
    :param bus_df: (pd.DataFrame) a dataframe a restaurant rankings for a specific user as predicted by
    the business-cluster model
    :param rf_df: (pd.DataFrame)  a dataframe a restaurant rankings for a specific user as predicted by
    the random forest model
    :param ensemble_method: (str) how should the individual model's results be combined. There are two recognized
    options: "lowest-sum", and "best-vote"
    :return: (pd.DataFrame) the ensemble rankings for all restaurants for a specific user.
    """
    if rf_df is not None:
        business_ids = list(bus_df['business_id']) + list(user_df['business_id']) + list(rf_df['business_id'])
        business_names = list(bus_df['name']) + list(user_df['name']) + list(rf_df['name'])
    else:
        business_ids = list(bus_df['business_id']) + list(user_df['business_id'])
        business_names = list(bus_df['name']) + list(user_df['name'])

    ensemble = pd.DataFrame({'business_id': business_ids})
    ensemble['name'] = business_names

    if rf_df is not None:
        ensemble['ensemble_ranking'] = [0] * (len(bus_df['ranking']) + len(user_df['ranking']) + len(rf_df['ranking']))
    else:
        ensemble['ensemble_ranking'] = [0] * (len(bus_df['ranking']) + len(user_df['ranking']))

    for bus_id in business_ids:
        ensemble_rank = get_ensemble_rank_for_business_id(bus_id, user_df, bus_df, rf_df, ensemble_method)
        ensemble.loc[(ensemble['business_id'] == bus_id), 'ensemble_ranking'] = ensemble_rank

    grouped_ensemble = ensemble.groupby(['business_id', 'name', 'ensemble_ranking'], as_index=False).count()
    ordered_ensemble = grouped_ensemble.sort_values('ensemble_ranking')
    ordered_ensemble = ordered_ensemble.reset_index()
    ordered_ensemble = ordered_ensemble.drop(columns='index')

    return ordered_ensemble


def print_top_x_resturants_from_ensemble_ranking(user_name: str,
                                                 ensemble: pd.DataFrame,
                                                 num_to_print: int = 5) -> None:
    """
    This function prints out the top 'num_to_print' ensemble rankings for a specific user

    :param user_name: (str) the username to print out the ranking for
    :param ensemble: (pd.DataFrame) a dataframe containing all the generated rankings in sorted order.
    :param num_to_print: (int) the number of rankings to print out for this user
    :return: None
    """
    print(f'+-- Top {num_to_print} restaurants for {user_name} ---')
    for i in range(0, num_to_print):
        resturant_name = ensemble['name'][i]
        print(f'|  {i+1}: {resturant_name}')


def ensemble_model_results_and_display_top_x_resturants(user_names_dict: dict,
                                                        top_x: int = 5,
                                                        ensemble_method: str = "lowest-sum") -> None:
    """
    This function computes and prints out the 'top_x' restaurants for a set of users

    :param user_names_dict: (dict) a dictionary contain the user_ids of the users to generate ensemble predictions for
    :param top_x: (int) The number of ensemble rankings to supply per user
    :param ensemble_method: (str) the method used to generate the ensemble rankings.
    The two options are: best-vote, lowest-sum
    :return: None
    """
    user_cluster_preds, bus_cluster_preds, rf_preds = get_data_DataFrames(user_names_dict=user_names_dict)
    for k in user_names_dict.values():
        user_df, bus_df, rf_df = get_data_for_user(user=k,
                                                   user_cluster_preds=user_cluster_preds,
                                                   bus_cluster_preds=bus_cluster_preds,
                                                   rf_preds=rf_preds)

        ordered_ensemble = ensemble_ordered_user_resturant_ranking(user_df, bus_df, rf_df,
                                                                   ensemble_method=ensemble_method)
        print_top_x_resturants_from_ensemble_ranking(user_name=k,
                                                     ensemble=ordered_ensemble,
                                                     num_to_print=top_x)


def get_test_user_names_dict() -> dict:
    """
    Returns a dictionary containing all the usernames for which we want to generate an ensemble prediction

    :return: (dict)
    """
    user_names_dict = {
        0: 'JeDgLoAcyBL87FAT4xNA',
        1: 'IHNqLX2tHqGzULVYKmpQkw',
        2: '6Q8dQGr_DBKcmQ1aXVWvTQ',
        3: '4wFZgzj4DXB3Mn7Q1WLhXg',
        4: 'ahnaBpvC29I66u_6JRzQFw',
    }
    return user_names_dict


def main(top_x_restaurants: int = 10) -> None:
    """
    This program generates ensemble predictions for a specific set of users, see the function get_test_user_names_dict()
    for the specific user ids, from three models: a cluster model of Yelp businesses, a cluster model of Yelp users,
    and random forest model of Yelp users. The ensemble rankings are then printed to the screen for each user using two
    methods, average ranking, and best-vote.

    Average ranking: produces the overall ensemble ranking by averaging the ranking for each specific business across
    all models in the ensemble.

    Best vote: produces the overall ensemble ranking by getting the top ranked restaurants from each model. This means
    if there are three models in the ensemble, then that the number 1 restaurant for each model in the ensemble would
    make up the top 3.

    :param top_x_restaurants: (int) the number of restaurants to include in the predictions for each user
    :return: None
    """
    user_names_dict = get_test_user_names_dict()

    #
    # The ensemble predictions as generated using the best-vote method
    #
    print(f'+===============================================+')
    print(f'|   Top {top_x_restaurants} restaurants using best-vote method    ')
    print(f'|')
    ensemble_model_results_and_display_top_x_resturants(user_names_dict=user_names_dict,
                                                        top_x=top_x_restaurants,
                                                        ensemble_method='best-vote')
    print(f'+================================================+')

    #
    # The ensemble predictions as generated using the lowest-sum method
    #
    print(f'+================================================+')
    print(f'|   Top {top_x_restaurants} restaurants using lowest-sum method    ')
    print(f'|')
    ensemble_model_results_and_display_top_x_resturants(user_names_dict=user_names_dict,
                                                        top_x=top_x_restaurants,
                                                        ensemble_method='lowest-sum')
    print(f'+================================================+')


if __name__ == "__main__":
    main()
