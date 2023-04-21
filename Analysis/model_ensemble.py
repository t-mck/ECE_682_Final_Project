import os
import pandas as pd


def read_in_data(base_path: str,
                 user_names: dict) -> dict:
    """

    :param base_path:
    :param user_names:
    :return:
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


def get_test_user_names_dict() -> dict:
    """

    :return:
    """
    user_names_dict = {
        0: 'JeDgLoAcyBL87FAT4xNA',
        1: 'IHNqLX2tHqGzULVYKmpQkw',
        2: '6Q8dQGr_DBKcmQ1aXVWvTQ',
        3: '4wFZgzj4DXB3Mn7Q1WLhXg',
        4: 'ahnaBpvC29I66u_6JRzQFw',
    }
    return user_names_dict


def standardize_df_names(dfx: pd.DataFrame,
                         base_set: str = 'rf') -> pd.DataFrame:
    """

    :param dfx:
    :param base_set:
    :return:
    """
    if base_set == "user":
        # user_cluster_df_names = ['business_id',
        #                          'name',
        #                          'address',
        #                          'city',
        #                          'state',
        #                          'latitude',
        #                          'longitude',
        #                          'ranking']
        bus_ids = list(dfx.iloc[:, 0])
        final_df = pd.DataFrame({'business_id': bus_ids})
        final_df['name'] = dfx['name']
        final_df['ranking'] = dfx['ranking']

    if base_set == "rf":
        # rf_df_names = ['Unnamed: 0', #business_id
        #                'name',
        #                'address',
        #                'city',
        #                'state',
        #                'latitude',
        #                'longitude',
        #                'Predicted_Rating']
        bus_ids = list(dfx.iloc[:, 0])
        final_df = pd.DataFrame({'business_id': bus_ids})
        final_df['name'] = dfx['name']
        rf_rank = list(dfx.index)
        rf_rank = [x + 1 for x in rf_rank]
        final_df['ranking'] = rf_rank

    if base_set == "bus":
        # bus_df_names = ['',
        #                 'business_id',
        #                 'name',
        #                 'address',
        #                 'city',
        #                 'state',
        #                 'latitude',
        #                 'longitude',
        #                 'ranking']
        bus_ids = list(dfx['business_id'])
        final_df = pd.DataFrame({'business_id': bus_ids})
        final_df['name'] = dfx['name']
        final_df['ranking'] = dfx['ranking']

    return final_df


def get_data_DataFrames() -> (dict, dict, dict, dict):
    """

    :return:
    """
    user_names_dict = get_test_user_names_dict()

    user_cluster_base_path = "/home/taylor/Duke/ECE 682/ECE_682_Final_Project/Analysis/data/user_cluster_rankings/user_cluster_rankings/"
    user_cluster_preds = read_in_data(base_path=user_cluster_base_path, user_names=user_names_dict)

    bus_cluster_base_path = "/home/taylor/Duke/ECE 682/ECE_682_Final_Project/Analysis/data/final_user_recs_BC/final_user_recs/"
    bus_cluster_preds = read_in_data(base_path=bus_cluster_base_path, user_names=user_names_dict)

    rf_file = "/home/taylor/Duke/ECE 682/ECE_682_Final_Project/Analysis/data/RF-closest-rankings/"
    rf_preds = read_in_data(base_path=rf_file, user_names=user_names_dict)

    return user_cluster_preds, bus_cluster_preds, rf_preds, user_names_dict


def get_data_for_user(user: int,
                      user_cluster_preds: pd.DataFrame,
                      bus_cluster_preds: pd.DataFrame,
                      rf_preds: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """

    :param user:
    :param user_cluster_preds:
    :param bus_cluster_preds:
    :param rf_preds:
    :return:
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


def get_df_rank(df: pd.DataFrame, id:str) -> int:
    """

    :param df:
    :param id:
    :return:
    """
    try:
        rank = df.loc[(df['business_id'] == id), 'ranking']
        if len(rank) == 0:
            rank = 51
        else:
            rank = int(list(rank)[0])
    except:
        rank = 51

    return rank


def ensemble_ordered_user_resturant_ranking(user_df: pd.DataFrame,
                                            bus_df: pd.DataFrame,
                                            rf_df: pd.DataFrame,
                                            ensemble_method="lowest-sum") -> pd.DataFrame:
    """

    :param user_df:
    :param bus_df:
    :param rf_df:
    :param ensemble_method:
    :return:
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

    for id in business_ids:
        user_rank = get_df_rank(user_df, id)
        bus_rank = get_df_rank(bus_df, id)

        total_rank = user_rank + bus_rank

        if rf_df is not None:
            rf_rank = get_df_rank(rf_df, id)
            total_rank = total_rank + rf_rank

        if ensemble_method == 'best-vote':
            if rf_df is not None:
                total_rank = min([user_rank, bus_rank, rf_rank])
            else:
                total_rank = min([user_rank, bus_rank])


        ensemble.loc[(ensemble['business_id'] == id), 'ensemble_ranking'] = total_rank

    grouped_ensemble = ensemble.groupby(['business_id', 'name' ,'ensemble_ranking'], as_index=False).count()
    ordered_ensemble = grouped_ensemble.sort_values('ensemble_ranking')
    ordered_ensemble = ordered_ensemble.reset_index()
    ordered_ensemble = ordered_ensemble.drop(columns='index')

    return ordered_ensemble


def print_top_x_resturants_from_ensemble_ranking(user_name: str,
                                                 ensemble: pd.DataFrame,
                                                 num_to_print: int = 5) -> None:
    """

    :param user_name:
    :param ensemble:
    :param num_to_print:
    :return:
    """
    print(f'+-- Top {num_to_print} restaurants for {user_name} ---')
    for i in range(0, num_to_print):
        resturant_name = ensemble['name'][i]
        print(f'|  {i+1}: {resturant_name}')


def ensemble_model_results_and_display_top_x_resturants(top_x: int = 5,
                                                        ensemble_method="lowest-sum") -> None:
    """

    :param top_x:
    :param ensemble_method:
    :return:
    """
    user_cluster_preds, bus_cluster_preds, rf_preds, user_names_dict = get_data_DataFrames()
    for k in user_names_dict.values():
        user_df, bus_df, rf_df = get_data_for_user(user=k,
                                                   user_cluster_preds=user_cluster_preds,
                                                   bus_cluster_preds=bus_cluster_preds,
                                                   rf_preds=rf_preds)

        ordered_ensemble = ensemble_ordered_user_resturant_ranking(user_df, bus_df, rf_df, ensemble_method=ensemble_method)
        print_top_x_resturants_from_ensemble_ranking(user_name=k,
                                                     ensemble=ordered_ensemble,
                                                     num_to_print=top_x)


def main():
    top_x_resturants = 10
    print(f'+===============================================+')
    print(f'|   Top {top_x_resturants} restaurants using best-vote method    ')
    print(f'|')
    ensemble_model_results_and_display_top_x_resturants(top_x=top_x_resturants, ensemble_method='best-vote')
    print(f'+================================================+')
    print(f'+================================================+')
    print(f'|   Top {top_x_resturants} restaurants using lowest-sum method    ')
    print(f'|')
    ensemble_model_results_and_display_top_x_resturants(top_x=top_x_resturants, ensemble_method='lowest-sum')
    print(f'+================================================+')


if __name__ == "__main__":
    main()
