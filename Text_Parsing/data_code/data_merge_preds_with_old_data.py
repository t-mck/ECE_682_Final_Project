import pandas as pd
import os
from Text_Parsing.data_code import data_builder as db


class YelpDataMerger:
    def __init__(self):
        self.ydb = db.YelpDataBuilder()
        pass

    @staticmethod
    def convert_starts_to_1_3_range(stars_vect):

        for i in range(0, len(stars_vect)):
            if stars_vect[i] == 1:
                stars_vect[i] = 0
                continue
            elif stars_vect[i] == 2:
                stars_vect[i] = 0
                continue
            elif stars_vect[i] == 3:
                stars_vect[i] = 1
                continue
            elif stars_vect[i] == 4:
                stars_vect[i] = 2
                continue
            elif stars_vect[i] == 5:
                stars_vect[i] = 2

        return stars_vect

    def merge_preds_and_data(self,
                             preds: list,
                             data_file: str = 'yelp_no_nashville_reviews.csv',
                             merge_file_name: str = 'merged_preds_with_yelp_no_nashville_reviews.csv'):

        dfx = pd.read_csv(os.getcwd() + '/data/' + data_file)
        dfx = dfx.drop(dfx.columns[0], axis=1)

        dfx['stars'] = pd.DataFrame(self.convert_starts_to_1_3_range(list(dfx['stars'])))
        converted_stars = [x + 1 for x in dfx['stars']]

        dfx['preds'] = pd.DataFrame(preds)

        dfx['pred_original_diff'] = dfx['stars'] - dfx['preds']

        bus_id_dat = dfx.groupby('business_id')['business_id']
        bus_star_col = dfx.groupby('business_id')['stars'].mean()
        bus_pred_col = dfx.groupby('business_id')['preds'].mean()
        bus_diff_col = dfx.groupby('business_id')['pred_original_diff'].mean()

        bus_id_dat['stars'] = bus_star_col
        bus_id_dat['preds'] = bus_pred_col
        bus_id_dat['pred_original_diff'] = bus_diff_col
        bus_id_dat.to_csv(file_name='business_reviews_predicited_data.csv')

        dfx.to_csv(merge_file_name)


def main():
    ydm = YelpDataMerger()

    data_file = 'merged_preds_with_yelp_no_nashville_reviews.csv'
    merge_file_name = 'merged_preds_with_yelp_no_nashville_reviews2.csv'

    dfx = pd.read_csv(os.getcwd() + '/' + data_file)
    dfx = dfx.drop(dfx.columns[0], axis=1)
    dfx = dfx.drop(dfx.columns[0], axis=1)

    stars = list(dfx["stars"])
    converted_stars = ydm.convert_starts_to_1_3_range(stars)

    converted_stars = [x + 1 for x in converted_stars]

    dfx['stars'] = converted_stars
    diff_counter = 0
    for i in range(0, len(dfx["stars"])):
        diff = dfx.at[i, "stars"] - dfx.at[i, "preds"]
        if diff != 0:
            diff_counter += 1

    print(diff_counter/len(dfx["stars"]))

    dfx['pred_original_diff'] = dfx['stars'] - dfx['preds']

    bus_id_dat_list = dfx.groupby('business_id')['business_id']
    gbnames = list(bstar.index)
    bstar = dfx.groupby('business_id')['stars'].mean()
    bus_star_col = list(dfx.groupby('business_id')['stars'].mean())
    bus_pred_col = list(dfx.groupby('business_id')['preds'].mean())
    bus_diff_col = list(dfx.groupby('business_id')['pred_original_diff'].mean())

    bus_id_dat = pd.DataFrame(gbnames)
    bus_id_dat['stars'] = bus_star_col
    bus_id_dat['preds'] = bus_pred_col
    bus_id_dat['pred_original_diff'] = bus_diff_col
    bus_id_dat.to_csv(file_name='business_reviews_predicited_data.csv')

    dfx.to_csv(file_name=merge_file_name)



if __name__ == "__main__":
    main()
