import argparse
import logging
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from category_encoders import TargetEncoder

from scipy.stats.mstats import winsorize

from imblearn.over_sampling import SMOTENC

from problem_config import ProblemConfig, ProblemConst, get_prob_config


class RawDataProcessor:
    @staticmethod
    def build_category_features(data, categorical_cols=None):
        if categorical_cols is None:
            categorical_cols = []
        category_index = {}
        if len(categorical_cols) == 0:
            return data, category_index

        df = data.copy()
        # process category features
        for col in categorical_cols:
            df[col] = df[col].astype("category")
            category_index[col] = df[col].cat.categories
            df[col] = df[col].cat.codes
        return df, category_index

    @staticmethod
    def apply_category_features(
        raw_df, categorical_cols=None, category_index: dict = None
    ):
        if categorical_cols is None:
            categorical_cols = []
        if len(categorical_cols) == 0:
            return raw_df

        apply_df = raw_df.copy()
        for col in categorical_cols:
            apply_df[col] = apply_df[col].astype("category")
            apply_df[col] = pd.Categorical(
                apply_df[col],
                categories=category_index[col],
            ).codes
        return apply_df

    @staticmethod
    def convert_config_to_integer(prob_config: ProblemConfig):
        import re

        categorical_cols = []

        for x in prob_config.categorical_cols:
            tmp = x
            tmp = re.sub('feature', '', tmp)
            tmp = int(tmp)
            tmp = tmp - 1
            categorical_cols.append(tmp)

        return categorical_cols

    @staticmethod
    def convert_numeric_columns_to_float(df):
        numeric_columns = df.select_dtypes(include=['int', 'float']).columns
        df.loc[:, numeric_columns] = df.loc[:, numeric_columns].astype(float)
        return df

    @staticmethod
    def process_raw_data(prob_config: ProblemConfig):
        logging.info("start process_raw_data")
        training_data = pd.read_parquet(prob_config.raw_data_path)
        training_data, category_index = RawDataProcessor.build_category_features(
            training_data, prob_config.categorical_cols
        )

        train, dev = train_test_split(
            training_data,
            test_size=prob_config.test_size,
            random_state=prob_config.random_state,
        )

        input_features = ['feature3', 'feature4', 'feature6', 'feature9', 'feature10', 'feature11', 'feature12',
                          'feature18', 'feature19', 'feature20', 'feature21', 'feature22', 'feature23', 'feature24',
                          'feature25', 'feature26', 'feature27', 'feature30', 'feature31', 'feature32', 'feature33',
                          'feature34', 'feature35', 'feature39', 'feature40']

        # convert label columnn to numerical
        labels_rank = {'Normal' : 0,
               'Other' : 1,
               'Information Gathering' : 2,
               'Denial of Service' : 3,
               'Exploits' : 4,
               'Malware' : 5
              }
        
        train['label'] = train['label'].map(labels_rank)
        dev['label'] = dev['label'].map(labels_rank)

        with open(prob_config.category_index_path, "wb") as f:
            pickle.dump(category_index, f)

        # deal with outliers
        for feature in prob_config.numerical_cols:
            train[feature] = winsorize(train[feature], (0.01, 0.02))

        target_col = prob_config.target_col
        train_x = train.drop([target_col], axis=1)[input_features]
        train_y = train[[target_col]]
        test_x = dev.drop([target_col], axis=1)[input_features]
        test_y = dev[[target_col]]


        # encode categorical features using ordinal encoding
        encoder = TargetEncoder()
        train_x = encoder.fit_transform(train_x, train_y)
        test_x = encoder.transform(test_x)

        # preprocessing, scale the dataset
        # scaler = MinMaxScaler()
        # train_x = scaler.fit_transform(train_x)
        # test_x = scaler.transform(test_x)

        # oversampling
        # oversampler = SMOTENC(random_state=prob_config.random_state,
        #                       categorical_features=RawDataProcessor.convert_config_to_integer(prob_config))
        # train_x, train_y = oversampler.fit_resample(
        #     train_x, train_y.values.ravel())

        cols = [col for col in train.columns if col != 'label']

        train_x = pd.DataFrame(train_x, columns=cols)
        train_y = pd.DataFrame(train_y, columns=['label'])
        test_x = pd.DataFrame(test_x, columns=cols)
        test_y = pd.DataFrame(test_y, columns=['label'])

        train_x.to_parquet(prob_config.train_x_path, index=False)
        train_y.to_parquet(prob_config.train_y_path, index=False)
        test_x.to_parquet(prob_config.test_x_path, index=False)
        test_y.to_parquet(prob_config.test_y_path, index=False)
        logging.info("finish process_raw_data")

    @staticmethod
    def load_train_data(prob_config: ProblemConfig):
        train_x_path = prob_config.train_x_path
        train_y_path = prob_config.train_y_path
        train_x = pd.read_parquet(train_x_path)
        train_y = pd.read_parquet(train_y_path)
        return train_x, train_y[prob_config.target_col]

    @staticmethod
    def load_test_data(prob_config: ProblemConfig):
        dev_x_path = prob_config.test_x_path
        dev_y_path = prob_config.test_y_path
        dev_x = pd.read_parquet(dev_x_path)
        dev_y = pd.read_parquet(dev_y_path)
        return dev_x, dev_y[prob_config.target_col]

    @staticmethod
    def load_category_index(prob_config: ProblemConfig):
        with open(prob_config.category_index_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_capture_data(prob_config: ProblemConfig):
        captured_x_path = prob_config.captured_x_path
        captured_y_path = prob_config.uncertain_y_path
        captured_x = pd.read_parquet(captured_x_path)
        captured_y = pd.read_parquet(captured_y_path)
        return captured_x, captured_y[prob_config.target_col]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE1)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    RawDataProcessor.process_raw_data(prob_config)
