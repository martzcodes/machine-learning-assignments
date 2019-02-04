import copy
import logging
import pandas as pd
import numpy as np

from collections import Counter

from sklearn import preprocessing, utils
import sklearn.model_selection as ms
from scipy.sparse import isspmatrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from abc import ABC, abstractmethod

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_pairplot(title, df, class_column_name=None):
    plt = sns.pairplot(df, hue=class_column_name)
    return plt


# Adapted from https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
def is_balanced(seq):
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum([(count/n) * np.log((count/n)) for clas, count in classes])
    return H/np.log(k) > 0.75


class DataLoader(ABC):
    def __init__(self, path, verbose, seed):
        self._path = path
        self._verbose = verbose
        self._seed = seed

        self.features = None
        self.classes = None
        self.testing_x = None
        self.testing_y = None
        self.training_x = None
        self.training_y = None
        self.binary = False
        self.balanced = False
        self._data = pd.DataFrame()

    def load_and_process(self, data=None, preprocess=True):
        """
        Load data from the given path and perform any initial processing required. This will populate the
        features and classes and should be called before any processing is done.

        :return: Nothing
        """
        if data is not None:
            self._data = data
            self.features = None
            self.classes = None
            self.testing_x = None
            self.testing_y = None
            self.training_x = None
            self.training_y = None
        else:
            self._load_data()
        self.log("Processing {} Path: {}, Dimensions: {}", self.data_name(), self._path, self._data.shape)
        if self._verbose:
            old_max_rows = pd.options.display.max_rows
            pd.options.display.max_rows = 10
            self.log("Data Sample:\n{}", self._data)
            pd.options.display.max_rows = old_max_rows

        if preprocess:
            self.log("Will pre-process data")
            self._preprocess_data()

        self.get_features()
        self.get_classes()

        self.log("Feature dimensions: {}", self.features.shape)
        self.log("Classes dimensions: {}", self.classes.shape)
        self.log("Class values: {}", np.unique(self.classes))
        class_dist = np.histogram(self.classes)[0]
        class_dist = class_dist[np.nonzero(class_dist)]
        self.log("Class distribution: {}", class_dist)
        self.log("Class distribution (%): {}", (class_dist / self.classes.shape[0]) * 100)
        self.log("Sparse? {}", isspmatrix(self.features))

        if len(class_dist) == 2:
            self.binary = True
        self.balanced = is_balanced(self.classes)

        self.log("Binary? {}", self.binary)
        self.log("Balanced? {}", self.balanced)

    def scale_standard(self):
        self.features = StandardScaler().fit_transform(self.features)
        if self.training_x is not None:
            self.training_x = StandardScaler().fit_transform(self.training_x)

        if self.testing_x is not None:
            self.testing_x = StandardScaler().fit_transform(self.testing_x)
        
        # pipe = Pipeline([
        #     ('Scale', StandardScaler()),
        #     ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
        #     ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
        #     ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
        #     ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median'))
        # ])
        
        # transformed = pipe.fit_transform(self.training_x, self.training_y)
        # print(transformed)
        # print(transformed.shape)
    
    def create_histograms(self):
        for feature in self._data.columns:
            plt.close()
            plt.figure()
            plt.hist(self._data[feature], color = 'blue', edgecolor = 'black')
            # Add labels
            # plt.title()
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig('./output/images/{}_{}_histogram.png'.format(self.data_name(), feature),
                        format='png', dpi=150)
        plt.close()
        plt.figure()
        features = [feature for feature in self._data.columns]
        for i in np.arange(18).astype(np.int):
            plt.subplot(3, 6, i+1)
            plt.hist(self._data[features[i]], color='blue', edgecolor='black')
            plt.xlabel("{}".format(features[i]))
            plt.subplots_adjust(bottom=0.01, right=1.0, top=1.0, left=0, wspace=0, hspace=0)
            # plt.axis('off')
            plt.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False) # labels along the bottom edge are off

        plt.savefig('./report/{}_histogram.png'.format(self.data_name()),
                    format='png', dpi=150)
            
        return

    def build_train_test_split(self, test_size=0.3):
        if not self.training_x and not self.training_y and not self.testing_x and not self.testing_y:
            self.training_x, self.testing_x, self.training_y, self.testing_y = ms.train_test_split(
                self.features, self.classes, test_size=test_size, random_state=self._seed, stratify=self.classes
            )

    def get_features(self, force=False):
        if self.features is None or force:
            self.log("Pulling features")
            self.features = np.array(self._data.iloc[:, 0:-1])

        return self.features

    def get_classes(self, force=False):
        if self.classes is None or force:
            self.log("Pulling classes")
            self.classes = np.array(self._data.iloc[:, -1])

        return self.classes

    def dump_test_train_val(self, test_size=0.2, random_state=123):
        ds_train_x, ds_test_x, ds_train_y, ds_test_y = ms.train_test_split(self.features, self.classes,
                                                                           test_size=test_size,
                                                                           random_state=random_state,
                                                                           stratify=self.classes)
        pipe = Pipeline([('Scale', preprocessing.StandardScaler())])
        train_x = pipe.fit_transform(ds_train_x, ds_train_y)
        train_y = np.atleast_2d(ds_train_y).T
        test_x = pipe.transform(ds_test_x)
        test_y = np.atleast_2d(ds_test_y).T

        train_x, validate_x, train_y, validate_y = ms.train_test_split(train_x, train_y,
                                                                       test_size=test_size, random_state=random_state,
                                                                       stratify=train_y)
        test_y = pd.DataFrame(np.where(test_y == 0, -1, 1))
        train_y = pd.DataFrame(np.where(train_y == 0, -1, 1))
        validate_y = pd.DataFrame(np.where(validate_y == 0, -1, 1))

        tst = pd.concat([pd.DataFrame(test_x), test_y], axis=1)
        trg = pd.concat([pd.DataFrame(train_x), train_y], axis=1)
        val = pd.concat([pd.DataFrame(validate_x), validate_y], axis=1)

        tst.to_csv('../data/{}_test.csv'.format(self.data_name()), index=False, header=False)
        trg.to_csv('../data/{}_train.csv'.format(self.data_name()), index=False, header=False)
        val.to_csv('../data/{}_validate.csv'.format(self.data_name()), index=False, header=False)

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def data_name(self):
        pass

    @abstractmethod
    def _preprocess_data(self):
        pass

    @abstractmethod
    def class_column_name(self):
        pass

    @abstractmethod
    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """
        return train_features, train_classes
    
    def pipe(self, clf_label, clf):
        return Pipeline([('Scale', StandardScaler()), (clf_label, clf)])

    def reload_from_hdf(self, hdf_path, hdf_ds_name, preprocess=True):
        self.log("Reloading from HDF {}".format(hdf_path))
        loader = copy.deepcopy(self)

        df = pd.read_hdf(hdf_path, hdf_ds_name)
        loader.load_and_process(data=df, preprocess=preprocess)
        loader.build_train_test_split()

        return loader

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))


class CreditDefaultData(DataLoader):

    def __init__(self, path='./data/default of credit card clients.xls', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_excel(self._path, header=1, index_col=0)

    def data_name(self):
        return 'CreditDefaultData'

    def class_column_name(self):
        return 'default payment next month'

    def _preprocess_data(self):
        pass

    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """
        return train_features, train_classes


class CreditApprovalData(DataLoader):

    def __init__(self, path='./data/crx.data', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def data_name(self):
        return 'CreditApprovalData'

    def class_column_name(self):
        return '12'

    def _preprocess_data(self):
        # https://www.ritchieng.com/machinelearning-one-hot-encoding/
        to_encode = [0, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15]
        label_encoder = preprocessing.LabelEncoder()
        one_hot = preprocessing.OneHotEncoder()

        df = self._data[to_encode]
        df = df.apply(label_encoder.fit_transform)

        # https://gist.github.com/ramhiser/982ce339d5f8c9a769a0
        vec_data = pd.DataFrame(one_hot.fit_transform(df[to_encode]).toarray())

        self._data = self._data.drop(to_encode, axis=1)
        self._data = pd.concat([self._data, vec_data], axis=1)

        # Clean any ?'s from the unencoded columns
        self._data = self._data[( self._data[[1, 2, 7]] != '?').all(axis=1)]

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


class PenDigitData(DataLoader):
    def __init__(self, path='./data/pendigits.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def class_column_name(self):
        return '16'

    def data_name(self):
        return 'PenDigitData'

    def _preprocess_data(self):
        pass

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


class AbaloneData(DataLoader):
    def __init__(self, path='./data/abalone.data', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def data_name(self):
        return 'AbaloneData'

    def class_column_name(self):
        return '8'

    def _preprocess_data(self):
        pass

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


class HTRU2Data(DataLoader):
    def __init__(self, path='./data/HTRU_2.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def data_name(self):
        return 'HTRU2Data'

    def class_column_name(self):
        return '8'

    def _preprocess_data(self):
        pass

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


class SpamData(DataLoader):
    def __init__(self, path='./data/spambase.data', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def data_name(self):
        return 'SpamData'

    def class_column_name(self):
        return '57'

    def _preprocess_data(self):
        pass

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


class StatlogVehicleData(DataLoader):
    def __init__(self, path='./data/statlog.vehicle.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def data_name(self):
        return 'StatlogVehicleData'

    def class_column_name(self):
        return '18'

    def _preprocess_data(self):
        to_encode = [18]
        label_encoder = preprocessing.LabelEncoder()

        df = self._data[to_encode]
        label_encoder.fit(self._data[to_encode])
        print(label_encoder.classes_)
        df = df.apply(label_encoder.fit_transform)

        self._data = self._data.drop(to_encode, axis=1)
        self._data = pd.concat([self._data, df], axis=1)

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

class WineData(DataLoader):
    def __init__(self, path='./data/wine/wine.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        wine_red = pd.read_csv('../data/wine/winequality-red.csv', header=0, sep=";")
        wine_red['is_red'] = 1
        wine_white = pd.read_csv('../data/wine/winequality-white.csv', header=0, sep=";")
        wine_white['is_red'] = 0

        self._data = pd.concat([wine_red, wine_white])


    def data_name(self):
        return 'WineData'

    def class_column_name(self):
        return 'is_red'

    def _preprocess_data(self):
        df = self._data['quality']

        self._data = self._data.drop(['quality'], axis=1)
        self._data = pd.concat([self._data, df], axis=1)

        print(self._data)

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

class OnlineShoppersData(DataLoader):
    def __init__(self, path='./data/online_shoppers/online_shoppers_intention.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=0)

    def data_name(self):
        return 'OnlineShoppersData'

    def class_column_name(self):
        return 'Revenue'

    def _preprocess_data(self):
        to_encode = ['Month', 'NewVisitor']
        label_encoder = preprocessing.LabelEncoder()

        df = self._data[to_encode]
        df = df.apply(label_encoder.fit_transform)
        
        self._data = self._data.drop(to_encode, axis=1)

        df_class = self._data['Revenue']

        self._data = self._data.drop(['Revenue'], axis=1)
        self._data = pd.concat([self._data, df, df_class], axis=1)
        self._data[['Weekend', 'Revenue']] = (self._data[['Weekend', 'Revenue']]).astype(int)

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

class StarcraftData(DataLoader):
    def __init__(self, path='./data/skillcraft/SkillCraft.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=0)

    def data_name(self):
        return 'StarcraftData'

    def class_column_name(self):
        return 'LeagueIndex'

    def _preprocess_data(self):
        df = self._data['LeagueIndex']

        self._data = self._data.drop(['LeagueIndex', 'GameID'], axis=1)
        self._data = pd.concat([self._data, df], axis=1)

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

class StarcraftModified(DataLoader):
    def __init__(self, path='./data/skillcraft/SkillCraft.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=0)

    def data_name(self):
        return 'StarcraftData'

    def class_column_name(self):
        return 'LeagueIndex'

    def _preprocess_data(self):
        df = self._data['LeagueIndex']

        # 1, 2
        # 3, 4, 5
        # 6, 7, 8

        df[df == 2] = 1
        df[df == 3] = 2
        df[df == 4] = 2
        df[df == 5] = 2
        df[df > 5] = 3
        
        self._data = self._data.drop(['LeagueIndex', 'GameID'], axis=1)
        self._data = pd.concat([self._data, df], axis=1)

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

class AdultData(DataLoader):
    def __init__(self, path='', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        adult = pd.read_csv('./data/adult/adult.data',header=None)
        adult.columns = ['age','employer','fnlwt','edu','edu_num','marital','occupation','relationship','race','sex','cap_gain','cap_loss','hrs','country','income']

        self._data = adult

    def data_name(self):
        return 'AdultData'

    def class_column_name(self):
        return 'income'

    def _preprocess_data(self):
        adult = self._data
        adult['cap_gain_loss'] = adult['cap_gain']-adult['cap_loss']
        adult = adult.drop(['fnlwt','edu','cap_gain','cap_loss'],1)
        adult['income'] = pd.get_dummies(adult.income)
        replacements = { 'Cambodia':' SE-Asia',
                        'Canada':' British-Commonwealth',
                        'China':' China',
                        'Columbia':' South-America',
                        'Cuba':' Other',
                        'Dominican-Republic':' Latin-America',
                        'Ecuador':' South-America',
                        'El-Salvador':' South-America ',
                        'England':' British-Commonwealth',
                        'France':' Euro_1',
                        'Germany':' Euro_1',
                        'Greece':' Euro_2',
                        'Guatemala':' Latin-America',
                        'Haiti':' Latin-America',
                        'Holand-Netherlands':' Euro_1',
                        'Honduras':' Latin-America',
                        'Hong':' China',
                        'Hungary':' Euro_2',
                        'India':' British-Commonwealth',
                        'Iran':' Other',
                        'Ireland':' British-Commonwealth',
                        'Italy':' Euro_1',
                        'Jamaica':' Latin-America',
                        'Japan':' Other',
                        'Laos':' SE-Asia',
                        'Mexico':' Latin-America',
                        'Nicaragua':' Latin-America',
                        'Outlying-US(Guam-USVI-etc)':' Latin-America',
                        'Peru':' South-America',
                        'Philippines':' SE-Asia',
                        'Poland':' Euro_2',
                        'Portugal':' Euro_2',
                        'Puerto-Rico':' Latin-America',
                        'Scotland':' British-Commonwealth',
                        'South':' Euro_2',
                        'Taiwan':' China',
                        'Thailand':' SE-Asia',
                        'Trinadad&Tobago':' Latin-America',
                        'United-States':' United-States',
                        'Vietnam':' SE-Asia',
                        'Yugoslavia':' Euro_2'}
        adult['country'] = adult['country'].str.strip()
        adult = adult.replace(to_replace={'country':replacements,
                                        'employer':{' Without-pay': ' Never-worked'},
                                        'relationship':{' Husband': 'Spouse',' Wife':'Spouse'}})    
        adult['country'] = adult['country'].str.strip()
        print(adult.groupby('country').country.count())   
        for col in ['employer','marital','occupation','relationship','race','sex','country']:
            adult[col] = adult[col].str.strip()
            
        adult = pd.get_dummies(adult)
        adult = adult.rename(columns=lambda x: x.replace('-','_'))

        self._data = adult

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

class MadelonData(DataLoader):
    def __init__(self, path='', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        madX1 = pd.read_csv('./data/madelon/madelon_train.data',header=None,sep=' ')
        madX2 = pd.read_csv('./data/madelon/madelon_valid.data',header=None,sep=' ')
        madX = pd.concat([madX1,madX2],0).astype(float)
        madY1 = pd.read_csv('./data/madelon/madelon_train.labels',header=None,sep=' ')
        madY2 = pd.read_csv('./data/madelon/madelon_valid.labels',header=None,sep=' ')
        madY = pd.concat([madY1,madY2],0)
        madY.columns = ['Class']
        mad = pd.concat([madX,madY],1)

        self._data = mad


    def data_name(self):
        return 'MadelonData'

    def class_column_name(self):
        return 'Class'

    def _preprocess_data(self):
        self._data = self._data.dropna(axis=1,how='all')

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

class MadelonCulled(DataLoader):
    def __init__(self, path='', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        madX1 = pd.read_csv('./data/madelon/madelon_train.data',header=None,sep=' ')
        madX2 = pd.read_csv('./data/madelon/madelon_valid.data',header=None,sep=' ')
        madX = pd.concat([madX1,madX2],0).astype(float)
        madY1 = pd.read_csv('./data/madelon/madelon_train.labels',header=None,sep=' ')
        madY2 = pd.read_csv('./data/madelon/madelon_valid.labels',header=None,sep=' ')
        madY = pd.concat([madY1,madY2],0)
        madY.columns = ['Class']
        mad = pd.concat([madX,madY],1)

        self._data = mad


    def data_name(self):
        return 'MadelonData'

    def class_column_name(self):
        return 'Class'

    def _preprocess_data(self):
        self._data = self._data.dropna(axis=1,how='all')

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

    def pipe(self, clf_label, clf):
        return Pipeline([
            ('Scale', StandardScaler()),
            ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
            ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
            ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
            ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
            (clf_label, clf)])

class WallFollowing(DataLoader):
    def __init__(self, path='./data/wall-following/sensor_readings_24.data', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=None)

    def data_name(self):
        return 'WallFollowingRobot'

    def class_column_name(self):
        return '24'

    def _preprocess_data(self):
        to_encode = [24]
        label_encoder = preprocessing.LabelEncoder()

        df = self._data[to_encode]
        df = df.apply(label_encoder.fit_transform)
        
        self._data = self._data.drop(to_encode, axis=1)
        self._data = pd.concat([self._data, df], axis=1)

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes

if __name__ == '__main__':
    # cd_data = CreditDefaultData(verbose=True)
    # cd_data.load_and_process()

    cw_data = WineData(verbose=True)
    cw_data.load_and_process()

    co_data = OnlineShoppersData(verbose=True)
    co_data.load_and_process()

    cp_data = PenDigitData(verbose=True)
    cp_data.load_and_process()
