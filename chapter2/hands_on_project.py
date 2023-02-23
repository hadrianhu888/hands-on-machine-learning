# Hand's on data science project with Python 

import os 
from pathlib import Path
import tarfile
import urllib.request
import sklearn as sk
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.model_selection import train_test_split
from scipy.stats import binom 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join(DOWNLOAD_ROOT, "housing.tgz")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    tarball_path = Path("datasets/housing/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/handson-ml2/raw/master/datasets/housing/housing.tgz"
        urllib.request.urlretrieve(url, housing_path)
        with tarfile.open(housing_path, "r:gz") as tar:
            tarball_path.extractall(path="datasets/housing")
    else:
        print("File already exists")
    return pd.read_csv(os.path.join("datasets/housing", "housing.csv"))

housing = fetch_housing_data()
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()

IMAGES_PATH = Path() / "images" / "chapter2"
IMAGES_PATH.mkdir(parents = True, exist_ok = True)

def save_fig(fig_id, tight_layout = True, fig_extension = "png", resolution = 300):
    path = IMAGES_PATH / fig_id
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format = fig_extension, dpi = resolution)
    
plt.rc('font', size = 14)
plt.rc('axes', titlesize = 14)
plt.rc('axes', labelsize = 14)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('legend', fontsize = 14)

housing.hist(bins = 50, figsize = (20, 15))
save_fig("housing_hist.png")
plt.show()

"""Create a test set"""

def shuffle_and_split_data(data, test_ratio):
    """Split the data into training and test sets"""
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = shuffle_and_split_data(housing, 0.2)
len(train_set), len(test_set)

np.random.seed(42)

def is_id_in_test_case(identifer, test_ratio):
    return crc32(np.int64(identifer)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test = ids.apply(lambda id_: is_id_in_test_case(id_, test_ratio))
    return data.loc[~in_test], data.loc[in_test]

housing_with_id = housing.reset_index()
train_set_with_id, test_set_with_id = split_data_with_id_hash(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set_with_id, test_set_with_id = split_data_with_id_hash(housing_with_id, 0.2, "id")

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

sample_size = 1000
ratio_female = 0.511
prob_too_small = binom.pmf(ratio_female, sample_size, len(train_set[train_set["ocean_proximity"] == "NEAR OCEAN"]) / len(train_set))
prob_too_large = binom.pmf(ratio_female, sample_size, len(train_set[train_set["ocean_proximity"] == "NEAR BAY"]) / len(train_set))

np.random.seed(42)

housing["income_cat"] = pd.cut(housing["median_income"], bins = [0., 1.5, 3.0, 4.5, 6., np.inf], labels = [1, 2, 3, 4, 5])

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
save_fig("housing_income_cat_bar_plot.png")  # extra code
plt.show()

splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
strat_split = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.loc[train_index]
    strat_test_set_n = housing.loc[test_index]
    strat_split.append((strat_train_set_n, strat_test_set_n))    

strat_train_set, strat_test_set = strat_split[0]

strat_train_set_n, strat_test_set_n = train_test_split(strat_train_set, test_size = 0.2, random_state = 42)

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

compare_props = pd.DataFrame({
    "Overall %": income_cat_proportions(housing),
    "Stratified %": income_cat_proportions(strat_test_set),
    "Random %": income_cat_proportions(test_set),
}).sort_index()

compare_props.plot.bar(rot = 0, grid = True)
plt.xlabel("Income category")
plt.ylabel("Proportion of districts")
save_fig("housing_income_cat_bar_plot.png")  # extra code
plt.show()

compare_props["Stratified %"].plot.bar(rot = 0, grid = True)
plt.xlabel("Stratified %")
plt.ylabel("Proportion of districts")
save_fig("Stratified_housing_income_cat_bar_plot.png")  # extra code
plt.show()

compare_props["Random %"].plot.bar(rot = 0, grid = True)
plt.xlabel("Random %")
plt.ylabel("Proportion of districts")
save_fig("Random_housing_income_cat_bar_plot.png")  # extra code
plt.show()

for set_ in (strat_train_set, strat_test_set, test_set):
    set_.drop("income_cat", axis = 1, inplace = True)
    
housing = strat_train_set.copy()

housing.plot(kind = "scatter", x = "longitude", y = "latitude")
save_fig("bad_visualization_plot.png")  # extra code
plt.show()

housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.2)
save_fig("better_visualization_plot.png")  # extra code
plt.show()

housing.plot(kind = "scatter", x = "longitude", y = "latitude", grid = True, alpha = 0.4, figsize = (10, 7), s = housing["population"] / 100, label = "population", c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True)
save_fig("housing_prices_scatterplot.png")  # extra code
plt.show()

# Download the California image
filename = "california.png"
if not (IMAGES_PATH / filename).is_file():
    homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
    url = homl3_root + "images/end_to_end_project/" + filename
    print("Downloading", filename)
    urllib.request.urlretrieve(url, IMAGES_PATH / filename)

housing_renamed = housing.rename(columns={
    "latitude": "Latitude", "longitude": "Longitude",
    "population": "Population",
    "median_house_value": "Median house value (ᴜsᴅ)"})
housing_renamed.plot(
                kind="scatter", x="Longitude", y="Latitude",
                s=housing_renamed["Population"] / 100, label="Population",
                c="Median house value (ᴜsᴅ)", cmap="jet", colorbar=True,
                legend=True, sharex=False, figsize=(10, 7))

california_img = plt.imread(IMAGES_PATH / filename)
axis = -124.55, -113.95, 32.45, 42.05
plt.axis(axis)
plt.imshow(california_img, extent=axis)

save_fig("california_housing_prices_plot.png")
plt.show()

# looking for corelations

corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending = False)

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(10, 10))
save_fig("scatter_matrix_plot.png")  # extra code
plt.show()

housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1, grid = True)
save_fig("housing_income_scatterplot.png")  # extra code
plt.show()

# experimenting with attribute combinations

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["bedrooms"] / housing["total_bedrooms"]
housing["people_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)

housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"]

null_rows_idx = housing[housing.isnull().any(axis = 1)]
housing.loc[null_rows_idx].head()

housing_option1 = housing.copy()
housing_option1.dropna(subset = ["total_bedrooms"], inplace = True)
housing_option1.loc[null_rows_idx].head()

housing_option2 = housing.copy()
housing_option2.drop("total_bedrooms", axis=1, inplace=True)  # option 2
housing_option2.loc[null_rows_idx].head()

housing_option3 = housing.copy()
median = housing["total_bedrooms"].median()
housing_option3["total_bedrooms"].fillna(median, inplace=True)  # option 3
housing_option3.loc[null_rows_idx].head()

imputer = SimpleImputer(strategy = "median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)

imputer.statistics_
imputer.median().values()

housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)

imputer.statistics_
housing_num.median().values

X = imputer.transform(housing_num)
imputer.feature_names_in_

housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)
housing_tr.loc[null_rows_idx].head()

imputer.strategy
housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)
housing_tr.loc[null_rows_idx].head()

from sklearn.ensemble import set_config 
set_config(pandas_in_out = True)

from sklearn.ensemble import IsolationForest
isolation_forest = IsolationForest(random_state = 42)
outlier_pred = isolation_forest.fit_predict(housing_num)

housing = housing.iloc[outlier_pred == 1]
housing_labels = housing_labels.iloc[outlier_pred == 1]

housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:8]

ordinal_encoder.categories_

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_encoded)
housing_cat_1hot
housing_cat_1hot.toarray()

cat_encoer = OneHotEncoder(spares = False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

cat_encoder.categories_

df_test_unknown = pd.DataFrame("ocean_proximity", ["INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND", "<1H OCEAN"])
pd.get_dummies(df_test_unknown)

cat_encoder.handle_unknown = "ignore"
cat_encoder.transform(df_test_unknown)
cat_encoder.feature_names_in_

cat_encoder.get_feature_names_out()
df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown), columns = cat_encoder.get_feature_names_out(), index = df_test_unknown.index)

df_output

# Feature Scaling










