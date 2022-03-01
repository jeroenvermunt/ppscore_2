import pandas as pd
import numpy as np
import calculation
import scipy
from sklearn import preprocessing


df = pd.DataFrame(data={'A':[1,100,50],'B':['foo','bar','foo']})
print(df)

sparse_input = False
feature_input = []
for feature in ['A','B']:
    if calculation._dtype_represents_categories(df[feature]):
        one_hot_encoder = preprocessing.OneHotEncoder()
        array = df[feature].__array__() # critical
        sparse_matrix = one_hot_encoder.fit_transform(array.reshape(-1, 1)) # critical
        feature_input.append(sparse_matrix)
        sparse_input = True
    else:
        # reshaping needed because there is only 1 feature
        array = df[feature].values # critical
        if not isinstance(array, np.ndarray):  # e.g Int64 IntegerArray
            array = array.to_numpy()
        feature_input.append(array.reshape(-1, 1)) # critical

if sparse_input:
    test = scipy.sparse.hstack((feature_input[0], feature_input[1]))
else:
    test = np.hstack((feature_input[0], feature_input[1]))

print(type(test))

# print(feature_input.reshape(-1,3))