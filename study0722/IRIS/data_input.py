from IRIS.header_ import *

#logger.debug(iris.DESCR)
logger.debug(iris.keys())
data = iris.data
label = iris.target
columns = iris.feature_names
print(data, label)

data = pd.DataFrame(data, columns=columns)
# logger.degbug(data.head())
# logger.degbug(data.shape())
# logger.degbug(data.describe())
# logger.degbug(data.info())
