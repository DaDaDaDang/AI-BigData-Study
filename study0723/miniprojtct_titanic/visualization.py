from miniprojtct_titanic import dataset
import matplotlib.pyplot as plt
import pandas as pd

#Sex,Pclass, Embarked 생존율 확인 (숙제)
def pie_chart(feature) :
    feature_ratio = dataset.train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = dataset.train[dataset.train['Survived'] == 1][feature].value_counts()
    dead = dataset.train[dataset.train['Survived']==0][feature].value_counts()

    print("feature_size\n", feature_size)
    print("feature_index\n", feature_index)
    print("survived count\n", survived)
    print("dead count\n", dead)

    plt.plot(aspect = 'auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()

    for i, index in enumerate(feature_index) :
        plt.subplot(1, feature_size + 1, i + 1, aspect = 'equal')
        plt.pie([survived[index], dead[index]], labels = ['Survivied', 'Dead'], autopct = '%1.1f%%')
        plt.title(str(index) + '\'s ratio')

    plt.show()

def bar_chart(feature) :
    survived = dataset.train[dataset.train['Survived']==1][feature].value_counts()
    dead = dataset.train[dataset.train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])

    print("survived", survived)
    print("dead", dead)
    df.index = ['Survived', 'Dead']
    df.plot(kind = 'bar', stacked = True)
    plt.show()

if __name__ == '__main__' :
    pie_chart('Sex')
    pie_chart('Pclass')
    pie_chart('Embarked')

    bar_chart('SibSp')
    bar_chart('Parch')
    bar_chart('Embarked')