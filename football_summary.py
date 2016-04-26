import json
import pandas as pd
import numpy as np
import glob
import sklearn
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
import time

#Loads the files provided by hudl into a dataframe
def loadFiles():
    #loadfiles
    path = 'C:\Users\Steves\PycharmProjects\csce474groupproject\data\*.txt'
    allFiles = glob.glob(path)
    rows = pd.DataFrame()
    plays_list = []
    for fileName in allFiles:
        f = open(fileName, 'rb')
        data = f.readlines()
        data_json_str = "[" + ','.join(data) + "]"
        data_df = pd.read_json(data_json_str)
        plays_list.append(data_df)
        f.close()
    rows = pd.concat(plays_list)
    return rows

#Will load the combined csv into a dataframe
def loadFile(filename):
    return pd.read_csv(filename)

#Outputs basic summary stats, will give mean/quartile for all numeric columns
def getSummary(rows):
    #summary
    print "Size:" + str(len(rows.index))
    #print list(rows)
    print rows.describe()

#exports the dataframe to a csv, motion has to be dropped because it has unicode characters that are a bitch to handle
def exportToCSV(rows):
    #export to csv
    rows = rows.drop('MOTION', 1)
    rows.to_csv("plays.csv")

#Will split the data into runs and passes, then drops the PLAY TYPE column
def filterPlayType(plays):
    runs = pd.DataFrame()
    passes = pd.DataFrame()
    runs = plays[plays['PLAY TYPE'] == 'Run']
    passes = plays[plays['PLAY TYPE'] == 'Pass']
    runs = runs.drop('PLAY TYPE', 1)
    passes = passes.drop('PLAY TYPE', 1)
    return runs, passes

#takes a columnname and will filter out rows with empty values for that column
def filterEmpty(plays, columnname):
    plays = plays[plays[columnname].notnull()]
    return plays

#filters the nominal and numeric columns, numeric columns are filtered based on percentile to eliminate obvious outliers
def filterColumns(plays, checkQTR):
    print "DN"
    print len(plays)
    plays = plays[plays["DN"].isin([1,2,3,4])]
    print len(plays)
    print "Play #"
    print len(plays)
    plays = plays[plays["PLAY #"] > 0]
    plays = plays[plays["PLAY #"] < 200]
    print len(plays)
    print "DIST"
    print len(plays)
    plays = plays[plays["DIST"] > 0]
    plays = plays[plays["DIST"] <= 35]
    print len(plays)
    print "GN/LS"
    print len(plays)
    plays = plays[plays["GN/LS"] >= -30]
    print len(plays)
    print "YARD LN"
    print len(plays)
    plays = plays[plays["YARD LN"] >= -50]
    print len(plays)
    if(checkQTR):
        print len(plays)
        plays = plays[plays["QTR"].isin([1,2,3,4])]
        print len(plays)
    return plays


def scaleYardLine(yardline):
    if(0 < yardline  < 50):
        return 100 - yardline
    if(0 > yardline >= -50):
        return abs(yardline)
    else:
        return yardline


#drops columns from the dataframe for the purpose of clustering, I manually comment out based on which columns I want
def dropColumnsInsufficientData(plays):
    plays = plays.drop('GAP', 1)
    plays = plays.drop('PLAY DIR', 1)
    #plays = plays.drop('QTR', 1)
    plays = plays.drop('RESULT', 1)
    plays = plays.drop('Unnamed: 0', 1)
    return plays

#drops columns from the dataframe for the purpose of clustering, I manually comment out based on which columns I want
def dropColumnsClustering(plays):
    #plays = plays.drop('MOTION', 1)
    plays = plays.drop('DIST', 1)
    #plays = plays.drop('DN', 1)
    plays = plays.drop('HASH', 1)
    plays = plays.drop('YARD LN', 1)
    plays = plays.drop('PLAY #', 1)
    return plays


#drops columns from the dataframe for the purpose of clustering, I manually comment out based on which columns I want
def dropColumnsClustering3(plays):
    #plays = plays.drop('MOTION', 1)
    #plays = plays.drop('DIST', 1)
    #plays = plays.drop('DN', 1)
    plays = plays.drop('HASH', 1)
    plays = plays.drop('YARD LN', 1)
    plays = plays.drop('PLAY #', 1)
    return plays


def clusterAnalysis(plays, k):
    print 'Total size: ' + str(len(plays))
    for i in range(0, k):
        filteredPlays = plays[plays["CLUSTER"] == i]
        print "Cluster-" + str(i)
        print getSummary(filteredPlays)


#Performs k-means clustering on the dataset, will return a dataframe with associated clusters
def clusterPlays(plays, k):
        filteredPlays = dropColumnsClustering(plays)
        columns = plays.columns.values.tolist()
        data = scale(filteredPlays)
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
        kmeans.fit(data)
        clusteredPlays = pd.DataFrame(plays, columns=columns)
        clusteredPlays.insert(0, 'CLUSTER', kmeans.labels_)
        filename = 'k' + str(k) + '_clusteredPlays_runs_'+ time.strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
        #clusteredPlays.to_csv(filename)
        #clusterAnalysis(clusteredPlays, k)
        return clusteredPlays

#Uses silhouette score to determine the goodness of the clusters, k is an array of the desired k values
def clusterGoodness(type, plays, k):
        plays = dropColumnsClustering(plays)
        #print  plays.columns.values
        #data = scale(plays)
        data = plays.values
        rows = []
        colList = plays.columns.values.tolist()
        cols = str(', '.join(colList))
        for i in k:
            start = time.time()
            kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels, sample_size=50000)
            end = time.time()
            runtime = (end - start)
            row = {'Features' : cols, 'K' : i, 'Silhouette_Avg' : silhouette_avg, 'Time' : runtime, 'Size' : len(data)};
            rows.append(row)
        output = pd.DataFrame(rows)
        #Will output a csv with the results and date ran
        filename = 'cluster_goodness_' + type + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
        output.to_csv(filename)


#Uses silhouette score to determine the goodness of the clusters, k is an array of the desired k values
def clusterGoodness3(type, plays, k):
        plays = dropColumnsClustering3(plays)
        #print  plays.columns.values
        #data = scale(plays)
        data = plays.values
        rows = []
        colList = plays.columns.values.tolist()
        cols = str(', '.join(colList))
        for i in k:
            start = time.time()
            kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels, sample_size=50000)
            end = time.time()
            runtime = (end - start)
            row = {'Features' : cols, 'K' : i, 'Silhouette_Avg' : silhouette_avg, 'Time' : runtime, 'Size' : len(data)};
            rows.append(row)
        output = pd.DataFrame(rows)
        #Will output a csv with the results and date ran
        filename = 'cluster_goodness_' + type + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
        output.to_csv(filename)


#Shows the amount of rows that do not contain a value for the specified columns
def variableFilteringAnalysis(plays, columns):
    rows = []
    for column in columns:
        filteredPlays = filterEmpty(plays, column)
        row = {'Column' : column, 'Size' : len(filteredPlays)}
        rows.append(row)
    output = pd.DataFrame(rows)
    filename = 'column_analysis.csv'
    output.to_csv(filename)



def linReg(plays):
    data = scale(plays)
    x = plays[['DN', 'DIST', 'PLAY #','YARD LN']]
    y = plays['GN/LS']
    x = sm.add_constant(x)
    est = sm.OLS(y, x).fit()
    print est.summary()


def randomForest(plays):
    x = plays[['GN/LS','DN', 'DIST', 'PLAY #','YARD LN']]
    y = plays['CLUSTER']
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, random_state=0)
    clf = DecisionTreeClassifier(max_depth=11)
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    print metrics.accuracy_score(ypred, ytest)
    print metrics.confusion_matrix(ypred, ytest)


def randomForestClassifier(plays):
    x = plays[['GN/LS','DN', 'DIST', 'PLAY #','YARD LN']]
    y = plays['CLUSTER']
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, random_state=0)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=2, max_depth=5)
    rf.fit(Xtrain, ytrain)
    ypred = rf.predict(Xtest)
    print metrics.accuracy_score(ypred, ytest)
    print metrics.confusion_matrix(ypred, ytest)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


def randomForestClassifier_FirstDown(plays):
    plays['Is_Run'] = (plays['PLAY TYPE'] == 'Run')
    perc = 1 - calcFirstDownPercentage(plays)
    print "Baseline Accuracy: " + str(perc)
    #x = plays[['DN', 'DIST', 'PLAY #','ScaledYardLine','Is_Run', 'QTR']]
    #columns = ['DN', 'DIST', 'PLAY #','ScaledYardLine','Is_Run', 'QTR']
    x = plays[['DN', 'DIST', 'PLAY #','ScaledYardLine','Is_Run']]
    columns = ['DN', 'DIST', 'PLAY #','ScaledYardLine','Is_Run']
    y = plays['First_Down']
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, random_state=0)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=2, max_depth=10)
    rf.fit(Xtrain, ytrain)
    ypred = rf.predict(Xtest)
    print 'Accuracy score: ' + str(metrics.accuracy_score(ypred, ytest))
    print metrics.confusion_matrix(ypred, ytest)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(x.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, columns[f], importances[indices[f]]))


def calcFirstDownPercentage(plays):
    firstdowns = plays[plays["First_Down"] == 1]
    perc = (float(len(firstdowns))/float(len(plays)))
    return perc


def calcFirstDowns(plays):
    plays['First_Down'] = (plays['GN/LS'] >= plays['DIST'])


def runpass_firstDownPercentages(plays):
    for j in [1,2,3,4]:
        qtr = plays[plays["QTR"] == j]
        for i in [1,2,3,4]:
            dn = qtr[qtr["DN"] == i]
            runs, passes = filterPlayType(qtr)
            run_perc = calcFirstDownPercentage(runs)
            pass_perc = calcFirstDownPercentage(passes)
            print 'Run, Size: ' + str(len(runs)) + ', Qtr: ' + str(j) + ', DN: ' + str(i) + ', FD%: ' + str(run_perc)
            print 'Pass, Size: ' + str(len(passes)) + ', Qtr: ' + str(j) + ', DN: ' + str(i) + ', FD%: ' + str(pass_perc)



def main():
    plays = pd.DataFrame()
    #plays = loadFiles()
    plays = loadFile('C:\Users\Steves\PycharmProjects\csce474groupproject\plays.csv')
    columns = plays.columns.values.tolist()
    #variableFilteringAnalysis(plays, columns)
    #exportToCSV(plays)
    plays = filterColumns(plays, True)
    plays = filterEmpty(plays, 'DIST')
    plays = filterEmpty(plays, 'DN')
    plays = filterEmpty(plays, 'PLAY #')
    plays = filterEmpty(plays, 'QTR')
    plays = filterEmpty(plays, 'YARD LN')
    #has_quarter = filterEmpty(has_quarter, 'YARD LN')
    #has_quarter_runs, has_quarter_passes = filterPlayType(has_quarter)
    plays = dropColumnsInsufficientData(plays)
    #runs, passes = filterPlayType(plays)
    plays['ScaledYardLine'] = plays['YARD LN'].apply(lambda x: scaleYardLine(x))
    #getSummary(plays)
    print len(plays)
    calcFirstDowns(plays)
    runpass_firstDownPercentages(plays)
    #print list(plays.columns.values)
    #randomForestClassifier_FirstDown(plays)

    #linReg(runs)
    #clusteredRuns = clusterPlays(runs, 8)
    #randomForest(clusteredRuns)
    #randomForestClassifier(clusteredRuns)
    #print len(clusteredRuns)
    #clusterGoodness('runs', runs, [6,7,8])
    #clusterGoodness(passes, [4,5,6,7,8,9,10])
    #clusterGoodness3('runs', runs, [4,5,6,7,8,9,10])
    #clusterGoodness3('passes' , passes, [4,5,6,7,8,9,10])

if __name__ == '__main__':
    main()