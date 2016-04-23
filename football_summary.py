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
import time

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


def loadFile(filename):
    return pd.read_csv(filename)


def getSummary(rows):
    #summary
    print len(rows.index)
    print list(rows)
    print rows.describe()


def exportToCSV(rows):
    #export to csv
    rows = rows.drop('MOTION', 1)
    rows.to_csv("plays.csv")

def filterPlayType(plays):
    runs = pd.DataFrame()
    passes = pd.DataFrame()
    runs = plays[plays['PLAY TYPE'] == 'Run']
    passes = plays[plays['PLAY TYPE'] == 'Pass']
    runs = runs.drop('PLAY TYPE', 1)
    passes = passes.drop('PLAY TYPE', 1)
    return runs, passes

def filterEmpty(plays, columnname):
    plays = plays[plays[columnname].notnull()]
    return plays

def filterColumns(plays):
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
    return plays

def dropColumns(plays):
    plays = plays.drop('MOTION', 1)
    plays = plays.drop('DIST', 1)
    #plays = plays.drop('DN', 1)
    plays = plays.drop('GAP', 1)
    plays = plays.drop('HASH', 1)
    plays = plays.drop('PLAY DIR', 1)
    plays = plays.drop('QTR', 1)
    plays = plays.drop('RESULT', 1)
    plays = plays.drop('YARD LN', 1)
    plays = plays.drop('PLAY #', 1)
    return plays


def clusterPlays(plays, k):
        data = scale(plays)
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
        kmeans.fit(data)
        clusteredPlays = pd.DataFrame(plays, columns=['DN','GN/LS'])
        clusteredPlays.insert(0, 'Cluster', kmeans.labels_)
        return plays


def clusterGoodness(plays, k):
        data = scale(plays)
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
            row = {'Features' : cols, 'K' : i, 'Silhouette_Avg' : silhouette_avg, 'Time' : runtime, 'Size' : len(plays)};
            rows.append(row)
        output = pd.DataFrame(rows)
        filename = 'cluster_goodness_' + time.strftime("%Y-%m-%d_%H-%M-%S") +'.csv'
        output.to_csv(filename)

def variableFilteringAnalysis(plays, columns):
    rows = []
    for column in columns:
        filteredPlays = filterEmpty(plays, column)
        row = {'Column' : column, 'Size' : len(filteredPlays)}
        rows.append(row)
    output = pd.DataFrame(rows)
    filename = 'column_analysis.csv'
    output.to_csv(filename)



def main():
    plays = pd.DataFrame()
    plays = loadFile('C:\Users\Steves\PycharmProjects\csce474groupproject\plays.csv')
    columns = plays.columns.values.tolist()
    #variableFilteringAnalysis(plays, columns)
    #getSummary(plays)
    #exportToCSV(plays)
    plays = filterColumns(plays)
    print len(plays)
    plays = filterEmpty(plays, 'DIST')
    plays = filterEmpty(plays, 'DN')
    plays = filterEmpty(plays, 'PLAY #')
    #plays = filterEmpty(plays, 'QTR')
    plays = filterEmpty(plays, 'YARD LN')
    #has_quarter = filterEmpty(has_quarter, 'YARD LN')
    #has_quarter_runs, has_quarter_passes = filterPlayType(has_quarter)
    plays = dropColumns(plays)
    
    runs, passes = filterPlayType(plays)
    print len(runs)
    temp = list(runs.columns.values)
    print temp
    print runs.dtypes

    temp = clusterPlays(runs, 5)
    #clusterGoodness(data, [5])
    

if __name__ == '__main__':
    main()