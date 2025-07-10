import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone 
from sklearn.datasets import fetch_openml

def get_mnist(tr_size=10000):

    mnist = fetch_openml('mnist_784', version=1)

    X, y = mnist['data'], mnist['target']
    y =  y.astype(np.uint8) # cast string category type (i.e. '1', '2' ) to integers (i.e. 1 , 2 )
    return X[:tr_size], X[tr_size:], y[:tr_size], y[tr_size:] # pre-shuffled dataset 

def plot_image(img_data):
    plt.figure()

    plt.imshow(img_data.reshape(28, 28),  cmap='binary')

    # remove ticks
    plt.xticks(ticks=[], labels=[])
    plt.yticks(ticks=[], labels=[])


def plot_images(data_x: pd.DataFrame , n: np.int32 = 5):

    fig, axes = plt.subplots(nrows= n, ncols=n )

    for r in range(n):

        for c in range(n):
            ax =  axes[r, c]
            idx = r*n + c 
            ax.imshow( data_x.iloc[[idx]].to_numpy().reshape(28, 28),  cmap='binary' )
            
            # remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # remove border 
            ax.set_frame_on(False)


    plt.subplots_adjust(hspace=0.2, wspace=0.2) # horizontal, width space

    plt.show()

def custom_cross_val(x_train, y_train, model):
    
    skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idices, test_idices in skfolds.split(x_train, y_train):

        model = clone(model)

        # create train, test folds 

        x_train_folds = x_train.iloc[train_idices]
        
        y_train_folds = y_train.iloc[train_idices]
        
        x_test_folds = x_train.iloc[test_idices]

        y_test_folds = y_train.iloc[test_idices]

        # train 

        model.fit(x_train_folds, y_train_folds)

        # predict  

        y_pred = model.predict(x_test_folds)

        # test section cross-val

        correct_count = sum(y_pred == y_test_folds)

        # ratio of correct predictions (i.e. accuracy)
        
        print(correct_count/len(y_pred),end='\t')
    
    print()

def precision_vs_recall_vs_threshold(precisions, recalls, thresholds, threshold_markers=[]):
    print(threshold_markers)
    plt.figure(1)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.legend(loc="center right")
    plt.ylabel("Recall")
    plt.xlabel("Threshold")
    plt.xlim(-40000, 40000)
    plt.grid(visible=True, axis="both")

    # marker 
    for marker_data in threshold_markers:
        x_threshold = marker_data[0]
        y_recall = marker_data[1]
        y_precision = marker_data[2]
        
        # point
        plt.plot(x_threshold, y_recall, marker="o")
        plt.plot(x_threshold, y_precision, marker="o")

        # vlines
        plt.vlines( ls ='--', x=[x_threshold, x_threshold], ymin=[0,0], ymax=[y_recall, y_precision] )

        # hlines
        plt.hlines( ls ='--',  xmin = [-40000,-40000], xmax = [x_threshold, x_threshold], y = [y_recall, y_precision] )


    plt.figure(2)
    plt.plot(recalls[:-1], precisions[:-1], "b--", label="Precision")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.grid(visible=True, axis="both")

def plot_roc_curve(fpr=[], tpr=[],labels=[], trace_color= []):
    plt.figure(3)
    for i in range(len(fpr)):
        plt.plot(fpr[i], tpr[i], trace_color[i], label=labels[i])
    plt.legend(bbox_to_anchor=[0, 0.5])
    plt.ylabel("True Positive Rate(Recall)")
    plt.xlabel("False Positive Rate")
    plt.grid(visible=True, axis="both")
   
