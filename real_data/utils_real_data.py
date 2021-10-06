import numpy as np
import scipy as sp
import pandas as pd

import sklearn.cluster as sk_cl
import sklearn.metrics
import scipy.spatial.distance as sp_dist

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


from tqdm import tqdm_notebook
from IPython.core.display import display, HTML

from sklearn.decomposition import PCA as PCA
from sklearn.metrics import adjusted_mutual_info_score as ami__
from sklearn.metrics import adjusted_rand_score as ari__


'''
DEFAULT PARAMS
'''

INIT = 'k-means++'
N_INIT = 50
MAX_ITER = 300
BATCH_SIZE = 100
TOLERANCE = .001
CCORE = True
METRIC = 'euclidean'
AFFINITY = 'euclidean'
LINKAGE = 'complete'

def shrink_(list_of_values, factor):
    
    min_, max_ = min(list_of_values), max(list_of_values)
    percentage_ = list_of_values/max_
    return list_of_values*factor**2*percentage_

def random_splitting(dset, proportion):
    '''
    Input:
        dataset : numpy array [num_points * dims]
        proportion : float in [0,1]
    
    Output:
        ds_1 : numpy_array [int(p*num_points) * dims]
        ds_2 : numpy_array[num_point- int(p*num_points) * dims]
        the union of ds_1, ds_2 gives back the original dataset 
    '''
    np.random.shuffle(dset)
    ds_1, ds_2 = dset[:int(proportion*len(dset))], dset[int(proportion*len(dset)):]
    return ds_1, ds_2

def Algorithm_1(train_set, test_set, method_dict, num_boots):
    '''
    Input:
        train_set : numpy array [len_train_set * dims]
        test_set : numpy array [len_test_set * dims]
        method_dict : dictionary with relevant keys for clustering method in use
        num_boots : int number of bootstrap iterations
    Output:
        r_ari : vector [num_boots] of ari scores
        r_ami : vector [num_boots] of ami scores
    '''    
    boot_size =  int(.8*min(len(train_set), len(test_set)))
    r_ari, r_ami = np.zeros(num_boots), np.zeros(num_boots)
    true_labels_test, predicted_labels_test = np.ones([num_boots, len(test_set)]), np.zeros([num_boots, len(test_set)]) 
    for b in tqdm_notebook(range(num_boots)):
        sub_train, sub_train_indices = subsample(train_set, len(train_set))
        sub_test, sub_test_indices =  subsample(test_set, len(test_set))
        r_ari[b], r_ami[b], true_labels, predicted_labels = repro_score(sub_train, sub_test, method_dict)
        true_labels_test[b] = pad_labels(true_labels, sub_test_indices, test_set)
        predicted_labels_test[b] = pad_labels(predicted_labels, sub_test_indices, test_set)
        
    return r_ari, r_ami, true_labels_test, predicted_labels_test

def pad_labels(list_of_labels, list_of_indices, test_set):
    '''
    Input 
        list_of_labels list or array of ints (len of subsample)
        list_of_indices list or array of ints
        length int
    Output 
        padded_labels list of ints (len of sample >> subsample)
    '''
    length = len(test_set)
    inactive_indices = [i for i in range(length) if i not in list_of_indices]
    padded_labels = -1 * np.ones(length)
    for i in range(len(list_of_indices)):
        padded_labels[list_of_indices[i]] = list_of_labels[i]
    nearest_neighbors = np.argmin(sp_dist.cdist(test_set, test_set[list_of_indices]), axis = 1)
    for i in inactive_indices:
        padded_labels[i] = padded_labels[list_of_indices[nearest_neighbors[i]]]
    return padded_labels

def Algorithm_1_affinity(train_set, test_set, method_dict, num_boots):
    '''
    Input:
        train_set : numpy array [len_train_set * dims]
        test_set : numpy array [len_test_set * dims]
        method_dict : dictionary with relevant keys for clustering method in use
        num_boots : int number of bootstrap iterations
    Output:
        r_ari : vector [num_boots] of ari scores
        r_ami : vector [num_boots] of ami scores
    '''    
    boot_size =  min(len(train_set), len(test_set))
    r_ari, r_ami = np.zeros(num_boots), np.zeros(num_boots)
    k_pred, k_true = np.zeros(num_boots), np.zeros(num_boots)
    true_labels_test, predicted_labels_test = np.ones([num_boots, len(test_set)]), np.zeros([num_boots, len(test_set)]) 
    for b in tqdm_notebook(range(num_boots)):
        sub_train, sub_train_indices = subsample(train_set, boot_size)
        sub_test, sub_test_indices =  subsample(test_set, boot_size)
        r_ari[b], r_ami[b], true_labels, predicted_labels, k_pred[b], k_true[b] = repro_score_affinity(sub_train, sub_test, method_dict)
        true_labels_test[b] = pad_labels(true_labels, sub_test_indices, test_set)
        predicted_labels_test[b] = pad_labels(predicted_labels, sub_test_indices, test_set)
    return r_ari, r_ami, true_labels_test, predicted_labels_test, k_pred, k_true

def subsample(dset, size):
    '''
    Input:
        dataset : numpy array [num_points * dims]
        size : int
    Output:
        subsample : numpy array [size * dims]
        subsample is a bootstrap subsample of dataset
    '''
    subsample_indices = np.random.choice(len(dset), size=int(size), replace=False)
    subsample = np.asarray([dset[i] for i in subsample_indices])
    return subsample, subsample_indices

def repro_score(train_set, test_set, method_dict):
    '''
    Input:
        train_set : numpy array [len_train_set * dims]
        test_set : numpy array [len_test_set * dims]
        method_dict : dictionary with relevant keys for clustering method in use
    Output:
        ari : ari repro score
        ami : ami repro score
    '''
    true_labels, predicted_labels = fit_clustering(train_set, test_set, method_dict)
    
    ari = sklearn.metrics.adjusted_rand_score(predicted_labels, true_labels)
    ami = sklearn.metrics.adjusted_mutual_info_score(predicted_labels, true_labels)
    
    return ari, ami, true_labels, predicted_labels

def repro_score_affinity(train_set, test_set, method_dict):
    '''
    Input:
        train_set : numpy array [len_train_set * dims]
        test_set : numpy array [len_test_set * dims]
        method_dict : dictionary with relevant keys for clustering method in use
    Output:
        ari : ari repro score
        ami : ami repro score
    '''
    true_labels, predicted_labels, k_pred, k_true = fit_clustering(train_set, test_set, method_dict)
    
    ari = sklearn.metrics.adjusted_rand_score(predicted_labels, true_labels)
    ami = sklearn.metrics.adjusted_mutual_info_score(predicted_labels, true_labels)
    
    return ari, ami, true_labels, predicted_labels, k_pred, k_true



def fit_clustering(train_set, test_set, method_dict):
    
    method_name = method_dict['name']
    
    if method_name == 'AffinityPropagation':
        true_labels, predicted_labels, k_pred, k_true = fit_clustering_affinity(train_set, test_set, method_dict)
        return true_labels, predicted_labels, k_pred, k_true
        
    if method_name == 'AgglomerativeClustering':
        true_labels, predicted_labels = fit_clustering_agglomerative(train_set, test_set, method_dict)

    if method_name == 'Birch':
        true_labels, predicted_labels = fit_clustering_birch(train_set, test_set, method_dict)

    if method_name == 'Kmeans':
        true_labels, predicted_labels = fit_clustering_kmeans(train_set, test_set, method_dict)
        
    if method_name == 'MiniBatchKMeans':
        true_labels, predicted_labels = fit_clustering_mbkmeans(train_set, test_set, method_dict)
        
    if method_name == 'Kmedians':
        true_labels, predicted_labels = fit_clustering_kmedians(train_set, test_set, method_dict)
        
    if method_name == 'MeanShift':
        true_labels, predicted_labels = fit_clustering_mean_shift(train_set, test_set, method_dict)
        
    return true_labels, predicted_labels

def fit_clustering_affinity(train_set, test_set, method_dict):
    
    fit_tr = sk_cl.AffinityPropagation().fit(train_set)
    predicted_labels = fit_tr.predict(test_set)
    true_labels = sk_cl.AffinityPropagation().fit_predict(test_set)
    k_pred, k_true = len(set(predicted_labels)), len(set(true_labels))
    
    return true_labels, predicted_labels, k_pred, k_true

def fit_clustering_agglomerative(train_set, test_set, method_dict):
    
    fit_tr = sk_cl.AgglomerativeClustering(n_clusters= method_dict['n_clusters'] , affinity= method_dict['affinity'], linkage = method_dict['linkage']).fit_predict(train_set)
    predicted_labels = label_induced_by_nearest_point(test_set, train_set, fit_tr, 'euclidean')
    true_labels = sk_cl.AgglomerativeClustering().fit_predict(test_set)
        
    return true_labels, predicted_labels


def fit_clustering_birch(train_set, test_set, method_dict):
    
    n_clusters = method_dict['n_clusters']
    
    fit_tr = sk_cl.Birch(n_clusters=n_clusters).fit(train_set)
    predicted_labels = fit_tr.predict(test_set)
    true_labels = sk_cl.Birch(n_clusters=n_clusters).fit_predict(test_set)
        
    return true_labels, predicted_labels


def fit_clustering_kmeans(train_set, test_set, method_dict):
    
    n_clusters, init, n_init, max_iter = method_dict['n_clusters'], method_dict['init'], method_dict['n_init'], method_dict['max_iter']

    fit_tr = sk_cl.KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter).fit(train_set)
    predicted_labels = fit_tr.predict(test_set)
    true_labels = sk_cl.KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter).fit_predict(test_set)
        
    return true_labels, predicted_labels

def fit_clustering_kmedians(train_set, test_set, method_dict):
    
    intial_centers_tr, initial_centers_te, tolerance, ccore = method_dict['intial_centers_tr'], method_dict['intial_centers_te'], method_dict['tolerance'], method_dict['ccore']
    metric = method_dict['metric']
    fit_tr = py_cl_km.kmedians(data = train_set, initial_centers= intial_centers_tr, tolerance=tolerance, ccore=ccore)
    fit_tr.process()

    tr_meds = fit_tr.get_medians()
        
    fit_te = py_cl_km.kmedians(data = test_set, initial_centers= initial_centers_te, tolerance=tolerance, ccore=ccore)
    fit_te.process()
    te_meds = fit_te.get_medians()
    
    true_labels = nearest_centroids(test_set, te_meds, metric)
    predicted_labels = nearest_centroids(test_set, tr_meds, metric)

    return true_labels, predicted_labels


def fit_clustering_mbkmeans(train_set, test_set, method_dict):
    
    n_clusters, init, max_iter, batch_size = method_dict['n_clusters'], method_dict['init'], method_dict['batch_size'], method_dict['max_iter']
#     print(init, max_iter, batch_size)
    fit_tr = sk_cl.MiniBatchKMeans(n_clusters=n_clusters, init=init, batch_size = batch_size, max_iter=max_iter).fit(train_set)
    predicted_labels = fit_tr.predict(test_set)
    true_labels = sk_cl.MiniBatchKMeans(n_clusters=n_clusters, init=init, batch_size = batch_size, max_iter=max_iter).fit_predict(test_set)
        
    return true_labels, predicted_labels

def fit_clustering_mean_shift(train_set, test_set, method_dict):
        
    fit_tr = sk_cl.MeanShift().fit(train_set)
    predicted_labels = fit_tr.predict(test_set)
    true_labels = sk_cl.MeanShift().fit_predict(test_set)
#     print(len(true_labels), len(predicted_labels))
    return true_labels, predicted_labels

def nearest_centroids(dataset, centroids, metric):
    '''
    distance can be any of :
        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, 
        ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’,
        ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
        ‘sqeuclidean’, ‘wminkowski’, ‘yule’
    '''
    cdist = sp_dist.cdist(dataset, centroids, metric = metric)
    
    return np.argmin(cdist, axis = 1)

def label_induced_by_nearest_point(test_set, train_set, labels_train_set, metric):
    '''
    Input :
        test_set np.array(N*d) of test_set_points
        train_set np.array(M*d) of train_set_points
        labels_train_set np.array(M*1) of integers -- labels
        metric : str; can be any of :
            ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, 
            ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’,
            ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
            ‘sqeuclidean’, ‘wminkowski’, ‘yule’
    Output :
        labels_test_set : np.array(M*1) of ints
    '''
    
    nearest_neighbors_indices = np.argmin(sp_dist.cdist(test_set, train_set, metric = metric), axis = 1)
    labels_test_set = np.asarray([labels_train_set[nni] for nni in nearest_neighbors_indices])
    
    return labels_test_set
    
    


def default_dictionary(method_name, **kwargs):
    
    if method_name == 'AffinityPropagation':
        default_dictionary = default_dictionary_clustering_affinity(**kwargs)
        
    if method_name == 'AgglomerativeClustering':
        default_dictionary = default_dictionary_clustering_agglomerative(**kwargs)

    if method_name == 'Birch':
        default_dictionary = default_dictionary_birch(**kwargs)

    if method_name == 'Kmeans':
        default_dictionary = default_dictionary_kmeans(**kwargs)
        
    if method_name == 'MiniBatchKMeans':
        default_dictionary = default_dictionary_mbkmeans(**kwargs)
        
    if method_name == 'Kmedians':
        default_dictionary = default_dictionary_kmedians(**kwargs)
        
    if method_name == 'MeanShift':
        default_dictionary = default_dictionary_mean_shift(**kwargs)

    if method_name == 'DBSCAN':
        default_dictionary = default_dictionary_dbscan(**kwargs)
        
    return default_dictionary
 
def default_dictionary_clustering_affinity():
    
    default_dic = {}
    default_dic['name'] = 'AffinityPropagation'
    return default_dic

def default_dictionary_clustering_agglomerative(affinity, linkage, n_clusters):
    if linkage == None:
        linkage = 'ward'
    default_dic = {}
    default_dic['name'] = 'AgglomerativeClustering'
    default_dic['affinity'] = affinity
    default_dic['linkage'] = linkage
    default_dic['n_clusters'] = n_clusters
    return default_dic

def default_dictionary_birch(n_clusters):
    
    default_dic = {}
    default_dic['name'] = 'Birch'
    default_dic['n_clusters'] = n_clusters
    return default_dic
    
def default_dictionary_kmeans(n_clusters, init, n_init, max_iter):

    default_dic = {}
    default_dic['name'] = 'Kmeans'
    default_dic['n_clusters'] = n_clusters
    default_dic['init'] = init
    default_dic['n_init'] = n_init
    default_dic['max_iter'] = max_iter
    
    return default_dic

def default_dictionary_kmedians(initial_centers, n_clusters, train_set, test_set , tolerance, ccore):
    default_dic = {}
    dims_sets = (len(train_set), len(test_set))
    default_dic['name'] = 'Kmedians'
    if initial_centers == 'random':
        ind_train, ind_test = np.random.choice(dims_sets[0], replace= True, size=n_clusters), np.random.choice(dims_sets[1], replace= True, size=n_clusters)
        default_dic['intial_centers_tr'], default_dic['intial_centers_te'] = train_set[ind_train, :], test_set[ind_test, :]
    default_dic['tolerance'], default_dic['ccore'] = TOLERANCE, CCORE
    default_dic['metric'] = METRIC

    return default_dic

def default_dictionary_mbkmeans(n_clusters, init, max_iter, batch_size):
    default_dic = {}
    default_dic['name'] = 'MiniBatchKMeans'
    default_dic['n_clusters'], default_dic['init'], default_dic['batch_size'], default_dic['max_iter'] = n_clusters, init, max_iter, batch_size 
    return default_dic


def default_dictionary_mean_shift():
    
    default_dic = {}
    default_dic['name'] = 'MeanShift'
    return default_dic

def default_dictionary_DBSCAN():
    
    default_dic = {}
    default_dic['name'] = 'DBSCAN'
    return default_dic

def run_exercise_1(train_set, test_set, method_name, num_boots, max_cluster_number, affinity, linkage):
    
    if method_name == 'AffinityPropagation':
        ari, ami, true_labels, predicted_labels, k_pred, k_true = run_exercise_1_clustering_affinity(train_set, test_set, num_boots)
        return ari, ami, true_labels, predicted_labels, k_pred, k_true 
        
    if method_name == 'AgglomerativeClustering':
        ari, ami, true_labels, predicted_labels = run_exercise_1_clustering_agglomerative(train_set, test_set, num_boots, affinity, linkage, max_cluster_number)

    if method_name == 'Birch':
        ari, ami, true_labels, predicted_labels = run_exercise_1_birch(train_set, test_set, num_boots, max_cluster_number)

    if method_name == 'Kmeans':
        ari, ami, true_labels, predicted_labels = run_exercise_1_kmeans(train_set, test_set, num_boots, max_cluster_number)
        
    if method_name == 'MiniBatchKMeans':
        ari, ami, true_labels, predicted_labels = run_exercise_1_mbkmeans(train_set, test_set, num_boots, max_cluster_number)
        
    if method_name == 'Kmedians':
        #ari, ami = run_exercise_1_kmedians(train_set, test_set, num_boots, max_cluster_number)
        ari, ami, true_labels, predicted_labels = np.zeros(num_boots), np.zeros(num_boots)
    if method_name == 'MeanShift':
        ari, ami, true_labels, predicted_labels = run_exercise_1_mean_shift(train_set, test_set, num_boots)
    if method_name == 'DBSCAN':
        
        ari, ami, true_labels, predicted_labels = run_exercise_1_dbscan(train_set, test_set, num_boots)        
    
        
    return ari, ami, true_labels, predicted_labels
    
    
    
def run_exercise_1_clustering_affinity(train_set, test_set, num_boots):
    
    method_dict = default_dictionary_clustering_affinity()
    ari, ami, true_labels, predicted_labels, k_pred, k_true = Algorithm_1_affinity(train_set, test_set, method_dict, num_boots)
    return ari, ami, true_labels, predicted_labels, k_pred, k_true

def run_exercise_1_clustering_agglomerative(train_set, test_set, num_boots,  affinity, linkage, max_cluster_number):
    
    ari, ami = np.zeros([max_cluster_number-1, num_boots]), np.zeros([max_cluster_number-1, num_boots])
    for c_, cl in enumerate(range(2, max_cluster_number+1)):
        #print('Number of Clusters : ', cl)
        method_dict = default_dictionary_clustering_agglomerative(affinity, linkage, cl)
        ari[c_,:], ami[c_, :], true_labels, predicted_labels = Algorithm_1(train_set, test_set, method_dict, num_boots)
    return ari, ami, true_labels, predicted_labels

def run_exercise_1_birch(train_set, test_set, num_boots, max_cluster_number):
    
    ari, ami = np.zeros([max_cluster_number-1, num_boots]), np.zeros([max_cluster_number-1, num_boots])
    for c_, cl in enumerate(range(2, max_cluster_number+1)):
        #print('Number of Clusters : ', cl)
        method_dict = default_dictionary_birch(cl)
        ari[c_,:], ami[c_, :], true_labels, predicted_labels = Algorithm_1(train_set, test_set, method_dict, num_boots)
    return ari, ami, true_labels, predicted_labels
 
def run_exercise_1_kmeans(train_set, test_set, num_boots, max_cluster_number):
    #print('Method : KMeans')
    ari, ami = np.zeros([max_cluster_number-1, num_boots]), np.zeros([max_cluster_number-1, num_boots])
    true_labels, predicted_labels = np.zeros([max_cluster_number-1, num_boots, len(test_set)]), np.zeros([max_cluster_number-1, num_boots, len(test_set)])
    for c_, cl in enumerate(range(2, max_cluster_number+1)):
        #print('Number of Clusters : ', cl )
        method_dict = default_dictionary_kmeans(n_clusters = cl, init = INIT, n_init = N_INIT, max_iter = MAX_ITER)
        ari[c_,:], ami[c_, :], true_labels[c_,:], predicted_labels[c_,:] = Algorithm_1(train_set, test_set, method_dict, num_boots)
    return ari, ami, true_labels, predicted_labels

def run_exercise_1_mbkmeans(train_set, test_set, num_boots, max_cluster_number):
    #print('Method : Mini Batch KMeans')
    ari, ami = np.zeros([max_cluster_number-1, num_boots]), np.zeros([max_cluster_number-1, num_boots])
    for c_, cl in enumerate(range(2, max_cluster_number+1)):
        #print('Number of Clusters : ', cl )
        method_dict = default_dictionary_mbkmeans(n_clusters = cl, init = INIT, max_iter = MAX_ITER, batch_size = BATCH_SIZE)
        ari[c_,:], ami[c_, :], true_labels, predicted_labels =  Algorithm_1(train_set, test_set, method_dict, num_boots)

def run_exercise_1_kmedians(train_set, test_set, num_boots, max_cluster_number):
    #print('Method : K Medians')
    ari, ami = np.zeros([max_cluster_number-1, num_boots]), np.zeros([max_cluster_number-1, num_boots])
    for c_, cl in enumerate(range(2, max_cluster_number+1)):
        #print('Number of Clusters : ', cl )
        method_dict = default_dictionary_kmedians(initial_centers = 'random', n_clusters = cl, train_set = train_set, test_set = test_set, tolerance = TOLERANCE, ccore =  CCORE)
        ari[c_,:], ami[c_, :], true_labels, predicted_labels =  Algorithm_1(train_set, test_set, method_dict, num_boots)
    return ari, ami, true_labels, predicted_labels

def run_exercise_1_kmedians(train_set, test_set, num_boots, max_cluster_number):
    #print('Method : K Medians')
    ari, ami = np.zeros([max_cluster_number-1, num_boots]), np.zeros([max_cluster_number-1, num_boots])
    for cl in range(2, max_cluster_number+1):
        #print('Number of Clusters : ', cl )
        method_dict = default_dictionary_kmedians(initial_centers = 'random', n_clusters = cl, train_set = train_set, test_set = test_set, tolerance = TOLERANCE, ccore =  CCORE)
        ari[cl-2,:], ami[cl-2, :], true_labels, predicted_labels =  Algorithm_1(train_set, test_set, method_dict, num_boots)
    return ari, ami, true_labels, predicted_labels

def run_exercise_1_mean_shift(train_set, test_set, num_boots):
    #print('Method : Mean Shift')
    method_dict = default_dictionary_mean_shift()
    ari, ami, true_labels, predicted_labels =  Algorithm_1(train_set, test_set, method_dict, num_boots)
    return ari, ami, true_labels, predicted_labels

def run_exercise_1_clustering_agglomerative(train_set, test_set, num_boots,  affinity, linkage, max_cluster_number):
    
    ari, ami = np.zeros([max_cluster_number-1, num_boots]), np.zeros([max_cluster_number-1, num_boots])
    for c_, cl in enumerate(range(2, max_cluster_number+1)):
        #print('Number of Clusters : ', cl)
        method_dict = default_dictionary_clustering_agglomerative(affinity, linkage, cl)
        ari[c_,:], ami[c_, :], true_labels, predicted_labels = Algorithm_1(train_set, test_set, method_dict, num_boots)
    return ari, ami, true_labels, predicted_labels   

