import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm_notebook
import matplotlib

import sklearn.cluster as sk_cl
import sklearn.metrics
import scipy.spatial.distance as sp_dist


# PLOTTING

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import  DBSCAN



from sklearn.decomposition import PCA as PCA
from sklearn.metrics import adjusted_mutual_info_score as ami__
from sklearn.metrics import adjusted_rand_score as ari__


import warnings
warnings.filterwarnings("ignore")

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


def repro_plot_all(train_set, test_set, train_point_ls, block_train_ls, num_clusters, num_boot, boot_size, markers, save, colorbar):
    
#     fig = plt.figure(figsize = (18,12))
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (24,10))
    num_plots = len(train_point_ls)
    title_ = ['(a)', '(b)', '(c)']
    n_train = len(train_set)
    
    p = 0
    for p in range(num_plots):
        special_point = train_point_ls[p]
        block_train = block_train_ls[p]
        
        if block_train == 0:
            marker, color = 'v', 'blue'
        if block_train == 1:
            marker, color = '^', 'firebrick'
        if block_train == 2:
            marker, color = 'x', 'orange'
           
    
        score_ari, score_ami, binary_labels_train, binary_labels_test = reproducibility_algo_individual(np.asarray([train_set, test_set]), special_point, 3,  num_boot, boot_size, 'Kmeans')
        mean_binary_labels_train, mean_binary_labels_test = binary_labels_train.mean(axis = 0), binary_labels_test.mean(axis = 0)
        levels = MaxNLocator(nbins=15).tick_values(0,1)
        #cmap = plt.get_cmap('RdYlGn')
        cmap = plt.get_cmap('cool')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        ax = plt.subplot(2, num_plots, p+1)
        
#         ax.scatter(special_point[0], special_point[1], marker = marker, color = 'k', facecolors='none', linewidths=5, s = 1000)

        for _, test_point in enumerate(test_set):
            ax.scatter(test_set[_,0],test_set[_,1],c=mean_binary_labels_train[_], s=100, marker = markers[_], alpha = .75, cmap = cmap, vmin = 0, vmax =1)
            
        ax.scatter(special_point[0], special_point[1], marker = marker, color = 'k', linewidths=3, s = 300)


        ax.set_ylim(-3.5,3.5)
        ax.set_xlim(-3.5,3.5)
        
        
        if p == 0:
            ax.set_ylabel('Cross'+'\n'+r'$\tilde{\Psi}_{x}(\cdot; \mathbf{X}^{(b)})$', fontsize = 40)
        
        ax.tick_params(axis=u'both', which=u'both',length=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        
        ax1 = plt.subplot(2, num_plots, num_plots + p +1)
        

        
        scatter = ax1.scatter(test_set[:,0],test_set[:,1],c=mean_binary_labels_test,s=1, cmap = cmap, vmin = 0, vmax =1)
        
        for _, test_point in enumerate(test_set):
            ax1.scatter(test_set[_,0],test_set[_,1],c=mean_binary_labels_test[_], s=100, marker = markers[_], alpha = .75, cmap = cmap, vmin = 0, vmax =1)
            
        ax1.scatter(special_point[0], special_point[1], marker = marker, color = 'k', linewidths=3, s = 300)


        ax1.set_ylim(-3.5,3.5)
        ax1.set_xlim(-3.5,3.5)
        if p == 0:
            ax1.set_ylabel('Self'+'\n'+r'$\tilde{\Psi}_{x}(\cdot; \mathbf{X}^{\prime(b)})$', fontsize = 40)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.tick_params(axis='both', which='both', length=0)
        
        ax.set_title('ARI: '+str(score_ari.mean())[:4]+' $\pm$ '+str(score_ari.std())[:4], x = .5, fontsize = 50)

        ax1.set_xlabel(title_[p], fontsize = 50)
    axins1 = inset_axes(ax,
                    width="5%",  # width = 50% of parent_bbox width
                    height="200%", #,  # height : 5%
                    bbox_to_anchor=(.05, 1.1, 1.1, 1.1), #,
                    bbox_transform=ax1.transAxes) #,
                    #loc='center')

    cbar = fig.colorbar(scatter, cax=axins1, orientation="vertical", ticks=[0, .25, .5, .75, 1])        
    cbar.ax.tick_params(labelsize=30)
    if save!= False:
        plt.savefig(save+'.pdf', dpi = 1000)
    p+=1
    plt.show()    


def my_formatter_fun(x, p):
    return "%.0f" % (x * (10 ** scale_pow))
scale_pow = -3

def shrink_(list_of_values, factor):
    
    min_, max_ = min(list_of_values), max(list_of_values)
    percentage_ = list_of_values/max_
    return list_of_values*factor**2*percentage_


def generate_set(indices, centroids, sigma):
    
    '''
    Input:
        indices : array of ints (class assignments)
        centroids : array  of means
        sigma : array of covariance matrices
    Output :
        dataset : array of points
    '''
    
    n_points, d = len(indices), np.shape(centroids)[-1]
    dataset = np.zeros([n_points, d])

    for n in range(n_points):
        dataset[n] = np.random.multivariate_normal(centroids[indices[n]], sigma * np.eye(d))
    return dataset


def reproducibility_algo_individual(datasets, special_point, n_clusters, num_boot, boot_size, algorithm):
    '''
    Takes as input:
        datasets
        train_point_index : int -- which datapoint we wan to test against
        n_clusters : int
        num_boot : int (number of bootstrap draws B_1)
        boot_size : int (size N(b) of each bootstrap draw b)
        num_points : int size of original datasets
        dims : int dimension of each datapoint
        algorithm : 
            - k_means 
            - gmm
            - dp gmm
            - k_medians (only works for n_clusters 3)
            
        metric :
            - ami
            - ari
        '''  
    
    train_set, test_set = datasets[0], datasets[1]
    num_points_train, num_points_test, dims = len(train_set), len(test_set), train_set.shape[1]
    score_ari, score_ami,  binary_labels_train, binary_labels_test = np.zeros([num_boot]), np.zeros([num_boot]), np.zeros([num_boot, num_points_test]), np.zeros([num_boot, num_points_test])
    
    
    for b in tqdm_notebook(range(num_boot)):
        # bootstrap draws from two train and test datasets
        
        r_ari, r_ami, binary_labels_train_, binary_labels_test_ = reproducibility_algo_ind_sub_step(train_set, test_set, boot_size, special_point,  n_clusters, algorithm)
        score_ari[b], score_ami[b] = r_ari, r_ami
        binary_labels_train[b] = binary_labels_train_
        binary_labels_test[b] = binary_labels_test_
        
    return score_ari, score_ami, binary_labels_train, binary_labels_test

def reproducibility_algo_ind_sub_step(train_set, test_set, boot_size, special_point, n_clusters, algorithm):
    '''
    Takes as input :
        train_set : np array (size of bootstrapped sample * dimension of datapoint)
        test_set : np array (size of bootstrapped sample * dimension of datapoint)
        n_clusters : int
        algorithm: k_means, k_medians *** k_medians is implemented only for the case n_clusters = 3 and has prior means as hyperparam
        metric: ari, ami
    Returns :
        score : np array 
        baseline : 
    '''
    train_index,  test_index = np.random.choice(range(len(train_set)), boot_size), np.random.choice(range(len(test_set)), boot_size)
        #if train_point_index not in train_index:
        #    train_index = np.append(train_index, train_point_index)
    sub_train_set, sub_test_set = np.asarray([train_set[i] for i in train_index]), np.asarray([test_set[i] for i in test_index])

    if algorithm == 'Kmeans':
        
        kmeans = sklearn.cluster.KMeans(n_clusters, init='k-means++').fit(sub_train_set)
        test_labels_predicted = kmeans.predict(test_set) # predicted labels on test using centroids learnt on train
        kmeans_test = sklearn.cluster.KMeans(n_clusters, init='k-means++').fit(sub_test_set) # run k_means on test_set
        test_labels = kmeans_test.predict(test_set) # centroids and labels of test set
        
        special_class_train = kmeans.predict(special_point.reshape(1,-1))[0]
        special_class_test = kmeans_test.predict(special_point.reshape(1,-1))[0]
    
    binary_labels_train = binarize(special_class_train, test_labels_predicted) #, sub_test_index, n_test)
    binary_labels_test = binarize(special_class_test, test_labels) #, sub_test_index, n_test)
    
    
    colors_test = {}
    colors_test[0] = 'red'
    colors_test[1] = 'green'

    score_ari = sklearn.metrics.adjusted_rand_score(binary_labels_train[test_index], binary_labels_test[test_index])
    score_ami = sklearn.metrics.adjusted_mutual_info_score(binary_labels_train[test_index], binary_labels_test[test_index])
        
    return score_ari, score_ami, binary_labels_train, binary_labels_test


def binarize(class_chosen, partition):
    
    '''
    Input :
        class_chosen : int of class of point wrt which we want to binarize
        partition: original partition
    Output :
        binary_partition
    '''
    
    binary_partition = -1 * np.ones(len(partition))
    for i, j in enumerate(range(len(partition))):
        binary_partition[j] = (partition[i] == class_chosen)

    return binary_partition.astype(int)

def plot_synth_1(results_ex_1, methods_list, dataset_ids, lengths_ls, true_k, save):
    matplotlib.rc('xtick', labelsize=40) 
    matplotlib.rc('ytick', labelsize=40) 
    dataset_id_ = ['A', 'B', 'C', 'D']
    fig = plt.figure(figsize=(40,22))
    for d_, dataset_id in enumerate(dataset_ids):
        scores = results_ex_1[dataset_id]
        leng = lengths_ls[d_]
        
        #################
        plt.subplot(len(dataset_ids),3,1+d_*3)
#         plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        plt.title(dataset_id, fontsize=60)
        dataset = np.loadtxt('Data/Shapes/'+dataset_id+'.txt')
        train, test = random_splitting(dataset, .5)
        plt.scatter(train[:,0], train[:,1], color = 'red', marker = 'o', s = 20, label = 'train set')
        plt.scatter(test[:,0], test[:,1], color = 'blue', marker = 'x', s = 20, label = 'test set')
        
        plt.ylabel(dataset_id_[d_], fontsize = 80)
        if d_ == 0:
            plt.ylim([min(dataset[:,1])- 2, max(dataset[:,1])*1.4])
#             plt.legend(fontsize = 35, loc = 'upper right')
            
        ################################
        plt.ticklabel_format(style='sci',scilimits=(1,1),axis='both')
        plt.subplot(len(dataset_ids),3,2 + d_*3)
        if leng == 'all':
            len_k = len(scores['Birch']['mean_ari'])
        else:
            len_k = leng
        
        s = int(np.ceil(len_k/7))

        for method in methods_list:
            if method == 'AffinityPropagation':
                plt.errorbar(x = range(len_k), y = [scores[method]['mean_ari'] for k in range(len_k)], yerr = [scores[method]['std_ari'] for k in range(len_k)],  linewidth  =5)
            else:
                plt.errorbar(x = range(len_k), y = [scores[method]['mean_ari'][k] for k in range(len_k)], yerr = [scores[method]['std_ari'][k] for k in range(len_k)],  linewidth  =5)
        if true_k[d_] != None:
            plt.vlines(x = true_k[d_] - 2, ymin = 0, ymax = 1, linestyles='--', color = 'k', linewidth=6, alpha = .75)
        
        if dataset_id == 'spiral':
            mean_ari, std_ari =  results_ex_1['spiral']['DBSCAN']['mean_ari'], results_ex_1['spiral']['DBSCAN']['std_ari']
            plt.errorbar(x = range(len_k), y = mean_ari[:len_k], yerr = std_ari[:len_k],  linewidth =5, color = 'purple')

            
        plt.ylim([0,1.05])
        
        if d_ == len(dataset_ids)-1:
            plt.xlabel(r'Number of clusters', fontsize = 40)
        plt.xticks(range(len_k)[::s],range(2,len_k+2)[::s]) 
        plt.yticks(np.linspace(0,1,6))
        if  d_ == 0:
            plt.title('ARI', fontsize = 60, y = 1.02)
        
        ################################
        
        plt.subplot(len(dataset_ids),3,3 + d_*3)
        plt.ticklabel_format(style='sci',scilimits=(2,2),axis='both')
        for method in methods_list:
            if method == 'AffinityPropagation':
                plt.errorbar(x = range(len_k), y = [scores[method]['mean_ami'] for k in range(len_k)], yerr = [scores[method]['std_ami'] for k in range(len_k)], linewidth  = 5,  label = 'AffProp')
            elif method == 'AgglomerativeClustering':
                plt.errorbar(x = range(len_k), y = [scores[method]['mean_ami'][k] for k in range(len_k)], yerr = [scores[method]['std_ami'][k]  for k in range(len_k)],  linewidth  = 5, label  = 'Ag.Clust.')
            elif method == 'MiniBatchKMeans':
                plt.errorbar(x = range(len_k), y = [scores[method]['mean_ami'][k] for k in range(len_k)], yerr = [scores[method]['std_ami'][k]  for k in range(len_k)],  linewidth  = 5, label  = 'MBKM')
            
            else:
                plt.errorbar(x = range(len_k), y = [scores[method]['mean_ami'][k] for k in range(len_k)], yerr = [scores[method]['std_ami'][k]  for k in range(len_k)],  linewidth  = 5, label = method)
        if true_k[d_]  != None:
        
            plt.vlines(x = true_k[d_] - 2, ymin = 0, ymax = 1, linestyles='--', color = 'k', linewidth=6, alpha = .75, label = r'True $k$')

        if dataset_id == 'spiral':
            mean_ari, std_ari =  results_ex_1['spiral']['DBSCAN']['mean_ami'], results_ex_1['spiral']['DBSCAN']['std_ami']
            plt.errorbar(x = range(len_k), y = mean_ari[:len_k], yerr = std_ari[:len_k],  linewidth =5, color = 'purple', label = 'DBSCAN')
            
        plt.ylim([0,1.05])
        plt.yticks(np.linspace(0,1,6), [0.0,.2,.4,.6,.8,1.0])
#         if d_ == 0:
#             plt.title('AMI', fontsize = 40, y  =1.02)
#             plt.legend(loc = 'lower right', fontsize = 25)
        plt.xticks(range(len_k)[::s],range(2,len_k+2)[::s])
        if d_ == 0:
            plt.title('AMI', fontsize = 60, y  =1.02)
        if d_ == 0:
            plt.legend(fontsize = 35, loc = 'lower right', ncol = 2)
        if d_ == len(dataset_ids)-1:
            
#             plt.legend(loc = 'lower right', fontsize = 25)
            plt.xlabel(r'Number of clusters', fontsize = 40)
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', dpi=1000,  bbox_inches = 'tight')        
    plt.show()

    
def plot_synth_2(results_ex_2, num_sam, perm, clust, save):    
    cmap = plt.get_cmap('cool')
    matplotlib.rc('xtick', labelsize = 20) 
    matplotlib.rc('ytick', labelsize = 20) 
    fig = plt.figure(figsize = (18,12))
    met_ls = ['Kmeans', 'AffinityPropagation', 'Birch', 'MiniBatchKMeans']#, 'AgglomerativeClustering']
    for j,j_c in enumerate(clust):
        for i,method in enumerate(met_ls):
            mean_ari = results_ex_2[num_sam][perm][method]['mean_ari'][:,:,j_c]
            std_ari = results_ex_2[num_sam][perm][method]['std_ari'][:,:,j_c]
            l_c = len(clust)
            plt.subplot(l_c,len(met_ls), l_c*j + i+1)
            im = plt.imshow(mean_ari, vmin=0, vmax=1, cmap=cmap)

            if j == 0:
                plt.title(method, fontsize = 15)
            plt.xticks(range(4),np.arange(1,5))
            plt.yticks(range(4),np.arange(1,5))
            #if i == 0:
            #    if j_c == 15:
            #        plt.ylabel(r'$K = $'+str(j_c+1)+' (TRUE)', fontsize = 15, color = 'red')
            #    else:
            #        plt.ylabel(r'$K = $'+str(j_c+1), fontsize = 15)
                    
            if j == 3:
                plt.xlabel(r'Testing dataset', fontsize = 15)
            if i ==0:
                if j == 2:
                    plt.ylabel(r'$k = $'+str(j_c+1)+'\n'+ 'Training dataset', color = 'r', fontsize = 15)
                else:
                    plt.ylabel(r'$k = $'+str(j_c+1)+'\n'+ 'Training dataset', color = 'k', fontsize = 15)

    cb_ax = fig.add_axes([.92, 0.125, 0.04, 0.755])
    cbar = fig.colorbar(im, cax=cb_ax)
#      set the colorbar ticks and tick labels
    cbar.set_ticks(np.arange(0, 1.1, 0.5))

    if save == True:
        plt.savefig('plots/ex_2_'+num_sam+'.pdf', dpi = 1000, bbox_inches  = 'tight')
    plt.show()
    
    
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

def Algorithm_1(train_set, test_set, method_dict, num_boots, verbose = False):
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
    #boot_size =  int(.8*min(len(train_set), len(test_set)))
    boot_size =  int(min(len(train_set), len(test_set)))
    r_ari, r_ami = np.zeros(num_boots), np.zeros(num_boots)
    true_labels_test, predicted_labels_test = np.ones([num_boots, len(test_set)]), np.zeros([num_boots, len(test_set)]) 
    if verbose == True:
        for b in tqdm_notebook(range(num_boots)):

            sub_train, sub_train_indices = subsample(train_set, len(train_set))
            sub_test, sub_test_indices =  subsample(test_set, len(test_set))
            r_ari[b], r_ami[b], true_labels, predicted_labels = repro_score(sub_train, sub_test, method_dict)
            true_labels_test[b] = pad_labels(true_labels, sub_test_indices, test_set)
            predicted_labels_test[b] = pad_labels(predicted_labels, sub_test_indices, test_set)
    else:
        
        for b in range(num_boots):

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

def Algorithm_1_affinity(train_set, test_set, method_dict, num_boots, verbose = False):
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
    
    if verbose == True:
        for b in tqdm_notebook(range(num_boots)):
            sub_train, sub_train_indices = subsample(train_set, boot_size)
            sub_test, sub_test_indices =  subsample(test_set, boot_size)
            r_ari[b], r_ami[b], true_labels, predicted_labels, k_pred[b], k_true[b] = repro_score_affinity(sub_train, sub_test, method_dict)
            true_labels_test[b] = pad_labels(true_labels, sub_test_indices, test_set)
            predicted_labels_test[b] = pad_labels(predicted_labels, sub_test_indices, test_set)
    else:
        for b in range(num_boots):
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
    #subsample_indices = np.random.choice(len(dset), size=int(size), replace=False)
    subsample_indices = np.random.choice(len(dset), size=int(size), replace=True)
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
        default_dictionary = default_dictionary_clustering_affinity(random_state=None, **kwargs)
        
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
        
    return ari, ami, true_labels, predicted_labels

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

