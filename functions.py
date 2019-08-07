
# coding: utf-8

# In[1]:


def sliding_window_2d(series,width,step):
    """
    Create a sliding window of time-series with step ad widht of the window given by user.
    
    Parameters
    ----------
    series:  array , 2-D
    width: integer, window width
    step: integer, window step size, in samples. If not provided, window and step size are equal.
    
    Returns
    ----------
    out : array-like , 3-D (number of window,windows,window_size,number of series)
          Vector with all sliding windows
    """
    import numpy as np
    def sliding_window_1d(series,width,step):     
        """
        Parameters
        ----------
        series:  array , 1-D 
        width: integer, window width
        steo: integer, window step size,
    
        Returns
        ----------
        out : array-like , 2-D (nuber of windows,window size)
              Vector with all sliding windows
        """   
        # calculate the number of windows to return, ignoring leftover samples, and
        # allocate memory to contain the samples
        
        import numpy as np
        
        valid = len(series) - width
        number_windows = abs((valid)//step)
    
        out = np.ndarray((number_windows,width), dtype=series.dtype)
    
        for i in range(number_windows):
    
            # "slide" the window along the samples
            start = i*step
            stop = start+width
            out[i] = series[start:stop]
    
        return out
    # calculate the dimension of series, number of windows to return
    dim = series.shape[1]
    valid = len(series)-width
    number_windows = abs((valid)//step)
    cut = np.zeros((number_windows, width, dim))
    
    for i in range(0,dim):                # slide the windows along the samples
        a = sliding_window_1d(series[:,i], width, step)
        cut[:,:, i] = a
    return cut


# In[2]:


def plot_sliding_window(series,width,step):  
    '''
    Plot a sliding-window graph
    
    Parameters
    ----------
    series:  time-series, array , 3-D
    width: window width
    step: window step size, in samples. If not provided, window and step size are equal.
    '''
    
    from nilearn import plotting
    import numpy as np
    from nilearn.connectome import sym_matrix_to_vec
    from nilearn.connectome import ConnectivityMeasure
    
    cut = sliding_window_2d(series, width, step)
    cut_matrix = np.zeros((cut.shape[0], cut.shape[2], cut.shape[2]))
    correlation_measure = ConnectivityMeasure(kind='correlation')
    
    for i in range (cut.shape[0]):
        matrix = correlation_measure.fit_transform([cut[i]])[0]
        cut_matrix[i,:,:] = matrix

    vectors = np.zeros((cut_matrix.shape[0], sym_matrix_to_vec(cut_matrix[1]).shape[0]))

    for i in range(cut_matrix.shape[0]):
        vec  = sym_matrix_to_vec(cut_matrix[i])
        vectors[i,:] = vec

    ax = np.corrcoef(vectors)
    plotting.plot_matrix(ax, title="width={} step={}".format(width,step))


# In[3]:


def number_of_clusters(n,t_vector,n_init=10):
    '''
    Compute the silhouette score for n clusters of time-series vector using K-means clustering,
    create a plot for "elbow" method.
    
    Parameters
    ----------
    n: number of clasters 
    t_vector: vector of time-series
    
    Returns
    ----------
    out: The average silhouette score for each number of clusters,
         plot for sihouette scores
    '''

    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score

    range_n_clusters = [i for i in range(2,n+1)]
    dist = []

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters,n_init=n_init, random_state=10)
        cluster_labels = clusterer.fit_predict(t_vector)
        silhouette_avg = silhouette_score(t_vector, cluster_labels)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        dist.append(silhouette_avg)
    
    plt.plot(range_n_clusters, dist)
    plt.grid()
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.show()


# In[4]:


def k_means(n_clusters,t_vector,n_init=10):
    '''
    Compute k-means labels for n clusters
    
    Parameters
    -----------
    n_clusters: int, number of clusters
    t_vector: array, vector of time_series
    n_init: int, number of time the k-means algorithm will be run with different centroid seeds,
            default: 10.
    
    Returns
    -----------
    out: array-like, 2D (number of clusters,k-means labels)  
    
    '''
    
    from sklearn.cluster import KMeans

    group_k_labels = np.zeros((n_clusters,t_vector.shape[0]))

    for k in range(n_clusters):
        kmeans = KMeans(n_clusters=k+2, n_init=n_init).fit(t_vector)
        k_labels = kmeans.labels_
        group_k_labels[k,:] = k_labels
        
    return(group_k_labels)


# In[5]:


def k_time_series(n_clusters,t_vector):
    '''
    Create array of mean time-series for each n clusters.
    
    Paramaters
    -----------
    n_clusters: number of clusters
    t_vector: vector of time_series
    
    Returns
    -----------
    out: array-like, 2D (number of clusters, mean time-series for each cluster)
    '''
    
    from sklearn.cluster import KMeans
    import pandas as pd
    
    #creating k-labels for time-series
    kmeans = KMeans(n_clusters=n_clusters).fit(t_vector)
    k_labels = kmeans.labels_
    k_labels = pd.DataFrame(k_labels)

    mean_matrix = np.zeros((n_clusters, t_vector.shape[1]))
    
    #create time-series for each cluster
    for i in range(n_clusters):
        dum = pd.get_dummies(k_labels[0])
        t_val = dum[i].values.astype(bool)
        t_vec = t_vector[t_val,:]
        vec_mean = t_vec.mean(axis=0) 
        mean_matrix[i,:] = vec_mean
    return(mean_matrix)


# In[6]:


def compare(t_vector, n_clusters):
    '''
    This function compare corralation between mean of the time-series for each cluster and modules.
    Requires the k_time_series function and the module txt file.    
    
    Parameters
    -----------
    t_vector: array, vector of time_series
    n_clusters: int, number of clusters
    
    
    Return
    -----------
    Minimum and maximum Pearson correlation value fro each cluster,
    plots for eaach cluster and module with Pearson correlation value.
    '''
    import numpy as np
    
    mean_matrix = k_time_series(n_clusters, t_vector)
    
    import pandas as pd
    

    data = pd.read_csv('modules.txt', header=None)
    values = np.zeros((len(np.unique(data)), 264))

    for i in range(len(np.unique(data))):
        dum = pd.get_dummies(data[0])
        name = dum.keys()[i]
        val = (dum[name])
        values[i,:] = val
    
#Pearson correlation between time-series and modules:

    from scipy.stats import pearsonr
    import matplotlib.pyplot as plt

    w = np.zeros((n_clusters,13))       #13 different modules

    for i in range(n_clusters):
        c = -1 
        d = 1
        for j in range(values.shape[0]):
            cor = pearsonr(mean_matrix[i][:], values[j][:])[0]
            w[i,j] = cor
            if cor>c:
                c=cor
                a=j
            if cor<d:
                d=cor
                b=j
        print("For", i,"cluster:")
        print("Max =",c, "with", dum.keys()[a])
        print("Min =",d, "with", dum.keys()[b],"\n")
    
    #plot
    
    for i in range(n_clusters):
        plt.bar(dum.keys(), w[i])
        plt.title(i)
        plt.show()


# In[7]:


def absent(n_sub,n_clusters,t_vector,n_init=10):
    '''
    This function conuts number of absent clusters for each cluster.
    Requires the k_means function and the module txt file.       
    
    Parameters
    -----------
    n_sub: int, number of subjects
    n_clusters: int, number of clusters
    t_vector: array, vector of time_series
    
    Return
    -----------
    out: Pandas Data Frame (sum of absent states for each cluster),
         bar plot of absent states. 
    
    '''
    import pandas as pd
    import numpy as np
    
    val = (t_vector.shape[0])/10
    
    k_labels = np.zeros((n_sub,n_clusters,int(val)))
    group_labels = k_means(n_clusters,t_vector,n_init=n_init)
    
    
    for k in range(n_clusters):
        k_labels[:,k,:] = np.split(group_labels[k], n_sub)
    
    group_absent = pd.DataFrame()
    
    for k in range(2,n_clusters+2):
        for sub in range(n_sub):
            states = len(np.unique(k_labels[sub,k-2,:]))
            absent = k-states
            group_absent = pd.concat([group_absent, pd.DataFrame({"sub":sub, "n_clusters":k, "absent":absent}, index=[0])], axis=0)
    
    
    ab_sum = group_absent['absent'].groupby(group_absent['n_clusters']).sum()

    frame = pd.DataFrame(np.array(ab_sum), index=ab_sum.index, columns=['absent']) 
    return(frame)          


# In[8]:
def transition_of_states(n_clusters,series):
    '''
    Matrix of probability of transition beetween states
    
    Parameters
    ------------
    n_clusters: number of clusters, int
    series: time-series, 2D array
    
    Return
    ------------
    out: 2D matrix (n_clusters,n_clusters)
    
    '''
    from sklearn.cluster import KMeans
    import numpy as np
    
    kmeans = KMeans(n_clusters=n_clusters).fit(series)            #obliczanie wartosci k-means
    k_labels = kmeans.labels_
    
    matx = np.zeros((n_clusters,n_clusters))
    
    for i in range(n_clusters):
        for j in range(n_clusters):
            count = 0
            for l in range(len(k_labels)-1):
                if k_labels[l]==i and k_labels[l+1]==j:
                    #print(k_labels[l], k_labels[l+1])
                    count += 1
                    p = count/(len(k_labels-1))                 #wartosc p dla kazdej pary 
        
            matx[i,j] = p                             #zapisujemy do macierzy
    return(matx)

#In[9]:
def multiple_transition(n_clusters, series):
    '''
    Matrix of probability of transition beetween states for more than one subjects
    
    Parameters
    ------------
    n_clusters: number of clusters, int
    series: time-series, 3D array (n_subjects, )
    
    Return
    ------------
    out: 3D matrix (n_subjects, n_clusters,n_clusters)
    
    '''
    import numpy as np
    
    multiple_matrix = np.zeros((series.shape[0],n_clusters,n_clusters))

    for sub in range(series.shape[0]):
        multiple_matrix[sub,:] = transition_of_states(n_clusters,series[sub])
        
    return(multiple_matrix)

#In[10]:
def mean_multiple_transition(n_clusters,series):
    '''
    Matrix of probability of transition beetween states for more than one subjects
    
    Parameters
    ------------
    n_clusters: number of clusters, int
    series: time-series, 3D array (n_subjects, )
    
    Return
    ------------
    out: 2D matrix (n_clusters,n_clusters)
    
    '''
    return multiple_transition(n_clusters,series).mean(axis=0)



#----------------------------------------------

def absent_states(time_series):    
    '''
    This function conuts number of absent clusters for each cluster.
    Requires the k_means function and the module txt file.       
    
    Parameters
    -----------
    time_series: 4D timeseries with dimension: (n_clusters, n_subjects, n_sessions, n_timepoints) 
    
    Return
    -----------
    out: Pandas Data Frame (sum of absent states for each cluster)
    
    '''     
    
    import pandas as pd
    import numpy as np
    
    absent_states_df = pd.DataFrame()

    for i in range(time_series.shape[2]):
        for j in range(time_series.shape[3]):
            for k in range(time_series.shape[0]):
                labels = time_series[k,:,i,j]
                states = len(np.unique(labels))
                absent = k+2-states
                absent_states_df = pd.concat([absent_states_df, 
                                           pd.DataFrame({"subject":f"sub-{i+1:02}", 
                                                        "session":f"ses-{i+1}", 
                                                        "k":k+2,
                                                        "absent":absent}, 
                                                        index=[0])], 
                                          axis=0)

        return absent_states_df
    