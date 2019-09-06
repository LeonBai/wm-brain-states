
def absent_states(time_series):    
    """
    This function conuts number of absent clusters for each cluster.
    Requires the k_means function and the module txt file.       
    
    Args:
        time_series: 4D timeseries with dimension: (n_clusters, n_subjects, n_sessions, n_timepoints) 
    
    Return
    
        out: Pandas Data Frame (sum of absent states for each cluster)
   """     
    
    import pandas as pd
    import numpy as np
    
    absent_states_df = pd.DataFrame()

    for i in range(time_series.shape[1]):
        for j in range(time_series.shape[2]):
            for k in range(time_series.shape[0]):
                labels = time_series[k,i,j,:]
                states = len(np.unique(labels))
                absent = k+2-states
                absent_states_df = pd.concat([absent_states_df, 
                                           pd.DataFrame({"Subject":f"sub-{i+1:02}", 
                                                        "Session":f"ses-{i+1}", 
                                                        "k":k+2,
                                                        "Absent":absent}, 
                                                        index=[0])], 
                                          axis=0)

        return absent_states_df
    
    

def dwell_time(vector, label):
    """Calculates the mean length of consecutives timepoints classified as a particular state
    
    Args:
        vector: list of labeling (e.g. from clustering)
        label: label with dwell time to calculate
 
    Return:
        dwell_time : the mean length of consecutives timepoints classified with particuar label
    """
    
    import numpy as np
    from itertools import groupby

    answer = []

    for key, iter in groupby(vector):
        answer.append((key, len(list(iter))))

    dwell_time = np.mean([x[1] for x in answer if x[0]==label])
    
    return(dwell_time)



def remove_repetitions(vector):
    """Reduces a vector by removing consecutive repetitons of given element
    in a vector, e.g. [1,1,1,2,2,2,3,3,3] --> [1,2,3]
    
    Args:
        vector: np.array
            List of numbers to reduce. 
    
    Return:
        reduced_vector: np.array
            Reduced list of numbers.
    """
    
    import numpy as np
    
    reduced_vector = []
    for i in range(len(vector)-1):
        if vector[i] != vector[i+1]: 
            reduced_vector.append(vector[i])
    reduced_vector.append(vector[i+1]) # Add last element
    return(np.asarray(reduced_vector))