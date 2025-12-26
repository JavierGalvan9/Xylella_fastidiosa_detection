
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

parentDir = os.path.abspath(os.getcwd())
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management


def groupby_split(df, labels='', inseparable_set='', test_size=0.05):
    # Given df and feature_2, groupby feature_2 and split each group into train and test sets
    # feature_1 is the feature to stratify the split
    # test_size is the size of the test set
    # Return a list of tuples (train, test) where train and test are dataframes
    
    # Create new dataframe with the first row for each unique value of feature_2
    df_new = df.groupby(inseparable_set).first().reset_index()
    # Stratify the split on feature_1
    train, test = train_test_split(df_new, test_size=test_size, stratify=df_new[labels], random_state=42)
    # From the original dataframe, select the rows that have the same values of feature_2 as the train and test dataframes
    train_df = df[df[inseparable_set].isin(train[inseparable_set])]
    test_df = df[df[inseparable_set].isin(test[inseparable_set])]
    # Return a list of tuples (train, test) where train and test are dataframes
    return train_df, test_df

# Define a function that, given a counter, returns the proportion of each class
def get_class_proportions(df, feature):
    counter = Counter(df[feature])
    total = sum(counter.values())
    new_dict = {cl: count / total for cl, count in counter.items()}
    # Sort the dictionary by key
    new_dict = {k: new_dict[k] for k in sorted(new_dict)}
    return new_dict

# Create a function that given a path creates the directory if it doesn't exist
# and returns the preprocessed data (X_train, X_test, y_train, y_test)
def data_preprocessing(path, n_samples=None, use_spectral_bands=True, use_indices=True):
    # Create a path to save/load the train and test sets which contains the features used
    if use_spectral_bands and not use_indices:
        train_test_path = os.path.join(path, 'Spectral bands')
    elif not use_spectral_bands and use_indices:
        train_test_path = os.path.join(path, 'Indices')
    elif use_spectral_bands and use_indices:
        train_test_path = os.path.join(path, 'Spectral bands and indices')
    else:
        # Print an error message and exit the program if the user doesn't choose any features
        print("Error: No features were selected!")
        sys.exit()
    # Check if the train and test sets directory exists and create it if it doesn't
    # This directory will be used to save the train and test sets
    # This is useful to avoid having to split the data again if the script is run again
    if not os.path.exists(train_test_path):
        os.makedirs(train_test_path)
        # Load data and remove unclassified pixels
        df = file_management.load_pickle('Processed Data/QPCR_labelled_df_qpcr.pkl')
        df.dropna(inplace=True)
        # Sample a subset of the data
        # n_samples = 10000
        if n_samples is not None:
            # loc n_samples first rows of df
            df = df.iloc[:n_samples]
            # df = df.sample(n=n_samples)

        # df.loc[df['PCR'] ==np.nan, 'PCR'] = 0

        ### Pad the cluster_id column with zeros to have the same length as the other columns
        ### pad with zero rows to make all cluster_id have the same number of rows
        # group the rows by their cluster_id
        # groups = df.groupby('cluster_id')
        # # get the maximum number of rows for any cluster_id
        # max_rows = groups.size().max()
        # # create a new DataFrame to hold the expanded groups
        # expanded_groups = []
        # # loop over each group and append new rows with zeros to it
        # for name, group in groups:
        #     n_rows = len(group)
        #     if n_rows < max_rows:
        #         n_missing_rows = max_rows - n_rows
        #         missing_rows = pd.DataFrame(np.zeros((n_missing_rows, len(df.columns))), columns=df.columns)
        #         missing_rows['cluster_id'] = name
        #         # get the pcr value of the first row of the group
        #         pcr = group['PCR'].iloc[0]
        #         # set the pcr value of the missing rows to the pcr value of the first row of the group
        #         missing_rows['PCR'] = pcr                
        #         group = pd.concat([group, missing_rows], axis=0)
        #     expanded_groups.append(group)
        # # concatenate all the groups back together
        # df_expanded = pd.concat(expanded_groups, axis=0)
        # # sort the rows by cluster_id
        # df = df_expanded.sort_values('cluster_id')

        # get the maximum number of rows with the same cluster_id
        max_cluster_id = df.groupby('cluster_id').size().max()
        def expand_cluster(group, target_size=50):
            cluster_size = len(group)
            if cluster_size == 1:
                # if there's only one row in the group, use a default scale of 0.1
                std = np.array([0.1] * len(group.columns))
            else:
                # calculate the std of each column within the group
                std = group.std().values
            # determine the number of samples to add
            num_samples = target_size - cluster_size
            if num_samples > 0:
                # select random rows from the group with replacement
                sample_indices = np.random.choice(cluster_size, size=num_samples, replace=True)
                samples = group.iloc[sample_indices]
                # add noise to the samples
                noise = np.random.normal(scale=std, size=(num_samples, len(group.columns)))
                noisy_samples = samples.values + noise
                # create a new data frame with the noisy samples
                noisy_df = pd.DataFrame(noisy_samples, columns=group.columns)
                # set the pcr, longitude and latitute value of the noisy samples to the pcr value of the first row of the group
                noisy_df['PCR'] = group['PCR'].iloc[0]
                noisy_df['cluster_id'] = group['cluster_id'].iloc[0]
                # concatenate the noisy data frame with the original group
                group = pd.concat([group, noisy_df], ignore_index=True)
            if num_samples < 0:
                # if the number of samples to add is negative, we remove the features that correlate the least with the PCR
                # get the correlation between the features and the PCR
                corr = group.corr()['PCR']

            return group

        # assuming your DataFrame is called df
        df = df.groupby('cluster_id').apply(expand_cluster, target_size=50).reset_index(drop=True)
        file_management.save_pickle(df, 'QPCR_labelled_df_qpcr_expanded', 'Processed Data')
        # Split the data into train and test sets (stratified to keep the same proportion of labels in each set but 
        # keeping each unique tree cluster in only one set)
        train_df, test_df = groupby_split(df, labels='PCR', inseparable_set='cluster_id', test_size=0.1)
        # Average the values of the features for each cluster
        # train_df = train_df.groupby('cluster_id').mean().reset_index()
        # test_df = test_df.groupby('cluster_id').mean().reset_index()
        
        # Get the cluster ids of the train and test sets
        cluster_id_train = train_df['cluster_id']
        cluster_id_test = test_df['cluster_id']
        # Get the latitudes and longitudes of the training and testing sets
        lat_train = train_df['Lats']
        lon_train = train_df['Longs']
        lat_test = test_df['Lats']
        lon_test = test_df['Longs']

        # Plot the trees used for training and testing
        if False:
            original_df = file_management.load_pickle('Processed Data/dataset.lzma')
            metadata_df = file_management.load_pickle('Processed Data/metadata_df.lzma')
            pan_shape = metadata_df['original_shape']
            # Loc three columns of original dataset
            original_df = original_df.loc[:, ['N', 'R', 'G', 'Longs', 'Lats']]	
            rgb_df = (255*original_df/original_df.max()).astype(np.uint8)
            rgb = np.dstack((rgb_df['N'].values.reshape(pan_shape),rgb_df['R'].values.reshape(pan_shape),rgb_df['G'].values.reshape(pan_shape)))
            fig = plt.figure()
            plt.imshow(rgb,
                    extent=[original_df['Longs'].min(), original_df['Longs'].max(), original_df['Lats'].min(), original_df['Lats'].max()])
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.ticklabel_format(useOffset=False)
            # Draw a circle around the trees used for training and testing 
            plt.scatter(lon_train, lat_train, s=(72/300)**2, alpha=1, c='yellow', marker='s')
            plt.scatter(lon_test, lat_test, s=(72/300)**2, alpha=1, c='indigo', marker='s')
            plt.legend(['Train', 'Test'])
            # Save the plot
            path = 'Images/Train_test_layout'
            os.makedirs(path, exist_ok=True)
            plt.savefig(os.path.join(path, 'layout_rgb.png'), dpi=300, transparent=True)
            plt.close()

        # Split the data into train and test dataframes (stratified to keep the same proportion of classes in each set)
        # train_df, test_df = train_test_split(df, test_size=0.05, stratify=df['PCR'], random_state=88)
        # Choose the features to use in the classification task
        if use_spectral_bands and not use_indices:
            spectral_bands = ['C', 'B', 'G', 'Y', 'R', 'RE', 'N', 'N2']
            X_train = train_df.loc[:, spectral_bands] # only spectral bands
            X_test = test_df.loc[:, spectral_bands] # only spectral bands
        elif not use_spectral_bands and use_indices:
            X_train = train_df.iloc[:, 8:-4] # indices
            X_test = test_df.iloc[:, 8:-4] # indices            
        elif use_spectral_bands and use_indices:
            spectral_bands = ['C', 'B', 'G', 'Y', 'R', 'RE', 'N', 'N2']
            X_train = train_df.loc[:, spectral_bands + list(train_df.columns[8:-4])]
            X_test = test_df.loc[:, spectral_bands + list(test_df.columns[8:-4])]
        else:
            # Print an error message and exit the program if the user doesn't choose any features
            print("Error: No features were selected!")
            sys.exit()

        # Print the column names used in X_train and X_test
        print("Features used in the classification task: ")
        print(X_train.columns)

        # Save the labels in a separate variable 
        y_train = train_df['PCR'].values
        y_test = test_df['PCR'].values
        
        # Print the number of samples in each set
        print('Training set size: ', len(train_df))
        print('Test set size: ', len(test_df))

        # Ensure that the proportion of samples in each set is the same as the original dataset
        # Print the proportion samples in each set by dividing the counter by the total number of samples
        feature = 'PCR'
        print('Original dataset proportions: ', get_class_proportions(df, feature))
        print('Training set proportions: ', get_class_proportions(train_df, feature) )
        print('Test set proportions: ', get_class_proportions(test_df, feature) )

        # Print the number of unique cluster_id in each set
        print('Number of unique cluster_id in the original dataset: ', len(df['cluster_id'].unique()))
        print('Number of unique cluster_id in the training set: ', len(train_df['cluster_id'].unique()))
        print('Number of unique cluster_id in the test set: ', len(test_df['cluster_id'].unique()))

        # Data preprocessing - Standard normalization of the features 
        # This is done to avoid the features with higher values to dominate the training process
        # The mean and standard deviation of the train set are used to normalize the test set 
        std_scale = StandardScaler()
        # Fit the scaler to the train set and transform it
        X_train = std_scale.fit_transform(X_train)
        # Transform the test set using the same mean and standard deviation
        X_test = std_scale.transform(X_test)

        # Number of features
        print('X_train shape:', X_train.shape)
        n_features = X_train.shape[1]
        print('Number of features:', n_features)
        # get the number of unique cluster_ids in the training set
        # count the number of times the first value of cluster_id_train is repeated
        cluster_id_train_array = np.array(cluster_id_train)
        cluster_id_test_array = np.array(cluster_id_test)
        max_cluster_size = np.sum(np.array(cluster_id_train_array) == cluster_id_train_array[0])
        n_train = len(set(cluster_id_train_array))
        n_test = len(set(cluster_id_test_array))
        print('Number of training samples:', n_train)
        print('Number of test samples:', n_test)
        print('Max cluster id:', max_cluster_size)
        # reshape X_train to be [n_train, max_cluster_size*n_features]
        X_train = X_train.reshape((n_train, 50*n_features))
        # reshape X_test to be [n_test, max_cluster_size*n_features]
        X_test = X_test.reshape((n_test, 50*n_features))
        # Redefine the number of features
        n_features = 50*n_features

        print('X_train shape before PCA:', X_train.shape)

        # Reduce the number of features of the training and testing sets by using PCA
        from sklearn.decomposition import PCA

        # use PCA to unsure 0.9999 of the variance is explained
        # pca = PCA(n_components=0.999)
        pca = PCA(n_components=20)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print('X_train shape after PCA:', X_train.shape)

        # for the labels, we need to average the labels of the samples in each cluster obtaining a (n_train,) array
        y_train = np.mean(y_train.reshape((n_train, max_cluster_size)), axis=1).astype(int)
        y_test = np.mean(y_test.reshape((n_test, max_cluster_size)), axis=1).astype(int)

        # Save the train and test sets
        file_management.save_pickle(X_train, 'X_train', train_test_path)
        file_management.save_pickle(y_train, 'Y_train', train_test_path)
        file_management.save_pickle(X_test, 'X_test', train_test_path)
        file_management.save_pickle(y_test, 'Y_test', train_test_path)
        # Save the scaler to use it later to normalize new data
        file_management.save_pickle(std_scale, 'scaler', train_test_path)
        # Save the cluster ids of the train and test sets
        file_management.save_pickle(cluster_id_train, 'cluster_id_train', train_test_path)
        file_management.save_pickle(cluster_id_test, 'cluster_id_test', train_test_path)

        return X_train, X_test, y_train, y_test, std_scale, cluster_id_train, cluster_id_test
    else:
        # Load the train and test sets
        X_train = file_management.load_pickle(os.path.join(train_test_path, 'X_train.pkl'))
        y_train = file_management.load_pickle(os.path.join(train_test_path, 'Y_train.pkl'))
        X_test = file_management.load_pickle(os.path.join(train_test_path, 'X_test.pkl'))
        y_test = file_management.load_pickle(os.path.join(train_test_path, 'Y_test.pkl'))
        # Load the scaler
        std_scale = file_management.load_pickle(os.path.join(train_test_path, 'scaler.pkl'))
        # Load the cluster ids of the train and test sets
        cluster_id_test = file_management.load_pickle(os.path.join(train_test_path, 'cluster_id_test.pkl'))
        cluster_id_train = file_management.load_pickle(os.path.join(train_test_path, 'cluster_id_train.pkl'))

        return X_train, X_test, y_train, y_test, std_scale, cluster_id_train, cluster_id_test