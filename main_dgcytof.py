
# ### Installation
# conda create -n dgcytof python=3.8 -y
# conda activate dgcytof

# conda install numpy=1.19.1 numba=0.52.0 pandas=1.1.0 scipy=1.5.2 matplotlib=3.3.0 seaborn=0.10.1 scikit-learn=0.23.2 scikit-bio=0.5.6 hdbscan=0.8.26 joblib=0.17.0 scikit-plot==0.3.7 umap-learn==0.4.6 tensorboard==2.3.0 -y
# conda install pytorch=1.6.0 torchvision=0.7.0 cpuonly -c pytorch -y

# ### Clone repo:
# git clone https://github.com/lijcheng12/DGCyTOF.git
# cd DGCyTOF/DGCyTOF_Package/
# python setup.py sdist
# pip install dist/DGCyTOF-1.0.0.tar.gz
# pip show DGCyTOF

# on weneg after installation of dgcytof install more recent pytorch version:
# pip uninstall torch torchvision torchaudio -y
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

def mwe():
    import os
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    import DGCyTOF

    # ### Dummy example
    dataset = np.load(os.path.join(os.getcwd(), 'results/param_influence_study/data_handling/data_train.npy'))
    dataset[:, -1] -= 1  # Change labels s.t. they start from 0, used for indexing by Dgcytof
    # print(np.unique(dataset[:, -1]))
    dataset = pd.DataFrame(
        data=dataset,
        columns=list(range(dataset.shape[1] - 1)) + ['label']
    )
    print(dataset)
    mwe = True
    if mwe:
        # ### Only for testing
        # Use subset for faster inference time
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)[0:10000]

        # Insert nan values to simulate input format of Dgcytof
        percentage = 0.20
        num_nans = int(len(dataset) * percentage)
        nan_indices = np.random.choice(dataset.index, size=num_nans, replace=False)
        dataset.loc[nan_indices, 'label'] = np.nan


    # ### Change dataset into correct input format, instantiate dataloaders
    X_data_labeled, y_data, data_unlabeled = DGCyTOF.preprocessing(dataset, columns_to_remove=[])
    # print(f'# ### X_data_labeled:\n{X_data_labeled}')  # Dataframe with data from labeled cells, label column removed
    # print(f'# ### y_data:\n{y_data}')  # Pandas series with labels
    # print(f'# ### data_unlabeled:\n{data_unlabeled}')  # Dataframe with intensity data from unlabeled cells

    x_train, x_val, y_train, y_val = train_test_split(
        X_data_labeled, y_data, stratify=y_data, test_size=0.20, random_state=5)

    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)

    x_val = torch.tensor(x_val.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.long)
    dataset_val = torch.utils.data.TensorDataset(x_val, y_val)

    x_test = torch.tensor(data_unlabeled.values, dtype=torch.float32)


    # ### Define and instantiate the Softmax-classifier
    class SoftmaxClassifier(nn.Module):
        def __init__(self, in_size, out_size):
            super(SoftmaxClassifier, self).__init__()
            # Define layers
            self.fc1 = nn.Linear(in_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, out_size)
            self.softmax = nn.Softmax(dim=1)  # Softmax for the output layer

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            # x = self.softmax(x)  # Softmax activation for output
            # Do not apply softmax, CrossEntropyLoss does so internally
            return x


    softmax_classifier = SoftmaxClassifier(in_size=x_train.shape[1], out_size=np.unique(y_train).shape[0])
    # ### Train the model
    DGCyTOF.train_model(
        model_fc=softmax_classifier,
        X_train=dataset_train,
        max_epochs=20, params_train={'batch_size': 128, 'shuffle': True, 'num_workers': 6}
    )

    # ### Validation
    validation_results = DGCyTOF.validate_model(
        model_fc=softmax_classifier,
        val_tensor=dataset_val,
        classes=np.unique(y_train).tolist(),
        params_val={'batch_size': 10000, 'shuffle': False, 'num_workers': 6}
    )
    # print(validation_results)  # List of triples: (predicted label, actual label, model output)

    # ### Calibrate data ???
    # X_test is the data tensor of the unlabeled data, unlabeled_data is the corresponding dataframe
    # updated_incorrect_data = DGCyTOF.calibrate_data(
    #     model_fc=softmax_classifier,
    #     X_test=x_test,
    #     classes=np.unique(y_train).tolist(),
    #     validation_results=validation_results,
    #     unlabeled_data=data_unlabeled,
    # )
    # print(updated_incorrect_data)

    import torch.nn.functional as F
    from torch.autograd import Variable
    from scipy.stats import spearmanr


    def calibrate_data(model_fc, X_test, classes, validation_results, unlabeled_data):
        '''

            Calibrates the test set of the data based on the training model and validation results in performing over the test set.
            Test data either are classified more accurately or labeled as a new subtype based on the minimum threshold
            in classification. Minumum threshold is computed by obtaining the lowest correlation probability from validation results.
            Returns calibrated incorrect data.

            **Params**:

            * model_fc: Trained PyTorch model, must have a forward function and utilize argmax as classification in its design
            * X_test: CyTOF data for test set.
            * classes: List of types of cells
            * validation_results: zip created by validate_model. Contains the following keys:
                * pred: Predicted label of a data point
                * label: Actual label of a data point
                * out: Output value of the data running forward through model_fc

            **Returns**:

            * updated_incorrect_data: Calibrated incorrect data.

        '''

        # Create list of instances from the val set where the prediction was correct,
        # save the predicted probability of the predicted class (i.e. the confidence of the class prediction)
        correct_pred_info = [(label, pred, np.max(F.softmax(out, dim=0).data.numpy())) for pred, label, out in
                             validation_results if (pred == label)]
        # Get a list of the prediction probs of the correct predictions
        corr_prob = [prob for label, pred, prob in correct_pred_info]

        # For the unlabeled data get the model output (forward pass), and the predicted class label (argmax of output)
        test_outputs = model_fc(Variable(X_test))
        _, test_predicted = torch.max(test_outputs.data, 1)  # Find the class index with the maximum value.

        # Get the predicted class probs for the unlabeled data
        output_logits = F.softmax(test_outputs, dim=1).data.numpy()
        output_labels = test_predicted.data.numpy()  # Why ???
        # Get the predicted probability of the predicted class (i.e. the confidence of the class prediction), and round
        probabilities = np.max(output_logits, axis=1)
        tem = [round(i, 4) for i in probabilities]
        # incorrect_index = [tem.index(a) for a in tem if a <= min(corr_prob)]
        # Accept the prediction for the new unlabelled data (correct_index),
        # if its confidence is larger than the min confidence on the validation set
        # otherwise reject the prediction (incorrect_index)
        correct_index = [i for i, v in enumerate(tem) if v > min(corr_prob)]
        incorrect_index = [i for i, v in enumerate(tem) if v <= min(corr_prob)]
        # Get the data vectors for the unlabeled events for which the prediction was rejected (due to too low confidence)
        incorrect_data = pd.DataFrame([X_test[i].data.numpy() for i in incorrect_index])
        print("A total of " + str(len(incorrect_data)) + "  incorrect labeling as been found")

        # ###### My addition: ###### #
        y_pred = np.full(len(X_test), -1)  # -1 indicates unassigned
        # Assign predicted labels where the prediction is deemed to be correct (high confidence)
        for i in correct_index:
            y_pred[i] = test_predicted[i].item()

        ##################

        # Get copy of df with unlabeled data
        test_with_labels = unlabeled_data.copy()
        # Get the predicted labels for the unlabeled data, add as label column in df
        test_with_labels['label'] = test_predicted.numpy()
        test_with_labels = test_with_labels.reset_index(drop=True)
        # Subset the df to the instances where the prediction was deemed correct (high confidence)
        test_correct_with_labels = test_with_labels.iloc[correct_index]

        # Initialize dict
        test_correct_list = dict()
        # Iterate over the classes in the training set
        for subtype_no in range(0, len(classes)):
            # Dict entry with 'subtype_no' as key is the intensity vectors of the unlabeled events
            # that were predicted to be of class 'subtype_no'
            test_correct_list[subtype_no] = test_correct_with_labels[test_correct_with_labels.label == subtype_no].drop(
                ['label'],
                axis=1).values.tolist()
        # => {label: data matrix from unlabeled events predicted to be label}

        # Each row (intensity vector corresponding to an event vor which the prediction was rejected) becomes a list
        # => list of lists
        incorrect_list = incorrect_data.values.tolist()

        # Initialize dict
        rho_dict = dict()

        # Iterate over the classes in the training set
        for i in range(len(classes)):
            # Correlation of intensity vectors of unlabeled events that were deemed to be correctly classified as class i
            # and the intensity vectors of unlabeled events that were deemed to be incorrectly classified

            # print(np.array(test_correct_list[i] + incorrect_list))
            # print(np.array(test_correct_list[i] + incorrect_list).shape)
            # print(np.transpose(test_correct_list[i] + incorrect_list))
            # print(np.transpose(test_correct_list[i] + incorrect_list).shape)
            # np.transpose(test_correct_list[i] + incorrect_list) has dim: (channels, n_events)
            # spearmanr: column represents a variable, with observations in the rows
            # => Compute correlation between events (= variable)

            if i == 1:
                rho_dict[i], _ = spearmanr(np.transpose(test_correct_list[i][-12000:] + incorrect_list))
            else:
                rho_dict[i], _ = spearmanr(np.transpose(test_correct_list[i] + incorrect_list))
                # print(np.transpose(test_correct_list[i] + incorrect_list).shape)
                # print(spearmanr(np.transpose(test_correct_list[i] + incorrect_list))[0])
                # print(spearmanr(np.transpose(test_correct_list[i] + incorrect_list))[0].shape)

        # => {label: correlation matrix of size n_events x n_events}

        #################

        # Intitialize dicts
        wrongly_incorrect_index = dict()
        wrongly_incorrect_corr = dict()

        # Iterate over the classes in the training set
        for subtype_no in range(0, len(classes)):
            # Active learning,
            # Input:
            # - data matrix from unlabeled events predicted to be label 'subtype_no'
            # - correlation matrix of unlabeled events that were deemed to be correctly classified as class 'subtype_no'
            #   and unlabeled events that were deemed to be incorrectly classified
            # Output:
            # - List of indices of wrongly classified events that probably belong to class 'subtype_no' (high correlation)
            # - The corresponding average correlation value
            if subtype_no == 1:
                wrongly_incorrect_index[subtype_no], wrongly_incorrect_corr[subtype_no], _ = active_learning_index(
                    test_correct_list[subtype_no][-12000:], rho_dict[subtype_no])
            else:
                wrongly_incorrect_index[subtype_no], wrongly_incorrect_corr[subtype_no], _ = active_learning_index(
                    test_correct_list[subtype_no], rho_dict[subtype_no])

        ###################

        # Initialize df of dim (n_incorrectly classified events, number of classes)
        temp = pd.DataFrame(np.zeros((len(incorrect_list), len(classes))))
        # Iterate over the classes in the training set
        for subtype_no in range(0, len(classes)):
            # For each class enter the previously computed average correlation value into the df
            temp[subtype_no][wrongly_incorrect_index[subtype_no]] = wrongly_incorrect_corr[subtype_no]
        # => Df with average correlations values for each wrongly classified event,
        #    they indicate the most probable class they actually belong to

        # Drop rows with all zeros
        temp_1 = temp.loc[(temp != 0).any(1)]

        # Set highest value in each row to 1 and other entries to 0, store in df
        # => Df with the corrected cell type predictions
        m = np.zeros_like(temp_1.values)
        m[np.arange(len(temp_1)), temp_1.values.argmax(1)] = 1
        temp_2 = pd.DataFrame(m, columns=temp_1.columns, index=temp_1.index).astype(int)

        # Initialize dict
        updated_test = dict()
        # Iterate over the classes in the training set
        # indices (of cells) that should be added to respective celltype
        for subtype_no in range(0, len(classes)):
            # In column of temp_2 corresponding to 'subtype_no' (-> holds info which cells are reassigned to 'subtype_no'),
            # Retrieve the indices (= events) where this columns has a 1 entry => np array with those indices
            temp_index = np.array(temp_2[subtype_no].index[temp_2[subtype_no].to_numpy().nonzero()])
            # Add dict entry:
            # - test_correct_list = {label: data matrix from unlabeled events predicted to be label, as a list of lists}
            # - incorrect_data.iloc[temp_index].values.tolist()
            #   = datamatrix from unlabeled events now also predicted to be label, as a list of lists
            updated_test[subtype_no] = test_correct_list[subtype_no] + incorrect_data.iloc[temp_index].values.tolist()

            # ###### My addition ###### #
            # Iterate over the indices of events that are now assigned to 'subtype_no'
            # - 'temp_index' contains indices of rows in 'incorrect_data' that were reassigned to 'subtype_no',
            #   'incorrect_data' = df with the data vectors for the unlabeled events for which the prediction was rejected
            # - 'incorrect_index' = indices of the rows in X_test for which the prediction was deemed wrong,
            #   correspond to the rows in 'incorrect_data'
            # => Each row in 'incorrect_data' corresponds to a row in 'X_test',
            #    which can be identified via 'incorrect_index'
            for idx in temp_index:
                y_pred[incorrect_index[idx]] = subtype_no

        # => updated_test = {label: data matrix from unlabeled events now finally predicted to be label, as a list of lists}

        # incorrect_data = data vectors for the unlabeled events for which the prediction was rejected, as df
        # Drop the events from the df that were reassigned with a new label
        # The remaining samples in updated_incorrect_data are those that could not be confidently assigned to any class
        # and might require further analysis.
        updated_incorrect_data = incorrect_data.drop(np.array(temp_2.index), axis=0)

        return updated_incorrect_data, y_pred


    def active_learning_index(test_correct_list, rho, indices=True):
        wrongly_incorrect_index = []  # indices of wrongly predicted incorrect
        # Number of events that were deemed to be correctly classified as some label
        correct_size = len(test_correct_list)  # length of correct class
        # Subset the correlation matrix to those events
        rho_correct = rho[:correct_size, :correct_size]  # correlation matrix of correct class
        # Get the entries of the upper triangle matrix
        rho_correct_array = rho_correct[np.triu_indices(correct_size, k=1)]  # array of correlations of correct class
        # Compute the mean for those entries
        rho_avg = np.mean(rho_correct_array)
        # => 'rho_avg' = mean of the correlation values among events that were deemed as correctly classified

        if indices == True:
            # For samples i (beyond 'correct_size', range(correct_size + 1, len(rho) ) that were deemed to be
            # incorrectly classified do:
            # - Compute average correlation of sample i with all correctly classified samples
            #   (np.mean(rho[i, :correct_size]))
            # - If the mean is larger than the internal mean of correctly classified samples (mean > rho_avg),
            #   store the event in 'wrongly_incorrect_index',
            #   i.e. it likely belongs to the class of the correctly classified events
            # - Also store the average correlation value of the event with the correctly classified samples
            #   Needed later on as a tiebreaker
            wrongly_incorrect_index = [(i - correct_size) for i in range(correct_size + 1, len(rho))
                                       if np.mean(rho[i, :correct_size]) > rho_avg]
            wrongly_incorrect_corr = [np.mean(rho[i, :correct_size]) for i in range(correct_size + 1, len(rho))
                                      if np.mean(rho[i, :correct_size]) > rho_avg]
            return wrongly_incorrect_index, wrongly_incorrect_corr, rho_avg
        else:
            return rho_avg


    a, b = calibrate_data(
        model_fc=softmax_classifier,
        X_test=x_test,
        classes=np.unique(y_train).tolist(),
        validation_results=validation_results,
        unlabeled_data=data_unlabeled,
    )

    print(a)
    print(b)


def dgcytof_pipeline():
    # Pipline constructed along the lines of
    # https://github.com/lijcheng12/DGCyTOF/tree/main/Code_Study/DGCyTOF/CyTOF2
    # https://github.com/lijcheng12/DGCyTOF/blob/main/Code_Study/DGCyTOF/CyTOF2/CyTOF2.ipynb
    import os
    import time
    import numpy as np
    import pandas as pd
    from typing import Tuple
    from validation.dgcytof import DgcytofClassifier
    from validation.val_utils import evaluate

    # ### Dummy example
    data_train = np.load(os.path.join(os.getcwd(), 'results/baseline_less_channels/data_handling/data_train.npy'))
    data_test = np.load(os.path.join(os.getcwd(), 'results/baseline_less_channels/data_handling/data_test.npy'))

    # Flag to truncate dataset for debug purposes
    mwe = False
    if mwe:
        # ### Only for testing
        # Use subset for faster inference time
        data_train_df = pd.DataFrame(
            data=data_train,
        )
        data_train = data_train_df.sample(frac=1, random_state=42).reset_index(drop=True)[0:10000].to_numpy()
        data_test_df = pd.DataFrame(
            data=data_test,
        )
        data_test = data_test_df.sample(frac=1, random_state=42).reset_index(drop=True)[0:10000].to_numpy()

    x_train = data_train[:, :-1].copy()
    y_train = data_train[:, -1].copy()
    x_test = data_test[:, :-1].copy()
    y_test = data_test[:, -1].copy()

    clf = DgcytofClassifier(
        val_size=0.2,
        layer_sizes=(128, 64, 32),
        n_epochs=20,
        train_params={'batch_size': 128, 'shuffle': True, 'num_workers': 6},
        verbosity=0
    )

    print('# ### Starting fit ...')
    st_fit = time.time()
    clf.fit(X=x_train, y=y_train)
    et_fit = time.time()
    fit_time = et_fit - st_fit
    print(f'# ### Fit finished, time: {fit_time}')

    print('# ### Starting Dgcytof prediction ...')
    st_pred = time.time()
    y_pred_dgcytof = clf.predict(X=x_test)
    et_pred = time.time()
    pred_time = et_pred - st_pred
    print(f'# ### Prediction Dgcytof finished, time: {pred_time}')

    # ### Also produce prediction just based on the softmax classifier
    print('# ### Starting Softmax prediction ...')
    st_pred_softmax = time.time()
    y_pred_softmax = clf.predict_proba(X=x_test)
    y_pred_softmax = y_pred_softmax.argmax(axis=1)
    y_pred_softmax = np.array([clf.new_to_og_classes_dict_[key] for key in y_pred_softmax])
    et_pred_softmax = time.time()
    pred_time_softmax = et_pred_softmax - st_pred_softmax
    print(f'# ### Prediction Softmax finished, time: {pred_time_softmax}')

    times_df = pd.DataFrame(
        data=np.array([fit_time, pred_time, pred_time_softmax]).reshape((1, 3)),
        index=['sek'],
        columns=['fit', 'pred_dgcytof', 'pred_softmax']
    )

    print(f'# ### Times:\n{times_df}')

    print('# ### Results Dgcytof:')
    res_df_dgcytof, res_df_class_wise_dgcytof, cf_df_dgcytof,  y_test_unknowns_val_counts = evaluate(
        y_true=y_test, y_pred=y_pred_dgcytof, wout_unknowns=True
    )
    print('\n')
    print('# ### Results (just) Softmax:')
    res_df_softmax, res_df_class_wise_softmax, cf_df_softmax = evaluate(
        y_true=y_test, y_pred=y_pred_softmax, wout_unknowns=False
    )

    save = True
    if save:
        base_p = os.path.join(os.getcwd(), f'results/dgcytof/all_data')
        if not os.path.exists(base_p):
            os.makedirs(os.path.join(base_p))

        clf.save(filepath=base_p)
        times_df.to_csv(os.path.join(base_p, 'times.csv'))
        y_test_unknowns_val_counts.to_csv(os.path.join(base_p, 'unclassifiable_label_count.csv'))
        res_df_dgcytof.to_csv(os.path.join(base_p, 'res_df_dgcytof.csv'))
        res_df_class_wise_dgcytof.to_csv(os.path.join(base_p, 'res_df_class_wise_dgcytof.csv'))
        cf_df_dgcytof.to_csv(os.path.join(base_p, 'cf_df_dgcytof.csv'))
        res_df_softmax.to_csv(os.path.join(base_p, 'res_df_softmax.csv'))
        res_df_class_wise_softmax.to_csv(os.path.join(base_p, 'res_df_class_wise_softmax.csv'))
        cf_df_softmax.to_csv(os.path.join(base_p, 'cf_df_softmax.csv'))


if __name__ == '__main__':

    # mwe()
    dgcytof_pipeline()

    print('done')




