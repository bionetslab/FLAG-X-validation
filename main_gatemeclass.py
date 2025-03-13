

def check_r():
    from rpy2.robjects import r

    r_home = r['R.home']()
    print(f"R home detected by rpy2: {r_home}")

    r_version = r('R.version.string')[0]
    print(f"R version detected by rpy2: {r_version}")


def mwe():
    import os
    import time
    import numpy as np
    import pandas as pd
    from typing import Tuple
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri

    # ### Data processing in R tutorial (https://github.com/simo1c/GateMeClass/tree/main/tutorial)
    # ## Load dataset => SummarizedExperiment object with: Assay data, row metadata, column metadata
    # d_SE < - Levine_32dim_SE()
    # ## Get expression matrix for columns (markers) that are annotated as "type" (relevant to cell type identification)
    # d_sub < - assay(d_SE[, colData(d_SE)$marker_class == "type"])
    # ## Extract the population labels => vector
    # population < - rowData(d_SE)$population_id
    # ### Perform arcsinh transform
    # cofactor < - 5
    # d_sub < - asinh(d_sub / cofactor)
    # ## Subset data to cells that are not "unassigned", expression matrix is transposed!!!
    # exp_matrix < - t(d_sub[population != "unassigned",])
    # population < - population[population != "unassigned"]

    # ### Load data
    data_train = np.load(os.path.join(os.getcwd(), 'results/baseline_less_channels/data_handling/data_train.npy'))
    data_test = np.load(os.path.join(os.getcwd(), 'results/baseline_less_channels/data_handling/data_test.npy'))

    # Flag to truncate dataset for debug purposes
    mwe_ = True
    if mwe_:
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

    x_train_df = pd.DataFrame(
        x_train.T,
        index=[f'MRKR{i}' for i in range(1, x_train.shape[1] + 1)],
    )

    ro.r('''
        library(GateMeClass)
        ''')

    # ### Run Gatemeclass
    # Activate numpy-to-R bridge
    numpy2ri.activate()
    pandas2ri.activate()

    # Load the necessary R libraries

    # Convert numpy arrays to R objects
    # r_x_train = numpy2ri.py2rpy(x_train.T.copy())
    r_x_train = pandas2ri.py2rpy(x_train_df)

    r_y_train = numpy2ri.py2rpy(y_train.astype(str).copy())  # GMC needs character vector as label input

    # Define the R function for GateMeClass training and annotation
    ro.r('''
        train_and_annotate <- function(exp_matrix, labels) {
            
            print(exp_matrix[,1:20])            
            print(type(exp_matrix))  # double
            print(class(exp_matrix))  # matrix, array
            print(dim(exp_matrix))  # channels x events
            
            exp_matrix <- as.matrix(exp_matrix)
            
            print(exp_matrix[,1:20])            
            print(type(exp_matrix))  # double
            print(class(exp_matrix))  # matrix, array
            print(dim(exp_matrix))  # channels x events
            
            print(labels[1:20])
            print(type(labels))  # character
            print(class(labels))  # array
            
            labels <- factor(labels)
            
            print(labels[1:20])
            print(type(labels))  # character
            print(class(labels))  # array
            
            
            inference <- TRUE
            if (inference){
            # GateMeClass training
            new_gate <- GateMeClass_train(
                reference = exp_matrix,
                labels = labels,
                GMM_parameterization = "V",
                verbose = TRUE,
                seed = 1
            )
            
            # reference             : The expression matrix of the reference annotated dataset.
            # labels                : A character vector with the labels of the reference dataset.
            # GMM_parameterization  : A character vector with the GMM (Gaussian-Mixture-Model) variance parameter: "V" (Variable) or "E" (Equal).
            # verbose               : TRUE to show output information.
            # seed                  : Seed for randomization.
            
            print('# ### Annotation finished with marker table:')
            print(new_gate)
            
            # GateMeClass annotation
            res <- GateMeClass_annotate(
                exp_matrix = exp_matrix,
                marker_table = new_gate,
                reject_option = FALSE,
                GMM_parameterization = "V",
                k = 20,
                sampling = 0.1,
                verbose = TRUE,
                seed = 1
            )
            
            # exp_matrix          : An expression matrix.
            # marker_table        : A data.frame with a manually curated or extracted marker table.
            # reject_option       : If TRUE this parameter tries to detect cell types not defined in the marker table using MNN algorithm.
            # GMM_parameterization: A character vector with the GMM (Gaussian-Mixture-Model) variance parameter: "V" (Variable) or "E" (Equal).
            # k                   : k parameter of k-NN (k-Nearest-Neighbour) used to refine uncertain labels to the most similar already annotated.
            # sampling            : Percentage of the cells used for the annotation.
            # narrow_marker_table : format of marker table.
            # verbose             : TRUE to show output information.
            # train_parameters    : A list with the parameters for the training function, i.e., reference and labels.
            # seed                : Seed for randomization.
    
            # Return results
            list(
                labels = res$labels,
                marker_table = res$marker_table,
                cell_signatures = res$cell_signatures
            )
            }
        }
    ''')

    # Call the R function
    gate_meclass_func = ro.globalenv['train_and_annotate']
    result = gate_meclass_func(r_x_train, r_y_train)

    # Convert R results back to numpy arrays
    labels = np.array(result[0])           # Annotation labels
    marker_table = np.array(result[1])     # Marker table
    cell_signatures = np.array(result[2])  # Cell signatures

    # Print or use the numpy results
    print("Labels:", labels)
    print("Marker Table:", marker_table)
    print("Cell Signatures:", cell_signatures)


def mwe_clf():
    import os
    import time
    import numpy as np
    import pandas as pd
    from typing import Tuple
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri

    # ### Data processing in R tutorial (https://github.com/simo1c/GateMeClass/tree/main/tutorial)
    # ## Load dataset => SummarizedExperiment object with: Assay data, row metadata, column metadata
    # d_SE < - Levine_32dim_SE()
    # ## Get expression matrix for columns (markers) that are annotated as "type" (relevant to cell type identification)
    # d_sub < - assay(d_SE[, colData(d_SE)$marker_class == "type"])
    # ## Extract the population labels => vector
    # population < - rowData(d_SE)$population_id
    # ### Perform arcsinh transform
    # cofactor < - 5
    # d_sub < - asinh(d_sub / cofactor)
    # ## Subset data to cells that are not "unassigned", expression matrix is transposed!!!
    # exp_matrix < - t(d_sub[population != "unassigned",])
    # population < - population[population != "unassigned"]

    # ### Load data
    data_train = np.load(os.path.join(os.getcwd(), 'results/baseline_less_channels/data_handling/data_train.npy'))
    data_test = np.load(os.path.join(os.getcwd(), 'results/baseline_less_channels/data_handling/data_test.npy'))

    # Flag to truncate dataset for debug purposes
    mwe_ = True
    if mwe_:
        # ### Only for testing
        # Use subset for faster inference time
        data_train_df = pd.DataFrame(
            data=data_train,
        )
        data_train = data_train_df.sample(frac=1, random_state=42).reset_index(drop=True)[0:10000].to_numpy()
        data_test_df = pd.DataFrame(
            data=data_test,
        )

    x_train = data_train[:, :-1].copy()
    y_train = data_train[:, -1].copy()

    x_train_df = pd.DataFrame(
        x_train.T,
        index=[f'Marker{i}' for i in range(1, x_train.shape[1] + 1)],
    )

    ro.r('''
        library(GateMeClass)
        ''')

    # ### Run Gatemeclass
    # Activate numpy-to-R bridge
    numpy2ri.activate()
    pandas2ri.activate()

    # Load the necessary R libraries

    # Convert numpy arrays to R objects
    # r_x_train = numpy2ri.py2rpy(x_train.T.copy())
    r_x_train = pandas2ri.py2rpy(x_train_df)
    r_y_train = numpy2ri.py2rpy(y_train.astype(str).copy())  # GMC needs character vector as label input

    # Define the R function for GateMeClass training and annotation
    ro.r('''
        gmc_fit <- function(exp_matrix, labels, GMM_parameterization, verbose, seed){
        
            print('######')
            print(GMM_parameterization)
            print(verbose)
            print(seed)
            print('######')
            
            print(exp_matrix[,1:20])            
            print(type(exp_matrix))  # double
            print(class(exp_matrix))  # data.frame
            print(dim(exp_matrix))  # channels x events

            exp_matrix <- as.matrix(exp_matrix)

            print(exp_matrix[,1:20])            
            print(type(exp_matrix))  # double
            print(class(exp_matrix))  # matrix, array
            print(dim(exp_matrix))  # channels x events

            print(labels[1:20])
            print(type(labels))  # character
            print(class(labels))  # array

            labels <- factor(labels)

            print(labels[1:20])
            print(type(labels))  # character
            print(class(labels))  # factor


            inference <- TRUE
            if (inference){
            # GateMeClass training
            gate <- GateMeClass_train(
                reference = exp_matrix,
                labels = labels,
                GMM_parameterization = "V",
                verbose = TRUE,
                seed = 1
            )

            # reference             : The expression matrix of the reference annotated dataset.
            # labels                : A character vector with the labels of the reference dataset.
            # GMM_parameterization  : A character vector with the GMM (Gaussian-Mixture-Model) variance parameter: "V" (Variable) or "E" (Equal).
            # verbose               : TRUE to show output information.
            # seed                  : Seed for randomization.
            
            print(type(gate))  # character
            print(class(gate))  # data.frame
            
            return(gate)
            
            }
        }
        ''')

    # Define the R function for GateMeClass training and annotation
    ro.r('''
        gmc_predict <- function(exp_matrix, marker_table) {

            print(exp_matrix[,1:20])            
            print(type(exp_matrix))  # double
            print(class(exp_matrix))  # matrix, array
            print(dim(exp_matrix))  # channels x events

            exp_matrix <- as.matrix(exp_matrix)

            print(exp_matrix[,1:20])            
            print(type(exp_matrix))  # double
            print(class(exp_matrix))  # matrix, array
            print(dim(exp_matrix))  # channels x events
            
            print(marker_table)
            print(type(marker_table))  # character
            print(class(marker_table))  # data.frame
            
            inference <- TRUE
            if (inference){

            # GateMeClass annotation
            res <- GateMeClass_annotate(
                exp_matrix = exp_matrix,
                marker_table = marker_table,
                reject_option = TRUE,
                GMM_parameterization = "V",
                k = 20,
                sampling = 0.1,
                verbose = TRUE,
                narrow_marker_table = TRUE,
                seed = 1
            )

            # exp_matrix          : An expression matrix.
            # marker_table        : A data.frame with a manually curated or extracted marker table.
            # reject_option       : If TRUE this parameter tries to detect cell types not defined in the marker table using MNN algorithm.
            # GMM_parameterization: A character vector with the GMM (Gaussian-Mixture-Model) variance parameter: "V" (Variable) or "E" (Equal).
            # k                   : k parameter of k-NN (k-Nearest-Neighbour) used to refine uncertain labels to the most similar already annotated.
            # sampling            : Percentage of the cells used for the annotation.
            # narrow_marker_table : format of marker table.
            # verbose             : TRUE to show output information.
            # train_parameters    : A list with the parameters for the training function, i.e., reference and labels.
            # seed                : Seed for randomization.
            
            print(type(res$labels))  # character
            print(class(res$labels))  # character
            print(type(res$marker_table))  # character
            print(class(res$marker_table))  # data.frame
            print(type(res$cell_signatures))  # character
            print(class(res$cell_signatures))  # data.frame
            
            # Return results
            list(
                labels = res$labels,
                marker_table = res$marker_table,
                cell_signatures = res$cell_signatures
            )
            }
        }
        ''')

    # Call the R function
    gmc_fit = ro.globalenv['gmc_fit']
    fit_params = {
        "GMM_parameterization": "V",  # Gaussian Mixture Model variance parameter
        "verbose": True,  # Show output information
        "seed": 1,  # Randomization seed
    }
    marker_table = gmc_fit(r_x_train, r_y_train, **fit_params)
    print(type(marker_table))
    marker_table = pandas2ri.rpy2py(marker_table)
    print(type(marker_table))
    print(f'Marker table:\n{marker_table}')

    marker_table = marker_table[0:3]

    print(marker_table)

    r_marker_table = pandas2ri.py2rpy(marker_table)
    gmc_predict = ro.globalenv['gmc_predict']
    result = gmc_predict(r_x_train, r_marker_table)

    print(type(result))
    print(type(result[0]))
    print(type(result[1]))
    print(type(result[2]))

    # Convert R results back to numpy arrays
    print(result[0])
    labels = result[0]  # .astype(float).astype(int)
    # Annotation labels, no conversion necessary, automatically handled by numpy2ri
    # => str array with labels and possibly 'Unclassified'
    marker_table = pandas2ri.rpy2py(result[1])  # Marker table
    cell_signatures = pandas2ri.rpy2py(result[2])  # Cell signatures

    # Print or use the numpy results
    print("Labels:\n", labels)
    print(np.unique(labels))
    print("Marker Table:\n", marker_table)
    print("Cell Signatures:\n", cell_signatures)

    print(labels.shape)
    print(x_train_df.shape)


def mwe2():
    import os
    import time
    import numpy as np
    import pandas as pd
    from typing import Tuple
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri

    # ### Data processing in R tutorial (https://github.com/simo1c/GateMeClass/tree/main/tutorial)
    # ## Load dataset => SummarizedExperiment object with: Assay data, row metadata, column metadata
    # d_SE < - Levine_32dim_SE()
    # ## Get expression matrix for columns (markers) that are annotated as "type" (relevant to cell type identification)
    # d_sub < - assay(d_SE[, colData(d_SE)$marker_class == "type"])
    # ## Extract the population labels => vector
    # population < - rowData(d_SE)$population_id
    # ### Perform arcsinh transform
    # cofactor < - 5
    # d_sub < - asinh(d_sub / cofactor)
    # ## Subset data to cells that are not "unassigned", expression matrix is transposed!!!
    # exp_matrix < - t(d_sub[population != "unassigned",])
    # population < - population[population != "unassigned"]
    # ###### #

    # ### Load data
    data_train = np.load(os.path.join(os.getcwd(), 'results/baseline_less_channels/data_handling/data_train.npy'))
    data_test = np.load(os.path.join(os.getcwd(), 'results/baseline_less_channels/data_handling/data_test.npy'))

    # Flag to truncate dataset for debug purposes
    mwe_ = True
    if mwe_:
        # ### Only for testing
        # Use subset for faster inference time
        data_train_df = pd.DataFrame(
            data=data_train,
        )
        data_train = data_train_df.sample(frac=1, random_state=42).reset_index(drop=True)[0:1000].to_numpy()

    x_train = data_train[:, :-1].copy()
    y_train = data_train[:, -1].copy()

    # ### Create dummy markertable
    mt = {
        "Cell": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "Marker1": ["+", "*", "+", "-", "+", "-", "*", "-"],
        "Marker2": ["-", "+", "*", "-", "+", "-", "-", "*"],
        "Marker3": ["-", "*", "+", "-", "-", "*", "-", "+"],
        "Marker4": ["+", "-", "*", "+", "*", "-", "+", "-"],
        "Marker5": ["-", "-", "-", "*", "-", "-", "+", "*"],
        "Marker6": ["*", "+", "-", "-", "*", "+", "-", "+"],
        "Marker7": ["+", "-", "+", "-", "+", "-", "*", "+"],
        "Marker8": ["-", "-", "+", "+", "*", "-", "-", "*"],
        "Marker9": ["-", "*", "-", "-", "-", "+", "+", "-"],
        "Marker10": ["+", "-", "*", "+", "-", "*", "-", "-"]
    }

    marker_table = pd.DataFrame(mt)

    print(marker_table)

    # Turn data matrix into dataframe
    x_train_df = pd.DataFrame(
        x_train.T,
        index=marker_table.columns.tolist()[1:],
    )
    print(x_train_df)
    print(x_train_df.shape)


    # ### Run Gatemeclass
    # Load the necessary R libraries
    ro.r('''
    library(GateMeClass)
    ''')
    # Activate numpy-to-R bridge
    numpy2ri.activate()
    pandas2ri.activate()

    # Convert numpy array/pandas dataframe to R object
    # r_x_train = numpy2ri.py2rpy(x_train.T.copy())
    r_x_train = pandas2ri.py2rpy(x_train_df)

    # Convert to R matrix and remove column names
    # r_x_train = ro.r('as.matrix')(r_x_train)
    # ro.r('colnames(r_x_train) <- NULL')
    r_marker_table = pandas2ri.py2rpy(marker_table)

    # numpy2ri.deactivate()
    # pandas2ri.deactivate()

    # Define the R function for GateMeClass training and annotation
    ro.r('''
        annotate_cells <- function(exp_matrix, marker_table, 
                       reject_option = FALSE, 
                       GMM_parameterization = "V", 
                       k = 20, 
                       sampling = 0.1, 
                       verbose = TRUE,
                       narrow_marker_table = FALSE,
                       seed = 1) {
            
            print(exp_matrix[,1:20])
            print(type(exp_matrix))
            print(class(exp_matrix))
            print(marker_table)
            print(type(marker_table))
            print(class(marker_table))
            
            exp_matrix <- as.matrix(exp_matrix)
            print(exp_matrix[,1:20])
            print(type(exp_matrix))
            print(class(exp_matrix))
            print(dim(exp_matrix))
            
            inference <- TRUE
            if (inference){
                # Perform annotation using GateMeClass_annotate
                res <- GateMeClass_annotate(
                    exp_matrix = exp_matrix,
                    marker_table = marker_table,
                    reject_option = reject_option,
                    GMM_parameterization = GMM_parameterization,
                    k = k,
                    sampling = sampling,
                    verbose = verbose,
                    narrow_marker_table = narrow_marker_table,
                    seed = seed
                )
    
                # exp_matrix          : An expression matrix.
                # marker_table        : A data.frame with a manually curated or extracted marker table.
                # reject_option       : If TRUE this parameter tries to detect cell types not defined in the marker table using MNN algorithm.
                # GMM_parameterization: A character vector with the GMM (Gaussian-Mixture-Model) variance parameter: "V" (Variable) or "E" (Equal).
                # k                   : k parameter of k-NN (k-Nearest-Neighbour) used to refine uncertain labels to the most similar already annotated.
                # sampling            : Percentage of the cells used for the annotation.
                # narrow_marker_table : format of marker table.
                # verbose             : TRUE to show output information.
                # train_parameters    : A list with the parameters for the training function, i.e., reference and labels.
                # seed                : Seed for randomization.
    
                # Return results
                return(list(
                    labels = res$labels,
                    marker_table = res$marker_table,
                    cell_signatures = res$cell_signatures
                ))
            }
        }
    ''')

    # Call the R function
    gate_meclass_func = ro.globalenv['annotate_cells']
    result = gate_meclass_func(r_x_train, r_marker_table)

    if False:
        # Convert R results back to numpy arrays
        labels = np.array(result[0])  # Annotation labels
        marker_table = np.array(result[1])  # Marker table
        cell_signatures = np.array(result[2])  # Cell signatures

        # Print or use the numpy results
        print("Labels:", labels)
        print("Marker Table:", marker_table)
        print("Cell Signatures:", cell_signatures)

    # Todo: Try with correct input dtype in mwe ...


def mwe_tut():
    # ### Tutorial as can be found on github (https://github.com/simo1c/GateMeClass/tree/main/tutorial)
    # Run with conda env gmc3_tut
    import rpy2.robjects as ro

    ro.r('''
        library(GateMeClass)
        library(HDCytoData)
        library(tidyr)
        library(readxl)
        
        d_SE <- Levine_32dim_SE()
        d_sub <- assay(d_SE[, colData(d_SE)$marker_class == "type"])
        population <- rowData(d_SE)$population_id
        cofactor <- 5
        d_sub <- asinh(d_sub / cofactor)
        
        exp_matrix <- t(d_sub[population != "unassigned", ])
        population <- population[population != "unassigned"]
        
        print(exp_matrix[,1:20])
        print(rownames(exp_matrix)) # Marker/channel names
        print(colnames(exp_matrix)) # Null
        print(dim(exp_matrix))  # (32, 104184) channels are in rows!!!
        print(typeof(exp_matrix))  # "double"
        print(class(exp_matrix))  # "matrix" "array"
        
        print(population[1:20])
        print(unique(population))
        print(type(population))  # "character"
        print(class(population))  # "factor"
        
        gate <- as.data.frame(read_excel("Levine32.xlsx"))
        gate[is.na(gate)] <- "*"
        
        print(gate)  # As expected
        print(typeof(gate))  # "list"
        print(class(gate))  # "data.frame"
        
        inference <- FALSE
        if (inference){
            res <- GateMeClass_annotate(exp_matrix = exp_matrix,
                            marker_table = gate,
                            GMM_parameterization = "V",
                            reject_option = F,
                            sampling = 0.1,
                            k = 20,				
                            verbose = T,
                            narrow_marker_table = F,
                            seed = 1)
        
            table(res$labels)
            print(res$marker_table)      
            # print(res$cell_signatures)
            
            print("######")
            
            new_gate <- GateMeClass_train(reference = exp_matrix,
                              labels = population,
                              GMM_parameterization = "V",
                              verbose = T, 
                              seed = 1)
            
            print(new_gate)
            res <- GateMeClass_annotate(exp_matrix = exp_matrix,
                                marker_table = new_gate,
                                reject_option = F,
                                GMM_parameterization = "V",
                                k = 20,				
                                sampling = 0.1,
                                verbose = T,
                                seed = 1)
            table(res$labels)
            print(res$marker_table)
            # print(res$cell_signatures)
        }
        ''')


def mwe_tut2():
    # ### Tutorial as can be found on github (https://github.com/simo1c/GateMeClass/tree/main/tutorial)
    # Run with conda env gmc3_tut
    import rpy2.robjects as ro

    ro.r('''
        library(GateMeClass)
        library(HDCytoData)
        library(tidyr)
        library(readxl)

        d_SE <- Levine_32dim_SE()
        d_sub <- assay(d_SE[, colData(d_SE)$marker_class == "type"])
        population <- rowData(d_SE)$population_id
        cofactor <- 5
        d_sub <- asinh(d_sub / cofactor)

        exp_matrix <- t(d_sub[population != "unassigned", ])
        population <- population[population != "unassigned"]

        print(exp_matrix[,1:20])
        print(rownames(exp_matrix)) # Marker/channel names
        print(colnames(exp_matrix)) # Null
        print(dim(exp_matrix))  # (32, 104184) channels are in rows!!!
        print(typeof(exp_matrix))  # "double"
        print(class(exp_matrix))  # "matrix" "array"

        print(population[1:20])
        print(unique(population))
        print(type(population))  # "character"
        print(class(population))  # "factor"

        inference <- TRUE
        if (inference){

            gate <- GateMeClass_train(reference = exp_matrix,
                              labels = population,
                              GMM_parameterization = "V",
                              verbose = T, 
                              seed = 1)

            print(gate)
            
            res <- GateMeClass_annotate(exp_matrix = exp_matrix,
                                marker_table = gate,
                                reject_option = F,
                                GMM_parameterization = "V",
                                k = 20,				
                                sampling = 0.1,
                                verbose = T,
                                seed = 1)
            table(res$labels)
            print(res$marker_table)
            # print(res$cell_signatures)
        }
        ''')


def gatemeclass_pipeline():
    import os
    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from typing import List, Union
    from validation.gatemeclass import GateMeClassClassifier
    from validation.val_utils import evaluate

    def stratified_downsampling(X: np.ndarray, y: np.ndarray, ds_fraction: float = 0.25, random_seed: int = 42):
        # ### Set random seed
        np.random.seed(random_seed)
        # ### Downsample train set to half its size in a stratified fashion for this study
        # Calculate the original class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_count = y.shape[0]
        class_ratios = class_counts / total_count
        # print(class_ratios)
        # print(class_counts)
        # print(x_train.shape)
        # Calculate the desired count for each class
        target_size = np.ceil(total_count * ds_fraction)
        target_counts = (class_ratios * target_size).round().astype(int)
        # Downsample
        downsampled_x = []
        downsampled_y = []
        for cls, count in zip(unique_classes, target_counts):
            # Get indices of the current class
            class_indices = np.where(y == cls)[0]
            # Randomly sample from these indices
            sampled_indices = np.random.choice(class_indices, size=count, replace=False)
            # Append downsampled data and labels
            downsampled_x.append(X[sampled_indices])
            downsampled_y.append(y[sampled_indices])
        # Concatenate results
        x = np.vstack(downsampled_x)
        y = np.concatenate(downsampled_y)

        return x, y

    # ### Load data
    data_train = np.load(os.path.join(os.getcwd(), 'results/baseline_less_channels/data_handling/data_train.npy'))
    data_test = np.load(os.path.join(os.getcwd(), 'results/baseline_less_channels/data_handling/data_test.npy'))

    # Flag to truncate dataset for debug purposes
    mwe_ = False
    if mwe_:
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

    downsample = False
    if downsample:
        x_train, y_train = stratified_downsampling(X=x_train, y=y_train, ds_fraction=0.01, random_seed=42)
        print('######')
        print(x_train.shape)
        # Dim of expression matrix in GMC tutorial is: (32, 104184)
        # Here we have: 0.01 - (29122, 10), 0.05 - (145611, 10)
        x_test, y_test = stratified_downsampling(X=x_test, y=y_test, ds_fraction=0.01, random_seed=42)
        print('######')
        print(x_train.shape)

    plot_hists = True
    if plot_hists:
        base_p = os.path.join(
            os.getcwd(), 'results/gatemeclass/'
        )
        if not os.path.exists(base_p):
            os.makedirs(os.path.join(base_p))

        def plot_helper(x: np.ndarray, channels: List[str], subdir: Union[str, None] = None):
            # ### For a datamatrix and channel names plot the histogram of the intensities of events (channel-wise)
            for i in range(x.shape[1]):
                fig, ax = plt.subplots(dpi=300)
                ax.hist(x[:, i], bins=200, color='lightblue', edgecolor='lightgrey')
                ax.set_xlabel('Intensity')
                ax.set_ylabel('# events')
                ax.set_title(channels[i])

                if subdir is None:
                    subdir = ''

                if not os.path.exists(os.path.join(base_p, subdir)):
                    os.makedirs(os.path.join(base_p, subdir))

                plt.savefig(os.path.join(base_p, subdir, f'hist_{channels[i]}.png'))
                plt.close('all')

        plot_helper(
            x=x_train,
            channels=['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
            subdir='hists'
        )

    # ### Initialize classifier
    # Set flag whether to allow classifications to include the new label 'unknown'
    allow_unknowns = False

    clf = GateMeClassClassifier(
        marker_names=['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO'],
        gmc_gmm_parameterization="V",
        gmc_k=20,
        gmc_sampling=0.1,
        gmc_reject_option=allow_unknowns,
        gmc_seed=1,
        time_fit_pred=True,
        verbosity=1
    )

    print('# ### Starting fit ...')
    st_fit = time.time()
    clf.fit(X=x_train, y=y_train)
    et_fit = time.time()
    fit_time = et_fit - st_fit
    print(f'# ### Fit finished, time: {fit_time}')

    print('# ### Starting GateMeClass prediction ...')
    st_pred = time.time()
    y_pred = clf.predict(X=x_test)
    et_pred = time.time()
    pred_time = et_pred - st_pred
    print(f'# ### Prediction GateMeClass finished, time: {pred_time}')

    times_df = pd.DataFrame(
        data=np.array([fit_time, pred_time]).reshape((1, 2)),
        index=['sek'],
        columns=['fit', 'pred']
    )

    print(f'# ### Times:\n{times_df}')

    print('# ### Results:')
    print(f'# ### Marker table:\n{clf.marker_table_}')

    if allow_unknowns:
        # If classification as 'unknown' is allowed, also generate results for prediction performance without unknowns
        res_df, res_df_class_wise, cf_df, y_test_unknowns_val_counts = evaluate(
            y_true=y_test, y_pred=y_pred, wout_unknowns=True
        )
    else:
        # Else Results for all labels (including unknowns/-1)
        res_df, res_df_class_wise, cf_df = evaluate(
            y_true=y_test, y_pred=y_pred, wout_unknowns=False
        )

    save = True
    if save:
        base_p = os.path.join(
            os.getcwd(), f'results/gatemeclass/{"allow_unknowns" if allow_unknowns else "no_unknowns"}'
        )
        if not os.path.exists(base_p):
            os.makedirs(os.path.join(base_p))

        clf.save(filepath=base_p)
        times_df.to_csv(os.path.join(base_p, 'times.csv'))
        res_df.to_csv(os.path.join(base_p, 'res_df.csv'))
        res_df_class_wise.to_csv(os.path.join(base_p, 'res_df_class_wise.csv'))
        cf_df.to_csv(os.path.join(base_p, 'cf_df.csv'))

        if allow_unknowns:
            y_test_unknowns_val_counts.to_csv(os.path.join(base_p, 'unclassifiable_label_count.csv'))

    print(clf.time_df_)


if __name__ == '__main__':

    # check_r()

    # mwe()

    # mwe_clf()

    # mwe2()

    # mwe_tut()

    # mwe_tut2()

    gatemeclass_pipeline()

    print('done')
