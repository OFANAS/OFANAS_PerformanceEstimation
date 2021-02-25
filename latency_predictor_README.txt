Description of Code Files for Latency Prediction Project:
The entirety of the code base exists in tutorial/

To test our code, we have created a jupyter_notebook (tutorial/latency_prediction_demo.ipynb) which run through
all the required steps to show a working version of our project.

Setup:

conda env create -f environment.yml
conda activate latency_predict_env
jupyter notebook
Open tutorial/latency_prediction_demo.ipynb
Run all


Information about all the constituent files -

1.   latency_prediction_demo.ipynb
    -   Jupyter notebook Demo with steps to run Dataset Creation, Model Evaluation, and OFA NAS
    -   Lists all necessary dependencies for project

2.   latency_predictor_driver.py
    -   Provides code to create latency datasets. Uses imports from evolution_finder to random sample OFA, CompOFA
    -   Code for inference time analysis of latency predictor and measurement

3.   latency_NAS_runner.py
    -   Similar code to demo jupyter notebook. Prepares and run code to perform NAS


4.   evolution_finder.py (modified)
    -   this file was modified to include create_latency_dataset()

5.   latency_predictor/
    6.   datasets/
        -   contains all datasets created for this project, organized into device subfolders. Also includes GPU generalization datasets
    
    7.   model_results/
        -   includes images and text files of all experimental results organized in subfolders
    
    8.   generalized_dataset_combine.py
        -   helper code to add hardware parameters to datasets
    
    9.   Iterations.txt
        -   Log file from search
    
    10.   latency_encoding.py
        -   helper code to perform one hot encoding of child architectures
    
    11.   latency_finetune.py
        -   code for training fine tuned models, case studies, and plotting loss curves
    
    12.   latency_predictor.py
        -   code for creating latency predictors, data_preprocessing, training using RayTune, testing
    
    13.   latency_predictor_generalized.py
        -   code for creating generalized latency predictors
    
    14.   lookup_table_calculation.py
        -   helper code to calculate time taken to create lookup table
    
    15.   other images are results from various experiments


16. checkpoints/
    17.   latency_prediction_model/
        -   contains all trained latency predictors organized in subfolders of hardware and search spaces