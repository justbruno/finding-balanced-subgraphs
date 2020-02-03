import numpy as np
import timbal
import utils

DEBUG=0

dataset_dir='./'
datasets=['toy_data.txt']
output_directory='.'

for dataset in datasets:
    input_file=dataset_dir+dataset
    dataset_name=dataset.split('.')[0]

    A = utils.read_sparse_matrix(input_file)

    # TIMBAL
    # We discard all connected components but the largest
    components = utils.find_connected_components(A)
    largest = components[np.argmax([len(i) for i in components])]
    A = A[largest,:][:,largest]
    
    subsample_flag=False # Whether or not to sample random graphs. Recommended when the graph is large
    max_removals=1 # Max number of removed vertices per iteration
    samples = 4 # Randomly sampled graphs. Must be a factor of the number of threads used
    avg_size = 200 # Approximate desired size of randomly sampled graph
    
    S = timbal.process(A, max_removals=max_removals, samples=samples, avg_size=avg_size, subsample=subsample_flag)
    As = A[S,:][:,S]
    utils.print_stats(As)    
