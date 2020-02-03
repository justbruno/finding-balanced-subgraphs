import numpy as np
import scipy as sp
from scipy import sparse
import utils
import sampling

# Minimum size of the graph resulting from randomized removals.
MIN_SUBSAMPLED_GRAPH = 50000
# Maximum rounds of randomized removal. The process repeats until we reach the desired MIN_SUBSAMPLED_GRAPH size or MAX_ROUNDS
MAX_ROUNDS = 10

def process(Ac, **kwargs):
    N = Ac.shape[0]
    A = sparse.csr_matrix.copy(Ac)
        
    MAX_REMOVALS = kwargs['max_removals'] # Max number of removed vertices per iteration
    SAMPLES = kwargs['samples'] # Randomly sampled graphs
    AVG_SIZE = kwargs['avg_size'] # Max size of randomly sampled graph
    SUBSAMPLE = kwargs['subsample'] # Whether or not to sample random graphs
    
    # A pointer to the function we use to compute the eigenvalue/eigenvector of the Laplacian
    estimate_eigenpair= utils.estimate_ep_CG

    all_vertices = np.arange(Ac.shape[0])
    true_R = []

    # Random sampling phase
    if SUBSAMPLE:
        R = np.array([])
        rounds = 0
        while N-len(R) > MIN_SUBSAMPLED_GRAPH:
            random_removals = sampling.clean_graph_bfs(Ac, subgraphs=SAMPLES, steps=1, avg_size=AVG_SIZE)
            R = np.hstack([np.unique(random_removals), R])
            R = np.unique(R).astype(np.int)        
            rounds += 1
            if rounds >= MAX_ROUNDS:
                break

        # If we remove too many...        
        if N-len(R) < 10000:
            R = np.random.choice(R, N-10000, replace=False)        
        A = utils.remove_vertices(Ac, R)
        true_R.extend(all_vertices[R])
        all_vertices = np.delete(all_vertices, R)

    # To store connected components that break away from the graph
    broken_components = []

    # Store true removal sequence here. This starts from the largest connected component, not the full graph
    true_remove_sequence=list(true_R)

    # The CG gives a better approximation if we run it a second time passing the previous vector as an initialization
    s,V = estimate_eigenpair(A)
    s,V = estimate_eigenpair(A,V)
    v=V[:,0]

    vals = []

    MAX_REMOVALS = np.min([MAX_REMOVALS, A.shape[0]-1])
    
    while A.shape[0] > 2:
        if MAX_REMOVALS < 0: # If negative, adaptive removal cap
            denom = float(np.abs(MAX_REMOVALS))
            MAX_REMOVALS = np.max([np.int(A.shape[0]/denom), 1])

        # When the graph becomes small, the eigendecomposition is fast and more realiable than CG
        if A.shape[0] < 100:
            estimate_eigenpair=utils.estimate_ep_ED
            MAX_REMOVALS=1
        L = utils.compute_signed_laplacian_sparse(A)
        # Compute the upper bound for each vertex
        ub=utils.rank_vertices(A, L=L, s=s, v=v**2)
        candidates = [np.argmin(ub)]

        # Find independent set of vertices to remove
        neighbours = np.nonzero(L[candidates,:])[1]
        La = np.abs(L)
        rSum = La[candidates[0],:]
        while len(neighbours) < L.shape[0]:
            if len(candidates) >= MAX_REMOVALS:
                break
            ub[neighbours]=np.inf
            candidate = np.argmin(ub)
            candidates.append(candidate) 
            absR = La[candidate,:]
            rSum += absR+rSum
            neighbours = rSum.indices
        R = candidates[:MAX_REMOVALS]
        A = utils.remove_vertices(A, R)
        true_R = all_vertices[R]
        all_vertices = np.delete(all_vertices, R)

        # If we're left with an edge, break
        if A.shape[0] < 3:
            break        

        components = utils.find_connected_components(A)
        if len(components) > 1:
            largest = np.argmax([len(i) for i in components])
            A = A[components[largest],:][:,components[largest]]
            broken_components = np.hstack(np.delete(components, largest))
            true_R = np.hstack([true_R, all_vertices[broken_components]])
            all_vertices = np.delete(all_vertices, broken_components)
        true_remove_sequence.append(list(true_R))


        # Resize last computed eigenvector for CG initialization
        V = np.delete(V, R)
        if len(components) > 1:
            V = np.delete(V, broken_components).reshape((V.shape[0]-len(broken_components), 1))
        else:
            V = V.reshape((V.shape[0], 1))
        s,V = estimate_eigenpair(A, V)    
        v=V[:,0]

        # Store the current (normalized) smallest eigenvalue
        vals.append(np.min(s)/(A.shape[0]-2))

        if utils.is_balanced(A):
            break
        
    # Second stage
    gone = np.hstack(true_remove_sequence)
    remaining = np.delete(np.arange(Ac.shape[0]), gone)
    B=Ac[remaining,:][:,remaining]

    # Compute an indicator vector of all vertices in the input graph.
    # The partition of the located balanced subgraph is -1,1.
    # Removed vertices are 0.
    ind = utils.get_indicator_vector(B)
    full_ind = np.zeros(Ac.shape[0])
    j=0
    for i in remaining:
        full_ind[i] = ind[j]
        j+=1
    ind=full_ind

    # Restore vertices that agree with the partition
    for i in gone:
        connected = np.abs(Ac[i,:]).dot(np.abs(ind)) > 0
        if connected:        
            x = Ac[i,:]
            ip = x.dot(ind)
            nip = np.abs(x).dot(np.abs(ind))
            agree = np.abs(ip)==nip
            if agree:
                ind[i] = 1 if ip>=0 else -1
            else:
                pass
    remaining = np.nonzero(ind)[0]
    
    return remaining

