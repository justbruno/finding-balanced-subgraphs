import numpy as np
import threading
from multiprocessing import Process, Queue, Pool
import traceback
import timbal
import utils

THREADS = 4

def remove_vertices(A):
    # Removal phase of timbal, with minor differences
    MAX_REMOVALS=100
    MAX_REMOVALS = np.min([MAX_REMOVALS, A.shape[0]-1])
    
    # A pointer to the function we use to compute the eigenvalue/eigenvector of the Laplacian
    estimate_eigenpair= utils.estimate_ep_CG
    if A.shape[0] < 100:
        estimate_eigenpair=utils.estimate_ep_ED
        MAX_REMOVALS=1

    # We discard all connected components but the largest
    components = utils.find_connected_components(A)
    largest = components[np.argmax([len(i) for i in components])]
    A = A[largest,:][:,largest]
    # To store connected components that break away from the graph
    broken_components = []

    all_vertices = np.arange(A.shape[0])
    # Store removal sequence here
    true_remove_sequence=[]

    # The CG gives a better approximation if we run it a second time passing the previous vector as an initialization
    s,V = estimate_eigenpair(A)
    s,V = estimate_eigenpair(A,V)
    v=V[:,0]

    #print('Checking balance...')
    if utils.is_balanced(A):
        return [[]]

    while A.shape[0] > 2:
        if MAX_REMOVALS < 0:
            denom = float(np.abs(MAX_REMOVALS))
            MAX_REMOVALS = np.max([np.int(A.shape[0]/denom), 1])

        # When the graph becomes small, the eigendecomposition is fast and more realiable than CG
        if A.shape[0] < 100:
            estimate_eigenpair=utils.estimate_ep_ED
            MAX_REMOVALS = 10
        MAX_REMOVALS = np.min([MAX_REMOVALS, A.shape[0]-1])

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
        R = candidates
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
            broken_components = np.hstack([components[i] for i in range(len(components)) if i != largest])
            true_R = np.hstack([true_R, all_vertices[broken_components]])
            all_vertices = np.delete(all_vertices, broken_components)            
        true_remove_sequence.append(list(true_R))

        # Resize last computed eigenvector for initialization
        V = np.delete(V, R)
        #print(V.shape)
        if len(components) > 1:
            V = np.delete(V, broken_components).reshape((V.shape[0]-len(broken_components), 1))
        else:
            V = V.reshape((V.shape[0], 1))
        s,V = estimate_eigenpair(A, V)    
        v=V[:,0]

        # Check if we have a balanced graph
        if utils.is_balanced(A):
            break
    return true_remove_sequence       


def sample_subgraph_bfs(A, steps=4, first=0, avg_size=-1, p=0.2):
    current = first
    chosen = list(np.nonzero(A[current,:])[1])
    chosen.append(first)
    done = False
    for i in np.arange(steps):
        candidates = list(np.nonzero(np.sum(np.abs(A[chosen,:]), axis=0))[1])
        if len(candidates) < 1:
            break
        sample_size = np.min([int(np.exp(np.log(avg_size)/steps)), len(candidates)])
        new = list(np.random.choice(candidates, sample_size, replace=False))
        chosen.extend(new)
        if avg_size > 0 and len(chosen) >= avg_size:
            break
    return np.unique(chosen)


def pool_job_bfs(input_val=0):
    A, starters, steps, avg_size, thread_id = input_val
    removals = []
    for i in starters:
        try:
            S = sample_subgraph_bfs(A, steps=steps, first=i, avg_size=avg_size)
            As = A[S,:][:,S]
            remove_sequence = np.hstack(remove_vertices(As))
            if len(remove_sequence) > 0:
                removals.extend(S[remove_sequence])
        except Exception as e:
            print(e)
    return removals


def clean_graph_bfs(A, subgraphs=100, steps=1, avg_size=100):
    total_removals = []
    starters = np.random.choice(np.arange(A.shape[0]), subgraphs, replace=False)
    first = np.split(starters, THREADS)
    pool = Pool(THREADS)
    processes = [(A, first[i], steps, avg_size, i) for i in range(THREADS)]
    removals = pool.map(pool_job_bfs, processes)
    pool.close()
    removals = np.unique(np.hstack(removals))
    return removals
