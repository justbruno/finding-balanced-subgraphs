import numpy as np
import scipy as sp
import random
import math
import importlib
from sklearn import metrics
import datetime
from scipy import sparse

DEBUG = 0


def compute_signed_laplacian(A):
    N = A.shape[0]
    At = -A
    for i in range(N):        
        x1 = A[i,:].T                
        At[i,i] = np.sum(np.abs(x1))
    return At


def compute_signed_laplacian_sparse(A):
    D = sparse.spdiags(np.sum(np.abs(A), axis=0), 0, A.shape[0], A.shape[1])
    return D-A


def remove_vertices(A, vertices):
    """
    Remove vertices indexed by given list
    """
    N = A.shape[0]
    r = np.arange(N)
    remaining = np.delete(r,vertices)
    return A[remaining,:][:,remaining]


def rank_vertices(A, L=None, s=None, v=None):
    if L == None:
        L = compute_signed_laplacian(A)
        s,V = sparse.linalg.eigsh(L)
        v = V[:,0]**2
    l=s[0]
    L=-np.abs(L) + sparse.spdiags(2*np.sum(np.abs(A), axis=0), 0, L.shape[0], L.shape[1]) \
    - sparse.spdiags(np.ones(L.shape[0])*2*l, 0, L.shape[0], L.shape[1])
    df=-L.dot(v)
    return (l-df)/(1.-v)


def read_sparse_matrix(filename):
    from scipy import sparse
    with open(filename, 'r') as f:
        data, rows, cols = [], [], []
        size = int(f.readline().strip('#'))
        for l in f:
            s=l.strip('\n').split()
            data.append(int(s[2]))
            rows.append(int(s[0]))
            cols.append(int(s[1]))
        Ac=sparse.csr_matrix((data, (rows, cols)), shape=(size, size)).astype(np.float)

    # Redundant info is aggregated by scipy, so we turn non-1's into 1's
    Ac[Ac>1]=1
    Ac[Ac<-1]=-1
    return Ac+Ac.T


def find_connected_components(A):
    Ab = np.abs(A)
    A = Ab.copy()
    all_indices = np.arange(A.shape[0])
    remaining_indices = np.arange(A.shape[0])
    removed_indices = []
    components = []

    while (A.shape[0] > 0):
        # Start with the neighbours of the max-degree vertex
        maxdeg = np.argmax(np.sum(A, axis=0))
        component = np.hstack([[maxdeg],np.nonzero(A[maxdeg,:])[1]])
        # BFS
        prelen = 0
        while len(component) > prelen:
            prelen = len(component)
            B = np.sum(A[component,:], axis=0)
            component = np.unique(np.hstack([component, np.nonzero(B)[1]]))
        components.append(remaining_indices[component])
        remaining_indices = np.delete(remaining_indices, component)
        A=Ab[remaining_indices,:][:,remaining_indices]
    return components


def estimate_ep_ED(A, x=[], maxiter=None):
    N = A.shape[0]
    SHIFT = N
    L = compute_signed_laplacian_sparse(A)
    H = sparse.spdiags(np.ones(N)*SHIFT, 0, A.shape[0], A.shape[1])
    s,V = sparse.linalg.eigsh(-L+H, k=1)
    s=N-s
    return s,V

def estimate_ep_CG(A, x=[], maxiter=100):
    if len(x) == 0:
        x=np.random.random((A.shape[0], 1))
    L = compute_signed_laplacian_sparse(A)
    s,V = sparse.linalg.lobpcg(L, x, largest=False, tol=1e-12, maxiter=maxiter)
    return s,V


def is_balanced_(A, sv=0):

    #-1 means we switch this node, 1 means we keep its sign, 0 means we have not visited it yet
    sign_of_node=A.shape[0]*[0]
    #sign of first node
    sign_of_node[0]=1
    #add the first node to the current level
    priority_queue=np.array((A.shape[0]+1)*[-1])
    start_pointer=0
    end_pointer=0
    priority_queue[end_pointer]=sv
    end_pointer+=1

    while start_pointer!=end_pointer:
        curr_node=int(priority_queue[start_pointer])
        priority_queue[start_pointer]=-1
        start_pointer+=1
        #print("removed: "+str(curr_node))
        adjacency_vector=A[curr_node,:]
        #print("neighbours: "+str(np.nonzero(adjacency_vector)[1]))
        for neighbour in np.nonzero(adjacency_vector)[1]:
            #the other endpoint of this edge has not beed explored so we assign it a sign and add it to the queue
            if sign_of_node[neighbour]==0:
                #assign according to algorithm in zaslavsky paper
                sign_of_node[neighbour]=sign_of_node[curr_node]*A[curr_node,neighbour]
                priority_queue[end_pointer]=neighbour
                end_pointer+=1
                #print("this neighbour has not been visited so is being added to the list: "+str(neighbour))
            #check if this edge is negative after the switch
            if sign_of_node[neighbour]*A[curr_node,neighbour]*sign_of_node[curr_node] < 0 :
                return False
    return True


def is_balanced(A):
    N = A.shape[0]
    if N < 3:
        return True

    first = np.argmax(np.sum(np.abs(A), axis=0))
    switch = A[first, :]

    s = switch<=-1
    neg = np.zeros(N)
    negi = sparse.find(s)[1]
    neg[negi]=1

    s = switch>=1
    pos = np.zeros(N)
    posi = sparse.find(s)[1]
    pos[posi]=1

    finished = False
    while not finished:
        posi = sparse.find(pos)[1]
        rows,cols=sparse.find(A[posi, :]>=1)[:2]
        pos = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + pos, 1, 0)
        rows,cols=sparse.find(A[posi, :]<=-1)[:2]
        neg = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + neg, 1, 0)

        negi = sparse.find(neg)[1]
        rows,cols=sparse.find(A[negi, :]>=1)[:2]
        neg = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + neg, 1, 0)
        rows,cols=sparse.find(A[negi, :]<=-1)[:2]
        pos = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + pos, 1, 0)

        pos = pos[0]
        neg = neg[0]

        balanced = pos.T.dot(neg) == 0
        switch = pos-neg
        finished = switch.T.dot(switch)==A.shape[0]
        if not balanced:
            break
    return balanced


def get_indicator_vector(A):
    N = A.shape[0]
    first = np.argmax(np.sum(np.abs(A), axis=0))
    switch = A[first, :]

    s = switch==-1
    neg = np.zeros(N)
    negi = sparse.find(s)[1]
    neg[negi]=1

    s = switch==1
    pos = np.zeros(N)
    posi = sparse.find(s)[1]
    pos[posi]=1

    finished = False
    while not finished:
        posi = sparse.find(pos)[1]
        rows,cols=sparse.find(A[posi, :]==1)[:2]
        pos = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + pos, 1, 0)
        rows,cols=sparse.find(A[posi, :]==-1)[:2]
        neg = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + neg, 1, 0)

        negi = sparse.find(neg)[1]
        rows,cols=sparse.find(A[negi, :]==1)[:2]
        neg = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + neg, 1, 0)
        rows,cols=sparse.find(A[negi, :]==-1)[:2]
        pos = np.where(np.sum(sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=A.shape), axis=0) + pos, 1, 0)

        pos = pos[0]
        neg = neg[0]

        balanced = pos.T.dot(neg) == 0
        switch = pos-neg
        finished = switch.T.dot(switch)==A.shape[0]
        if not balanced:
            break
    return switch


def create_graph(edgelist):
    graph = {}
    for e1, e2 in edgelist:
        graph.setdefault(e1, []).append(e2)
        graph.setdefault(e2, []).append(e1)
    return graph

# Prim's
def mst(start, A, graph):
    signs_of_nodes={}
    signs_of_nodes[start]=1
    closed = set()
    rows = []
    cols = []
    edges = []
    q = [(start, start)]
    while q:
        v1, v2 = q.pop()
        if v2 in closed:
            continue
        closed.add(v2)
        if signs_of_nodes[v1]*A[v1,v2]<0:
            signs_of_nodes[v2]=-1
        else:
            signs_of_nodes[v2]=1

        edges.append((v1, v2))
        rows.append(v1)
        cols.append(v2)
        for v in graph[v2]:
            if v in graph:
                q.append((v2, v))
    del edges[0]
    assert len(edges) == len(graph)-1
    return edges,rows[1:], cols[1:],signs_of_nodes

def print_stats(A):
    d,W = sparse.linalg.eigsh(A, k=1)
    v = W[:,0]
    x = np.sign(v)
    edges = np.sum(np.abs(A))/2.#float(sparse.spmatrix.dot(x.T, A).dot(x))/2
    degrees = np.array(np.sum(np.abs(A), axis=0))
    x1 = np.where(x==1, 1, 0)
    x2 = np.where(x==-1, 1, 0)
    edges1 = float(sparse.spmatrix.dot(x1.T, A).dot(x1))/2
    edges2 = float(sparse.spmatrix.dot(x2.T, A).dot(x2))/2
    edges_ac = -float(sparse.spmatrix.dot(x1.T, A).dot(x2))

    N = A.shape[0]
    M = np.sum(np.abs(A))/2.
    print('balanced: {}'.format(is_balanced(A)))
    print('size: {}'.format(A.shape[0]))
    print('edges: {}'.format(edges))
    print('edges_C1: {}'.format(edges1))
    print('edges_C2: {}'.format(edges2))
    print('edges_across: {}'.format(edges_ac))
    print('mean_degree: {}'.format(np.mean(degrees)))
    print('median_degree: {}'.format(np.median(degrees)))
    print('max_degree: {}'.format(np.max(degrees)))
    print('min_degree: {}'.format(np.min(degrees)))
    print('C1: {}'.format((x1.T.dot(x1))))
    print('C2: {}'.format((x2.T.dot(x2))))
    print('x_size: {}'.format(x.shape))
    print('x_nnz: {}'.format(x.dot(x)))
    print()
