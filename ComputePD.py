import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gudhi
import gudhi.wasserstein
from scipy.spatial import distance_matrix
from utils import __plot_persistence_diagram_

def computePD(X, min_persistence=0, max_dimension=2, max_edge_length=None):
    '''
    Input
        X - point cloud in NxD, N is number of samples, D is dimension
        min_persistence - only preserve structures that last longer than the minimum persitence 
        max_dimension - the maximum geometry dimension used to construct a complex
        max_edge_length - the maximum edge length used to construct a Rips complex
    Return
        PD - the persistence diagram (as a numpy array of dimension 2) of X
    '''
    X = np.asarray(X)
    assert len(X.shape) == 2, 'Input should be shaped as NxD'

    dm = distance_matrix(X, X)
    if max_edge_length:
        assert max_edge_length>0, 'Invalid max_edge_length.'
        st = gudhi.RipsComplex(distance_matrix=dm, max_edge_length=max_edge_length).create_simplex_tree(max_dimension=max_dimension)
    else:
        st = gudhi.RipsComplex(distance_matrix=dm).create_simplex_tree(max_dimension=max_dimension)
    st.compute_persistence(homology_coeff_field=2, min_persistence=min_persistence, persistence_dim_max=False)
    PD = st.persistence_intervals_in_dimension(0)
    if max_dimension > 1:
        for d in range(1, max_dimension):
            PD = np.vstack((PD, st.persistence_intervals_in_dimension(d)))
    return PD

def totalPersistence(PD, dimension='zero'):
    '''
    Input
        PD - The persistence diagram (as a numpy array of dimension 2) used for computation. 
        dimension - The dimension of structures. Only this dimension will be considerer for computing.
                    All dimension will be considered if value is 'all'. Choices = ['zero', 'one', 'all']
    Return
        P - total persistence value, excluding infinity values
    '''
    if dimension == 'zero':
        P = np.sum([p[1]-p[0] for p in PD if (p[1]!=np.inf and p[0]==0)])
    elif dimension == 'one':
        P = np.sum([p[1]-p[0] for p in PD if (p[1]!=np.inf and p[0]!=0)])
    elif dimension == 'all':
        P = np.sum([p[1]-p[0] for p in PD if p[1]!=np.inf])
    else:
        raise AttributeError('diemnsion \'{:}\' not supported.'.format(diemnsion))
    return P


def distance(PD1, PD2, metric='Wasserstein', order=1.0, max_dimension=2):
    '''
    Input
        PD1, PD2 - two persistence diagrams returned by computePD() function
        metric -  the metric used to compute distance between two PDs. Choices = {'Wasserstein', 'Bottleneck'}
        order - the order of Wasserstein distance. Only useful when metric='Warsserstein'
        max_dimension - the maximum geometry dimension used to construct a complex. 
                        Currently only support 2.
    Return
        distance between PD1 and PD2 under 'metric', in a form as 
        [distance_0_dimension, distance_1_dimension, ..., distance_d_dimension, total_distance]
    '''
    assert max_dimension == 2, 'max_dimension not supported.'
    distances = []
    for dimension in range(max_dimension):
        if dimension == 0:
            PD1, PD2 = PD1[PD1[:,0]==0], PD2[PD2[:,0]==0]
        elif dimension == 1:
            PD1, PD2 = PD1[PD1[:,0]!=0], PD2[PD2[:,0]!=0]
        if metric == 'Wasserstein':
            distance = gudhi.wasserstein.wasserstein_distance(PD1, PD2, order=order)
        elif metric == 'Bottleneck':
            distance = gudhi.bottleneck_distance(PD1, PD2)
        else:
            raise AttributeError('metric \'{:}\' not supported.'.format(metric))
        distances.append(distance)
    distances.append(np.sum(distances))
    return distances

def plotPD(PD, MAX_DEATH=None, root='./', filename='dgm.png', tilte='Persistence Diagram'):
    '''
    Input
        PD - persistence diagram to plot
        MAX_DEATH - Unify the scale of y-axis by MAX_DEATH. If None, MAX_DEATH is chosen by 
                    the maximum death time of current PD. This parameter is only useful when 
                    comparing different PDs.
        root - root directory to save the figure
        filename - name of the saved figure
        title - figure title
    '''
    max_death = max([x[1] for x in PD if x[1]!=np.inf])
    if MAX_DEATH:
        if max_death > MAX_DEATH:
            warnings.warn("The MAX_DEATH is not chosen properly (too small). Some points will not show up.")
        ax = __plot_persistence_diagram_(PD, MAX_DEATH)
        ax.set_title(tilte)
        plt.savefig(os.path.join(root, filename))
    else:
        ax = gudhi.plot_persistence_diagram(PD)
        ax.set_title(tilte)
        plt.savefig(os.path.join(root, filename))


if __name__ == '__main__':
    '''Create point clouds'''
    X = np.array([[1,0,0], [0,1,0], [0,0,1]])
    Y = np.array([[1,0,0], [0,0,0], [-1,0,0]])

    '''Compute PDs from point clouds'''
    PDX = computePD(X)
    PDY = computePD(Y)
    print(PDX.shape)
    print(PDY.shape)

    '''Compute distance between PDs'''
    metrics = ['Wasserstein', 'Bottleneck'] # Choose a metric
    d = distance(PDX, PDY, metric=metrics[0], order=2.0)
    print(d)

    '''Plot PD if needed'''
    #plotPD(PDX)

    '''
    If intend to compare PDX and PDY qualitively, the same scale of y-axis is recommended.
    It is achieved by setting the MAX_DEATH as the maximum death time (except infinit) in two PDs.
    The scale of x-axis can be set inside plotPD() function by ax.set_xlim() if needed.
    '''
    #MAX_DEATH = max([x[1] for x in np.vstack((PDX, PDY)) if x[1]!=np.inf])
    #plotPD(PDX, MAX_DEATH=MAX_DEATH, filename='dgmX.png')
    #plotPD(PDY, MAX_DEATH=MAX_DEATH, filename='dgmY.png')

    '''
    Calculate total persistence
    '''
    # P_0 = totalPersistence(PDX, dimension='zero') # Total 0-dimension persistence
    # P_1 = totalPersistence(PDX, dimension='one') # Total 1-dimension persistence
    # P = totalPersistence(PDX, dimension='all') # Total persistence
    # print(P_0, P_1, P)


