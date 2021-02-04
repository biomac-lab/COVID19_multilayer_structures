import networkx as nx
import community as community_louvian
import heapq
from operator import itemgetter

def get_partition(G, n_biggest=3):

    # part_runs = np.zeros([len(G),runs])
    # for i in range(0,runs):
    #     partition_i    = community_louvian.best_partition(G)     # Get partition/clusters
    #     part_runs[:,i] = np.array(list(partition_i.values()))
    
    # part_mode = mode(part_runs, axis=1)[0].tolist()              # Get mode of partitions
    # part_mode = [int(val[0]) for val in part_mode]
    # nodes_z   = list(np.arange(0,len(G)))
    #partition = dict(zip(nodes_z,part_mode))            # Create dictionary {node:partition}

    partition = community_louvian.best_partition(G, random_state=0)     # Get partition/clusters

    n_cluster = max(partition.values())+1               # Get number of clusters

    cluster_nodes = {}                                  # Get nodes per cluster {cluster:[nodes]}
    
    for key, value in partition.items(): 
        if value not in cluster_nodes: 
            cluster_nodes[value] = [key] 
        else: 
            cluster_nodes[value].append(key)

    n_nodes_in_cluster = { cluster:len(nodes) for cluster, nodes in cluster_nodes.items() }       # Get number of nodes per cluster

    top_clusters = dict(heapq.nlargest(n_biggest, n_nodes_in_cluster.items(), key=itemgetter(1))) # Get n biggest cluster

    # small_clusters = dict(heapq.nsmallest(n_biggest-1, n_nodes_in_cluster.items(), key=itemgetter(1))) # Get n smallest cluster
    
    top_cluster_nodes = {cluster:cluster_nodes[cluster] for cluster,_ in top_clusters.items()}    # Get biggest clusters items

    # small_cluster_nodes = {cluster:cluster_nodes[cluster] for cluster,_ in small_clusters.items()}    # Get smallest clusters items
    
    return partition, n_cluster, cluster_nodes, top_clusters, top_cluster_nodes

