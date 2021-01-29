
# coding: utf-8

''' This script provides the python implementations for the analytical part of the thesis '''

# In[231]:


import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
#import pylab as plt
import math
import statistics as stat
import collections
import itertools
import statsmodels.api as sm
import random
import scipy
import time
import seaborn as sns
#
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1 import AxesGrid

from collections import Counter


# #### set global plot parameters

# In[1]:


sns.reset_defaults()
sns.set()
plt.rcParams.update({'font.size': 22})


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# In[3]:


data = pd.read('file')


# # Generate the Graphs

# ## Undirected graph

# In[21]:


graph_nd = nx.from_pandas_edgelist(data, source = 'CP1', target = 'CP2', edge_attr = ['exp', 'class'])


# ## Directed Graph

# It is necessary to manually build the sum of exposures over all asset classes, as constructing the network from the df only takes the edgeweight of the edge last added. Nr of edges shrinks due to 'condensation' of asset classes.

# In[18]:


graph_d = nx.DiGraph()
for cp_tuple in zip(data['CP1'], data['CP2']):
    sum_exp = sum(data.loc[(data['CP1'] == cp_tuple[0]) & (data['CP2'] == cp_tuple[1]), 'exp'])
    #print(cp_tuple[0], cp_tuple[1], sum_exp)
    graph_d.add_edge(cp_tuple[0], cp_tuple[1], exp = sum_exp)
    


# In[21]:


print(nx.info(graph_d)) 


# ## Multiplex Graph

# Again, it is neccessary to add each edge seperately to the multigraph and specify the corresponding asset class.

# ##### Note that technically this is not a multiplex yet, as the layers to not share the same set of nodes

# In[23]:


graph_mp = nx.MultiDiGraph()
for i, row in data.iterrows():
     graph_mp.add_edge(row['CP1'], row['CP2'], exp = row['exp'], key = row['class'])


# In[24]:


print(nx.info(graph_mp))


# ## Asset Class Graphs

# ##### Split the Multiplex graph into its respective layers represented by distinct directed graphs

# In[25]:


graph_CO = nx.DiGraph()
graph_CU = nx.DiGraph()
graph_CR = nx.DiGraph()
graph_IR = nx.DiGraph()
graph_EQ = nx.DiGraph()
for n, nbrs in graph_mp.adj.items():
    for nbr, eattr in nbrs.items():
        for key, attr in eattr.items():
            if key == "CO":
                graph_CO.add_edge(n, nbr, exp = attr["exp"])
            if key == "CU":
                graph_CU.add_edge(n, nbr, exp = attr["exp"])
            if key == "CR":
                graph_CR.add_edge(n, nbr, exp = attr["exp"])
            if key == "IR":
                graph_IR.add_edge(n, nbr, exp = attr["exp"])
            if key == "EQ":
                graph_EQ.add_edge(n, nbr, exp = attr["exp"])
            else:
                continue


# #### Commodity

# In[28]:


print(nx.info(graph_CO))


# #### Currency

# In[29]:


print(nx.info(graph_CU))


# #### Credit

# In[30]:


print(nx.info(graph_CR))


# #### Interest Rate

# In[31]:


print(nx.info(graph_IR))


# #### Equity

# In[32]:


print(nx.info(graph_EQ))


# # Component Decomposition and Connectivity
# 

# ## Weak connectivity
# 

# In[48]:


print(nx.info(graph_d))


# In[49]:


nx.is_weakly_connected(graph_d)


# In[50]:


nx.number_weakly_connected_components(graph_d)


# In[51]:


Counter([len(c) for c in sorted(nx.weakly_connected_components(graph_d),key=len, reverse=True)])


# ## Strong Connectivity

# In[52]:


nx.is_strongly_connected(graph_d)


# In[53]:


nx.number_strongly_connected_components(graph_d)


# In[54]:


Counter([len(c) for c in sorted(nx.strongly_connected_components(graph_d),key=len, reverse=True)])


# ## By Asset Classes

# ## Generate subgraph of Weakly Connected Component 

# ### Currency

# In[58]:


print(nx.info(graph_CU))


# In[59]:


nx.is_strongly_connected(graph_CU)


# In[60]:


nx.number_weakly_connected_components(graph_CU)


# In[61]:


Counter([len(c) for c in sorted(nx.weakly_connected_components(graph_CU),key=len, reverse=True)])


# In[62]:


Counter([len(c) for c in sorted(nx.strongly_connected_components(graph_CU),key=len, reverse=True)])


# In[63]:


max_scc_CU = max(nx.weakly_connected_components(graph_CU),key=len)


# In[64]:


graph_sc_CU = graph_CU.subgraph(max_scc_CU).copy()


# In[65]:


print(nx.info(graph_sc_CU))


# Total sum of values in the currency layer

# In[66]:


attr_sc_CU = nx.get_edge_attributes(graph_sc_CU, "exp")
attr_CU = nx.get_edge_attributes(graph_CU, "exp")
sum_CU = sum(attr_sc_CU.values())
sum_CU


# fraction of the sum of absolute values of WSC and original layer. <br>
# Indicates the relative size of the reduces graph i.e. how much is dropped constructing the WCC

# In[67]:


sum(map(abs,attr_sc_CU.values()))/sum(map(abs,attr_CU.values()))


# ### Commodity

# In[78]:


print(nx.info(graph_CO))


# In[79]:


nx.is_weakly_connected(graph_CO)


# In[80]:


nx.number_weakly_connected_components(graph_CO)


# In[81]:


Counter([len(c) for c in sorted(nx.weakly_connected_components(graph_CO),key=len, reverse=True)])


# In[82]:


Counter([len(c) for c in sorted(nx.strongly_connected_components(graph_CO),key=len, reverse=True)])


# In[83]:


max_scc_CO = max(nx.weakly_connected_components(graph_CO),key=len)


# In[84]:


graph_sc_CO = graph_CO.subgraph(max_scc_CO).copy()


# In[85]:


print(nx.info(graph_sc_CO))


# In[86]:


attr_sc_CO = nx.get_edge_attributes(graph_sc_CO, "exp")
attr_CO = nx.get_edge_attributes(graph_CO, "exp")
sum(attr_sc_CO.values())


# In[88]:


sum(map(abs,attr_sc_CO.values()))/sum(map(abs,attr_CO.values()))


# ### Credit

# In[89]:


print(nx.info(graph_CR))


# In[90]:


nx.is_weakly_connected(graph_CR)


# In[91]:


nx.number_weakly_connected_components(graph_CR)


# In[92]:


Counter([len(c) for c in sorted(nx.weakly_connected_components(graph_CR),key=len, reverse=True)])


# In[93]:


Counter([len(c) for c in sorted(nx.strongly_connected_components(graph_CR),key=len, reverse=True)])


# In[94]:


max_scc_CR = max(nx.weakly_connected_components(graph_CR),key=len)


# In[95]:


graph_sc_CR = graph_CR.subgraph(max_scc_CR).copy()


# In[189]:


print(nx.info(graph_sc_CR))


# In[107]:


attr_sc_CR = nx.get_edge_attributes(graph_sc_CR, "exp")
attr_CR = nx.get_edge_attributes(graph_CR, "exp")
sum_CR = sum(attr_sc_CR.values())


# In[98]:


sum(map(abs,attr_sc_CR.values()))/sum(map(abs,attr_CR.values()))


# ### Interest Rate

# In[99]:


print(nx.info(graph_IR))


# In[100]:


nx.is_weakly_connected(graph_IR)


# In[101]:


nx.number_weakly_connected_components(graph_IR)


# In[102]:


Counter([len(c) for c in sorted(nx.weakly_connected_components(graph_IR),key=len, reverse=True)])


# In[103]:


Counter([len(c) for c in sorted(nx.strongly_connected_components(graph_IR),key=len, reverse=True)])


# In[104]:


max_scc_IR = max(nx.weakly_connected_components(graph_IR),key=len)


# In[105]:


graph_sc_IR = graph_IR.subgraph(max_scc_IR).copy()


# In[106]:


print(nx.info(graph_sc_IR))


# In[108]:


attr_sc_IR = nx.get_edge_attributes(graph_sc_IR, "exp")
attr_IR = nx.get_edge_attributes(graph_IR, "exp")
sum_IR = sum(attr_sc_IR.values())


# In[109]:


sum(map(abs,attr_sc_IR.values()))/sum(map(abs,attr_IR.values()))


# ### Equity

# In[110]:


print(nx.info(graph_EQ))


# In[111]:


nx.is_weakly_connected(graph_EQ)


# In[112]:


nx.number_weakly_connected_components(graph_EQ)


# In[113]:


Counter([len(c) for c in sorted(nx.weakly_connected_components(graph_EQ),key=len, reverse=True)])


# In[114]:


Counter([len(c) for c in sorted(nx.strongly_connected_components(graph_EQ),key=len, reverse=True)])


# In[115]:


max_scc_EQ = max(nx.weakly_connected_components(graph_EQ),key=len)


# In[116]:


graph_sc_EQ = graph_EQ.subgraph(max_scc_EQ).copy()


# In[117]:


print(nx.info(graph_sc_EQ))


# In[118]:


attr_sc_EQ = nx.get_edge_attributes(graph_sc_EQ, "exp")
attr_EQ = nx.get_edge_attributes(graph_EQ, "exp")
sum_EQ = sum(attr_sc_EQ.values())


# In[119]:


sum(map(abs,attr_sc_EQ.values()))/sum(map(abs,attr_EQ.values()))


# #### drop Commodity layer before subsampling WCC

# In[ ]:


data_no_CO = data.drop(np.where(data['class'] == 'CO')[0])
graph_d = nx.DiGraph()
for cp_tuple in zip(data_no_CO['CP1'], data_no_CO['CP2']):
    sum_exp = sum(data_no_CO.loc[(data_no_CO['CP1'] == cp_tuple[0]) & (data_no_CO['CP2'] == cp_tuple[1]), 'exp'])
    #print(cp_tuple[0], cp_tuple[1], sum_exp)
    graph_d.add_edge(cp_tuple[0], cp_tuple[1], exp = sum_exp)


# In[120]:


max_scc = max(nx.weakly_connected_components(graph_d),key=len)


# In[121]:


graph_sc = graph_d.subgraph(max_scc).copy()


# In[124]:


print(nx.info(graph_sc))


# In[126]:


attr_sc = nx.get_edge_attributes(graph_sc, "exp")
sum_total = sum(attr_sc.values())



# #### Combine the asset class subgraphs to multigraph.

# In[ ]:


layer_graphs_directed = {'CU': graph_sc_CU, 'CR': graph_sc_CR, 'IR':graph_sc_IR, 'EQ':graph_sc_EQ}


# In[ ]:


layer_graphs_mp = {}
for key, graph in layer_graphs_directed.items():
    graph_adj = graph.copy()
    # generate a set of nodes that are in the set of nodes of the whole graph but not in the respective layer graph
    layer_graphs_mp.update({key:graph_adj})


# ### Function to decompose the network into its components

# In[ ]:


def in_out_components(graph):
    wc_nodes = set(max(nx.weakly_connected_components(graph),key=len))
    sc_nodes = set(max(nx.strongly_connected_components(graph),key=len))
    in_out = list(wc_nodes - sc_nodes)
    in_component = []
    out_component = []
    tendrils = []
    for node in in_out:
        pred = set(graph.predecessors(node))
        succ = set(graph.successors(node))
        if len(pred & sc_nodes) > 0:
            out_component.append(node)
        elif len(succ & sc_nodes) > 0:
            in_component.append(node)
        else:
            tendrils.append(node)
            
    g_in_out = graph.subgraph(in_out)
    g_sc = graph.subgraph(list(sc_nodes))
    g_in = graph.subgraph(in_component)
    g_out = graph.subgraph(out_component)
    g_tend = graph.subgraph(tendrils)
    #val_total = sum(nx.get_edge_attributes(graph, 'exp').values())
    val_total = sum(nx.get_edge_attributes(graph, 'exp').values())
    share_sc = sum(nx.get_edge_attributes(g_sc, 'exp').values())/val_total
    share_in = sum(nx.get_edge_attributes(g_in, 'exp').values())/val_total
    share_out = sum(nx.get_edge_attributes(g_out, 'exp').values())/val_total
    share_tend = sum(nx.get_edge_attributes(g_tend, 'exp').values())/val_total
    share_in_out = sum(nx.get_edge_attributes(g_in_out, 'exp').values())/val_total
    
    return {"G_in": {'nodes':len(in_component), 'share': share_in},
            "G_out": {'nodes': len(out_component), 'share': share_out},
            "tendrils": {'nodes': len(tendrils), 'share': share_tend},
            'share_SC': share_sc,
           'share_in_out': share_in_out}


# In[99]:


in_out_components(graph_d)


# In[100]:


in_out_components(graph_CO)


# In[101]:


in_out_components(graph_CU)


# In[102]:


in_out_components(graph_CR)


# In[103]:


in_out_components(graph_EQ)


# In[104]:


in_out_components(graph_IR)


# # Size and Density

# ##### Density as number of edges relative to fully connected graph
# 

# In[34]:


nx.density(graph_sc)


# ### By Asset Classes

# In[35]:


asset_classes = ['Whole Network', 'Currency', 'Credit', 'Interest Rate', 'Equity']


# In[36]:


nx.density(graph_sc_CU)


# In[37]:


nx.density(graph_sc_CR)


# In[38]:


nx.density(graph_sc_IR)


# In[39]:


nx.density(graph_sc_EQ)


# ## Layer Activity

# #### Get information about layer activity of nodes, How many nodes are active in 1,2,..,5 layers

# In[1057]:


layer_activity = collections.defaultdict(list)


# collects for each node the layers it is active in.

# In[1058]:



for n, nbrs in graph_mp.adj.items():
    for nbr, eattr in nbrs.items():
        for key, attr in eattr.items():
            layer_activity[n].append(key)
            layer_activity[nbr].append(key)


# generate the set of active classes for each node

# In[1059]:


layer_activity_set = collections.defaultdict(list)
for key, value in layer_activity.items():
    for asset_class in list(set(value)):
        layer_activity_set[key].append(asset_class)


# counts the number of layers a node is active in

# In[46]:


layer_activity_count = Counter([len(val) for key, val in layer_activity_set.items()])
layer_activity_count


# In[47]:


{key: val/len(graph_mp.nodes()) for key, val in layer_activity_count.items()}


# # Distance and Diameter

# ## Average Shortest Path Length

# ### Whole Network

# In[144]:


asp_whole= nx.average_shortest_path_length(graph_sc.to_undirected())
asp_whole


# ### By Asset Classes

# #### Currency

# In[145]:


asp_CU = nx.average_shortest_path_length(graph_sc_CU.to_undirected())
asp_CU


# #### Credit

# In[146]:


asp_CR = nx.average_shortest_path_length(graph_sc_CR.to_undirected())
asp_CR


# #### Interest Rate

# In[143]:


asp_IR = nx.average_shortest_path_length(graph_sc_IR.to_undirected())
asp_IR


# #### Equity

# In[147]:


asp_EQ = nx.average_shortest_path_length(graph_sc_EQ.to_undirected())
asp_EQ


# ### plot ASPL

# In[149]:


asp = [asp_whole, asp_CU, asp_CR, asp_IR, asp_EQ]


# In[1119]:


asset_classes = ['Aggregate', 'Currency', 'Credit', 'Interest Rate', 'Equity']


# In[1317]:


ax = sns.barplot(x = asp, y = asset_classes)
ax.set(xlabel = "Average Shortest Path Length")
for i, v in enumerate(asp):
    ax.text(v - 0.5, i + .1, str(round(v, 2)))
plt.show()


# ### Mass Distance Function

# In[195]:


def mass_distance_function(graph):
    mass_dict = {}
    graph_len = len(graph)
    pair_sp = nx.all_pairs_shortest_path_length(graph.to_undirected())
    path_lengths = []
    for source in pair_sp:
        for target, p_len in source[1].items():
            path_lengths.append(p_len)
    path_count = collections.Counter(path_lengths)
    sum_counts = 0
    for i in range(1,len(path_count)):
        sum_counts += path_count[i]
        if not i == 1:
            mass_dict.update({'M_{}'.format(i): sum_counts/(graph_len * (graph_len - 1))})
    return mass_dict
        
        
        
        


# In[197]:


mass_distance_function(graph_sc)


# In[196]:


mass_distance_function(graph_sc_CR)


# In[198]:


mass_distance_function(graph_sc_CU)


# In[199]:


mass_distance_function(graph_sc_EQ)


# In[200]:


mass_distance_function(graph_sc_IR)


# ## Eccentricity

# As the networks are not strongly connected, in order to avoid infinite(not existing) paths from some node to another, the graphs have to be transformed to an undirected in order to properly compute the mean eccentricity

# ### Whole Network

# In[159]:


ecc_whole = nx.eccentricity(graph_sc.to_undirected())
ecc_whole_mean = stat.mean(ecc_whole.values())
ecc_whole_mean


# ### By Asset Class

# #### Currency

# In[160]:


ecc_CU = nx.eccentricity(graph_sc_CU.to_undirected())
ecc_CU_mean = stat.mean(ecc_CU.values())
ecc_CU_mean


# #### Credit

# In[161]:


ecc_CR = nx.eccentricity(graph_sc_CR.to_undirected())
ecc_CR_mean = stat.mean(ecc_CR.values())
ecc_CR_mean


# #### Interest Rate

# In[162]:


ecc_IR = nx.eccentricity(graph_sc_IR.to_undirected())
ecc_IR_mean = stat.mean(ecc_IR.values())
ecc_IR_mean


# #### Equity

# In[163]:


ecc_EQ = nx.eccentricity(graph_sc_EQ.to_undirected())
ecc_EQ_mean = stat.mean(ecc_EQ.values())
ecc_EQ_mean


# ### Plot Eccentricity

# In[164]:


ecc = [ecc_whole_mean, ecc_CU_mean, ecc_CR_mean, ecc_IR_mean, ecc_EQ_mean]


# In[1318]:


ax = sns.barplot(x = ecc, y = asset_classes)
ax.set(xlabel = "Mean Eccentricity")
for i, v in enumerate(ecc):
    ax.text(v - 1, i + .1, str(round(v, 2))) 
plt.show()


# ## Diameter

# ### Whole Network

# In[167]:


dia_whole = nx.diameter(graph_sc, ecc_whole)
dia_whole


# ### By Asset Class

# #### Currency

# In[168]:


dia_CU = nx.diameter(graph_sc_CU, ecc_CU)
dia_CU


# #### Credit

# In[170]:


dia_CR = nx.diameter(graph_sc_CR, ecc_CR)
dia_CR


# #### Interest Rate

# In[171]:


dia_IR = nx.diameter(graph_sc_IR, ecc_IR)
dia_IR


# #### Equity

# In[172]:


dia_EQ = nx.diameter(graph_sc_EQ, ecc_EQ)
dia_EQ


# ### plot diameter

# In[173]:


dia = [dia_whole, dia_CU, dia_CR, dia_IR, dia_EQ]


# In[1320]:


ax = sns.barplot(x = dia, y = asset_classes)
ax.set(xlabel = "Diameter")
for i, v in enumerate(dia):
    ax.text(v - 0.5, i + .1, str(round(v, 2)))
plt.show()


# # Degree Distribution

# ## Degree Histogram

# ### Whole Network

# In[1329]:


#dg = graph_sc.in_degree()
dg = graph_sc.out_degree()


# sorted list of degree values

# In[1330]:


dg_seq = sorted([val for key, val in dg], reverse = True)


# count the frquencies of degree occurences

# In[1331]:


dg_count = collections.Counter(dg_seq)


# distribute count and degree into distinct objects to plot in the next step

# In[1333]:


deg,cnt = zip(*dg_count.items())


# In[1334]:


cnt_normalized = [count/len(graph_sc) for count in cnt]


# Plot degree distribution

# In[1335]:


fig,ax = plt.subplots()
#ax.axis([min(deg), max(deg), min(cnt_normalized), 1])
ax.set_xscale('symlog')
ax.set_yscale('log')
plt.scatter(deg,cnt_normalized ,color = "royalblue", s = 20)
#plt.plot(deg,cnt)

#ax.set_yticklabels([str(10**-3),10**-2,10**-1, 1])
ax.set_yticks([0.001, 0.01, 0.1, 1])
ax.set_yticklabels(['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1'])

ax.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter())
#ax.get_yaxis().set_major_formatter(matplotlib.ticker.LogFormatter())

#plt.title("Degree Distribution")
plt.ylabel("Node Frequency")
plt.xlabel("Degree")


plt.show()
("")


# ## Degree Correlation and Neighbor Successor Degree Plots

# ### Whole Network

# ### Degree Correlation

# In[164]:


nx.degree_pearson_correlation_coefficient(graph_sc)


# ### N-S Degree Plots

# ##### Function to compute the average successor neighbour degree for each node in the graph

# In[422]:


nnd = nx.average_neighbor_degree(graph_sc, source = 'out', target = 'out')


# In[423]:


nd = graph_sc.out_degree()


# In[1336]:


plt.scatter(list(dict(nd).values()),list(nnd.values()), s = 5, c = 'royalblue')
#plt.title("Average nearest successor out-degree as a function of out-degree.")
plt.ylabel("Successors Out-Degree")
plt.xlabel("Node out-degree")
plt.show()


# ## Clustering

# ### Whole Network

# In[196]:


graph_sc_nd = graph_sc.to_undirected()


# In[1102]:


print(nx.info(graph_sc_nd))


# In[1099]:


nx.average_clustering(graph_sc_nd)


# ##### Clustering of individual nodes

# In[1100]:


cluster = nx.clustering(graph_sc_nd)


# In[1101]:


#cluster


# In[1103]:


cluster_node = [val for key,val in cluster.items()]


# As most nodes with low degrees will have clustering coeffs of 0 or 1, it might make sense to filter them out before plottign to not mess up scaling

# In[1104]:


cluster_node_nonzero = list(filter(lambda x: x != 0, cluster_node))


# In[1097]:


cluster_node_nonzero = list(filter(lambda x: ((x != 0) & (x != 1)), cluster_node))


# In[1105]:


len(cluster_node_nonzero)/len(cluster_node)


# In[205]:


n, bins, patches = plt.hist(cluster_node_nonzero, bins = 50, edgecolor = "black", linewidth = 0.5)
plt.show()


# ##### clustering of nodes with degree larger than 2

# In[1114]:


def degree_cond(x):
    deg = graph_sc_nd.degree(x)
    return deg > 1


# In[1115]:


deg_nodes = list(filter(degree_cond, list(graph_sc_nd.nodes())))


# In[1116]:


len(deg_nodes)


# In[1117]:


nx.average_clustering(graph_sc_nd, deg_nodes)


# In[210]:


cluster = nx.clustering(graph_sc_nd, deg_nodes)


# In[211]:


cluster_node = [val for key,val in cluster.items()]


# In[215]:


n, bins, patches = plt.hist(cluster_node, bins = 50, edgecolor = "black", linewidth = 0.5, log = True)
plt.show()


# # Strength Degree Distribution

# #### Function to generate the average strength over different degrees

# In[534]:


def strength_degree_average(graph):
    strength = dict(graph.degree(weight = 'exp'))
    degree = dict(graph.degree())                
    str_degree = collections.defaultdict(list)
    for node_s, s in strength.items():
        for node_d, d in degree.items():
            if node_s == node_d:
                str_degree[d].append(s)
    for deg, str_list in str_degree.items():
        avg_str = sum(str_list)/len(str_list)            
        str_degree[deg] = avg_str
    str_degree.pop(0,None)
    return str_degree


# In[535]:


s_d_agg = strength_degree_average(graph_sc)
s_d_CU =  strength_degree_average(graph_sc_CU)
s_d_CR =  strength_degree_average(graph_sc_CR)
s_d_EQ =  strength_degree_average(graph_sc_EQ)
s_d_IR =  strength_degree_average(graph_sc_IR)


# In[536]:


s_d_all = {'Aggregate': s_d_agg, 'Currency': s_d_CU, 'Credit': s_d_CR, 'Equity': s_d_EQ, 'Interest Rate': s_d_IR}


# #### Plot the average strength degree distribution

# In[1337]:


cdict = {1: 'red', 2: 'royalblue', 3: 'green', 4: 'orange', 5:'purple'}
#cdict = {1: 'red', 2: 'lightsteelblueblue', 3: 'cornflowerblue', 4: 'blue', 5:'navy'}

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel('Strength')
ax.set_xlabel('Degree')
for i,layer in enumerate(s_d_all.keys()):
    if layer == "Aggregate":
        markup = 10
    else:
        markup = 0
    x = np.array(list(s_d_all[layer].keys()))
    y = np.array(list(s_d_all[layer].values()))
    m, b = np.polyfit(np.log(x), np.log(y), 1)
        
    ax.scatter(x, y, c = cdict[i+1], label = layer, s = 15 + markup)
    ax.plot(x, np.exp(b)*(x**m), c = cdict[i+1], linewidth = 1)
    
    
ax.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter())

ax.legend()


plt.show()


# #### Get the corresponding regression coefficients of the relationship of avg strength and degree

# In[555]:


for i,layer in enumerate(s_d_all.keys()):
    
    x = np.log(np.array(list(s_d_all[layer].keys())))
    y = np.log(np.array(list(s_d_all[layer].values())))
    m, b = np.polyfit(x, y, 1)
    x = sm.add_constant(x)
    regression = sm.OLS(y,x).fit()
    
    print(layer)
    print('---------')
    print('slope:', m)
    print('intercept:', b)
    print(regression.summary())


# ### Layer Similarity

# In[562]:


def get_jaccard_coefficients_undirected(G, H, common_only = True):
    scores = []
    for v in G:
        if v in H:
            n = set(G[v]) # neighbors of v in G
            m = set(H[v]) # neighbors of v in H
            length_intersection = len(n & m)
            #length_union = len(n|m)
            length_union = len(n) + len(m) - length_intersection
            
            if length_union == 0:
                score = 0
               
            else:
                score = float(length_intersection) / length_union
            scores.append(score)
        else:
            if common_only:           
                continue
            else:
                scores.append(0)
    similarity = stat.mean(scores)

    return similarity


# In[563]:


def jaccard_similarity_by_layer(graph_dict,  common_only = True):
    jaccard_similarity = dict()
    for layer_a, layer_b in itertools.combinations(graph_dict, 2):
    
        graph_a = graph_dict[layer_a]
        
        graph_b = graph_dict[layer_b]
        
        similarity = get_jaccard_coefficients_undirected(graph_a, graph_b, common_only)
        label = layer_a + '-' + layer_b
        jaccard_similarity.update({label: similarity})
    return jaccard_similarity


layer_graphs_undirected = {'CU': graph_sc_CU.to_undirected(), 'CR': graph_sc_CR.to_undirected(), 'IR':graph_sc_IR.to_undirected(), 'EQ':graph_sc_EQ.to_undirected()}

# In[566]:


jaccard_similarity_by_layer(layer_graphs_undirected, common_only = True)


# In[567]:


jaccard_similarity_by_layer(layer_graphs_undirected, common_only = False)


# # Kendalls Tau Degree Correlation

# #### first step is to get the ordered (by nodes) degree of nodes in each layer

# In[569]:


layer_graphs_directed = {'CU': graph_sc_CU, 'CR': graph_sc_CR, 'IR':graph_sc_IR, 'EQ':graph_sc_EQ}


# In[570]:


all_nodes = list(set(graph_d.nodes()))


# ##### generate dictionary to store the different layer degrees to ensure comparison of the right values and define the default value for nodes not in the respective layers.

# In[572]:


layer_degrees = dict()
for node in all_nodes:
    layer_degrees.update({node: {'CU':0, 'CR':0, 'IR':0, 'EQ':0}})


# ##### Now iterate over the different asset layers, compute the degree and sort the values into the layer degree dictionary

# In[573]:


for key, graph in layer_graphs_directed.items():
    degrees = graph.degree()
    for node, degree in degrees:
        layer_degrees[node][key] = degree
    
    


# ##### Now iterate over each node and append the degree for the respective layer to the list of that layer...should ensure right ordering

# In[575]:


layer_degree_list = collections.defaultdict(list)
for node, degree_vals in layer_degrees.items():
    for layer, degree in degree_vals.items():
        layer_degree_list[layer].append(degree)


# In[577]:


kendall_tau_scores = dict()
for layer_a, layer_b in itertools.combinations(layer_graphs_directed, 2):
    deg_list_a = layer_degree_list[layer_a]
    deg_list_b = layer_degree_list[layer_b]
    tau, p_value = scipy.stats.kendalltau(deg_list_a, deg_list_b)
    label = layer_a + '-' + layer_b
    kendall_tau_scores.update({label:{'tau':tau, 'p_value':p_value}})


# # Multiplex Centrality

# #### generate 'true' multiplex graph, i.e. have the same set of nodes only with potentially different edges in the different layers

# In[579]:


layer_graphs_mp_adj = {}
nodes_mp_total = set(graph_sc)
for key, graph in layer_graphs_directed.items():
    graph_adj = graph.copy()
    # generate a set of nodes that are in the set of nodes of the whole graph but not in the respective layer graph
    nodes_add = set(graph_sc) - set(graph_adj)
    graph_adj.add_nodes_from(nodes_add)
    layer_graphs_mp_adj.update({key:graph_adj})


# ##### convert the graphs to numpy arrays in order to allow matrix calculations <br>
# first however, it is important to set a common ordering so that the elements of the matrices correspond to eachother i.e. the same node. Same would make sense for the layers, as their order also is not fixed when drawn from the dictionary.
# 

# In[580]:


node_order = list(graph_sc.nodes())
layer_order = ['CU', 'CR', 'EQ', 'IR']


# In[581]:


layer_adj_matrices = []
for layer in layer_order:
    matrix = nx.to_numpy_array(layer_graphs_mp_adj[layer], weight = 'exp', nodelist = node_order)
    layer_adj_matrices.append(matrix)


# ## MultiRank

# In[582]:


def MultiRank(A, M, N, alpha = 0.85, gamma = 1, s = -1, a = 1, quadratic_error = 0.0001):

################################################

# Python implementation of the original Matlab code as takes from:
    
# https://github.com/ginestrab/MultiRank/blob/master/MultiRank_Nodes_Layers.m
    
    # The following program takes in input:



# 1. a cell A of M sparse adjacency matrices (ordered layers of the multiplex network)
#    ex. A{1} is a N times N adjacency matrix of layer 1
#    we adopt the convention for directed network A{1}(i,j)=1 if node i
#    points to node j in layer 1

# 2. M the number of layers of the multiplex network

# 3. N the number of nodes of the multiplex networks

# 4. alpha the teleportation parameter usually set at 0.85

# 5. gamma parameter of the MultiRank gamma>0.

# 6. s=1,-1 parameter of the MultiRank s=-1 elite layers s=1 popular layers

# 7. a=0,1 parameter of the MultiRank a=1 centrality of the layers not
#    normalized with the total number of links, a=0 centrality of the layers
#   normalized with the total number of links.


# It produces as an output:


# 1. vector of  centrality of each node  
#    x(i) centrality of node i 

# 2. vector z centrality (influence) of each layer 
#    z(m) influence of layerm 

##################################################
    
    for n in range(M):
        A[n] = A[n].T
    
    B_in = np.empty((M,N))
    for n in range(M):
        for i in range(N):
            B_in[n,i] = sum(A[n][i, :])
            
    z = np.ones((M))    
    G = np.zeros((N,N))
    
    for n in range(M):
        G = G + A[n] * z[n]
        
    K = G.sum(axis = 0) + (G.sum(axis = 0) == 0)
    K = np.ones((1,N))/K
    K = scipy.sparse.diags(K[0],shape = (N,N)).toarray()
    
    x_0 = ((G.sum(axis = 0) > 0) + (G.sum(axis = 1) > 0)) > 0
    x_0 = x_0/np.count_nonzero(x_0)
    
    l = G.sum(axis = 0) > 0
    jump = alpha*l
    
    x = x_0.copy()
    
    x = G @ K @ (x*jump) + (((1-jump)*x).sum(axis = 0) * x_0)
    
    z = B_in.sum(axis = 1)**a*((B_in @ (x + (x == 0)) ** (s*gamma))/(B_in.sum(axis = 1) + (B_in.sum(axis = 1) == 0)))**(s)
    z = z/sum(z)
    
    x_last = np.ones((N,1)) * math.inf
    
    while np.linalg.norm(x - x_last) > (quadratic_error * np.linalg.norm(x)):
        x_last = x.copy()
        G = np.zeros((N,N))

        for n in range(M):
            G = G + A[n] * z[n]

        K = G.sum(axis = 0) + (G.sum(axis = 0) == 0)
        K = np.ones((1,N))/K
        K = scipy.sparse.diags(K[0],shape = (N,N)).toarray()

        x_0 = ((G.sum(axis = 0) > 0) + (G.sum(axis = 1) > 0)) > 0
        x_0 = x_0/np.count_nonzero(x_0)

        l = G.sum(axis = 0) > 0
        jump = alpha*l

        x = G @ K @ (x*jump) + (((1-jump)*x).sum(axis = 0) * x_0)

        z = B_in.sum(axis = 1)**a*((B_in @ (x + (x == 0)) ** (s*gamma))/(B_in.sum(axis = 1) + (B_in.sum(axis = 1) == 0)))**(s)
        z = z/sum(z)
        
    return {'centralities': x, 'layer_influences': z}


# In[583]:


n_nodes = len(layer_adj_matrices[1])


# ##### default with s = -1 giving more influence to layers with few highly connected nodes

# In[584]:


start = time.time()
multi_rank_results_minus = MultiRank(layer_adj_matrices, 4, n_nodes)
end = time.time()
end - start


# ##### alternatively set s = 1 for more influence of layers with more highly central nodes

# In[585]:


start = time.time()
multi_rank_results = MultiRank(layer_adj_matrices, 4, n_nodes, s = 1)
end = time.time()
end - start


# In[631]:


MRank_minus = multi_rank_results_minus["centralities"]


# In[632]:


MRank = multi_rank_results["centralities"]


# #### obtain the ranking of values

# In[626]:


I = MRank.argsort()
MRank_ranks = np.empty(len(I))
for i in range(len(I)):
    MRank_ranks[i] = np.where(I == i)[0]


# In[1065]:


I = MRank_minus.argsort()
MRank_ranks_minus = np.empty(len(I))
for i in range(len(I)):
    MRank_ranks_minus[i] = np.where(I == i)[0]


# ### Plot layer influences

# In[812]:


influences = multi_rank_results["layer_influences"]


# In[1338]:


ax = sns.barplot(x =influences, y = layer_order)
ax.set(xlabel = "Layer Influence",
      ylabel = 'Asset Classes')
for i, v in enumerate(influences):
    if i == 3:
        ax.text(v - 0.8, i + .1, str(round(v, 2)))
    else:
        ax.text(v + 0.05, i + .1, str(round(v, 2)))

plt.show()


# ### MultiRank over Gamma values in interval [0.1;3]

# In[586]:


def MultiRank_Gamma(A, M, N, alpha = 0.85, s = 1, a = 1):
    
    X = np.empty((N, 30))
    Z = np.empty((M, 30))
    XR = np.empty((N, 30))
    ZR = np.empty((M, 30))
    
    for ig in range(0,30, 1):
        if ig == 30: # neccessary for simultaneous iteration of 0:29 and 1:30
            break
        gamma = (ig + 1)*0.1

        rank = MultiRank(A, M, N, s = s, gamma = gamma)
        
        x = rank['centralities']
        z = rank['layer_influences']
        
        X[:, ig] = x
        
        I = x.argsort()
        
        Z[:, ig] = z
        
        I2 = z.argsort()
        
        for i in range(N):
            XR[i, ig] = np.where(I == i)[0]
            
        for m in range(M):
            ZR[m, ig] = np.where(I2 == m)[0]
            
    return  {'X': X,'Z': Z,'XR': XR,'ZR': ZR}


# In[730]:


start = time.time()
MRank_gamma = MultiRank_Gamma(layer_adj_matrices, 4, n_nodes)
end = time.time()
end - start


# ### Get the average ranking over all values of gamma. <br>
# for each node, build the sum of ranks over gamma values and divide by 30 to get the average. Then rank the avg rankings again to obtain the aggregate ranking

# In[738]:


avg_over_gamma = collections.Counter()
for i in range(30):
    labeled_ranks = dict(list(zip(node_order, MRank_gamma['X'][:,i])))    
    avg_over_gamma.update(labeled_ranks)
    
for key in avg_over_gamma:
    avg_over_gamma[key] /= 30


# In[741]:


__, gamma_vals_avg = map(np.array, zip(*list(avg_over_gamma.items())))


# In[734]:


I = gamma_vals_avg.argsort()
ranks_gamma_avg = np.empty(len(I))
for i in range(len(I)):
    ranks_gamma_avg[i] = np.where(I == i)[0]


# In[735]:


ranks_gamma_avg


# ## PageRank for Aggregated Network

# ##### computes the PageRank from the aggregate network

# In[658]:


PRank_agg_results = nx.pagerank_numpy(graph_sc, weight = 'exp')


# In[659]:


PRanks_agg_nodes, Prank_agg_vals = map(np.array, zip(*PRank_agg_results.items()))


# check if node order is indeed the same

# In[661]:


any(PRanks_agg_nodes != np.array(node_order))


# #### obtain rankings

# In[710]:


I = Prank_agg_vals.argsort()
PRank_agg_ranks = np.empty(len(I))
for i in range(len(I)):
    PRank_agg_ranks[i] = np.where(I == i)[0][0]


# ## PageRank for all asset classes seperately

# iterate over the layer graphs stored in dict, compute PageRanks for each and save results per layer in dictionary

# In[688]:


PRank_by_layer = {}
for key, graph in layer_graphs_mp_adj.items():
    results = nx.pagerank_numpy(graph, weight = 'exp')
    PRank_by_layer.update({key: results})


# ### Generate simple degree centrality 

# In[588]:


degree_centrality = nx.degree(graph_sc, weight = 'exp')


# In[589]:


__, degree_centr_vals = map(np.array, zip(*list(dict(degree_centrality).items())))


# In[590]:


I = np.array(degree_centr_vals).argsort()
degree_ranks = np.empty(len(I))
for i in range(len(I)):
    degree_ranks[i] = np.where(I == i)[0]


# ## Plot Rank against degree, weighted degree and maybe layer_activity

# In[1069]:


dg = graph_sc.degree()


# In[1070]:


degree = list(dict(dg).items())


# In[1071]:


__, degree = map(np.array, zip(*degree))


# In[1073]:


degree_weighted = dict(graph_sc.degree(weight = 'exp'))


# In[1074]:


degree_weighted = list(degree_weighted.items())


# In[1075]:


__, degree_weighted = map(np.array, zip(*degree_weighted))


# In[1078]:


layer_activity = dict((key, len(val)) for key, val in layer_activity_set.items() if key in node_order)


# sort values according to common node order

# In[1079]:


layer_activity =  list(sorted(list(layer_activity.items()), key = lambda x: node_order.index(x[0])))


# In[1080]:


__, layer_activity = map(np.array, zip(*layer_activity))


# #### MultiRank

# In[1339]:



cdict = {1: 'plum', 2: 'yellow', 3: 'cornflowerblue', 4: 'green', 5:'red'}
#cdict = {1: 'red', 2: 'lightsteelblueblue', 3: 'cornflowerblue', 4: 'blue', 5:'navy'}

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_ylabel('Strength')
ax.set_xlabel('Degree')
for g in np.unique(layer_activity):
    ix = np.where(layer_activity == g)[0]
    ax.scatter(degree[ix], degree_weighted[ix], c = cdict[g], label = g, s = MRank[ix] * 10000, alpha = 0.7)
ax.legend(markerscale = 1)
lgnd = ax.legend(markerscale = 1)
for handle in lgnd.legendHandles:
    handle.set_sizes([100.0])


plt.show()


# #### MultiRank s=-1

# In[1341]:


cdict = {1: 'plum', 2: 'yellow', 3: 'cornflowerblue', 4: 'green', 5:'red'}
#cdict = {1: 'red', 2: 'lightsteelblueblue', 3: 'cornflowerblue', 4: 'blue', 5:'navy'}

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_ylabel('Strength')
ax.set_xlabel('Degree')
for g in np.unique(layer_activity):
    ix = np.where(layer_activity == g)[0]
    ax.scatter(degree[ix], degree_weighted[ix], c = cdict[g], label = g, s = MRank_minus[ix] * 10000, alpha = 0.7)
ax.legend(markerscale = 1)
lgnd = ax.legend(markerscale = 1)
for handle in lgnd.legendHandles:
    handle.set_sizes([100.0])

plt.show()


# #### MultiRank Gamma

# In[3]:


cdict = {1: 'plum', 2: 'yellow', 3: 'cornflowerblue', 4: 'green', 5:'red'}
#cdict = {1: 'red', 2: 'lightsteelblueblue', 3: 'cornflowerblue', 4: 'blue', 5:'navy'}

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_ylabel('Strength')
ax.set_xlabel('Degree')
for g in np.unique(layer_activity):
    ix = np.where(layer_activity == g)[0]
    ax.scatter(degree[ix], degree_weighted[ix], c = cdict[g], label = g, s = gamma_vals_avg[ix] * 10000, alpha = 0.7)
ax.legend(markerscale = 1)
lgnd = ax.legend(markerscale = 1)
for handle in lgnd.legendHandles:
    handle.set_sizes([100.0])

plt.show()


# # Compare Multiplex centrality with simple aggregate centrality

# #### Generate betweenness centrality for aggregate and individual layers

# ##### By layer

# In[755]:


Betweenness_by_layer = {}
for key, graph in layer_graphs_mp_adj.items():
    results = nx.betweenness_centrality(graph)
    Betweenness_by_layer.update({key: results})


# In[756]:


Betweenness_by_layer_top = {}
for key in Betweenness_by_layer:
    ordered_btw = list(sorted(Betweenness_by_layer[key].items(), key = lambda x: x[1], reverse = True))
    nodes, scores = map(np.array, zip(*ordered_btw[:65]))
    Betweenness_by_layer_top.update({key: nodes})
        


# ##### aggregate

# In[747]:


agg_betweenness = nx.betweenness_centrality(graph_sc)


# In[748]:



ordered_btw = list(sorted(agg_betweenness.items(), key = lambda x: x[1], reverse = True))
Betweenness_agg_top, __ = map(np.array, zip(*ordered_btw[:65]))


# In[750]:


btw_nodes, betweenness_centr_vals = map(np.array, zip(*agg_betweenness.items()))


# generate ranks for subsequent plotting

# In[674]:


I = np.array(betweenness_centr_vals).argsort()
betweenness_ranks = np.empty(len(I))
for i in range(len(I)):
    betweenness_ranks[i] = np.where(I == i)[0]


# ### Degree Centrality

# #### aggregate

# In[751]:


degree_agg = nx.degree(graph_sc, weight = 'exp')


# In[752]:


ordered_deg = list(sorted(dict(degree_agg).items(), key = lambda x: x[1], reverse = True))
Degree_agg_top, __ = map(np.array, zip(*ordered_deg[:65]))


# #### By layer

# In[757]:


Degree_by_layer = {}
for key, graph in layer_graphs_mp_adj.items():
    results = nx.degree(graph, weight = 'exp')
    Degree_by_layer.update({key: results})


# In[758]:


Degree_by_layer_top = {}
for key in Degree_by_layer:
    ordered_deg = list(sorted(dict(Degree_by_layer[key]).items(), key = lambda x: x[1], reverse = True))
    nodes, scores = map(np.array, zip(*ordered_deg[:65]))
    Degree_by_layer_top.update({key: nodes})


# ### PageRank

# ##### aggregate

# In[759]:


ordered_PRank_agg = list(sorted(PRank_agg_results.items(), key = lambda x: x[1], reverse = True))
PRank_agg_top, __ = map(np.array, zip(*ordered_PRank_agg[:65]))


# ##### By layer

# In[760]:


PRank_by_layer_top = {}
for key in PRank_by_layer:
    ordered_rank = list(sorted(PRank_by_layer[key].items(), key = lambda x: x[1], reverse = True))
    nodes, scores = map(np.array, zip(*ordered_rank[:65]))
    PRank_by_layer_top.update({key: nodes})


# #### MultiRank average over all gamma values

# In[744]:


gamma_w_labels = sorted(list(zip(node_order, ranks_gamma_avg)), key = lambda x: x[1], reverse = True)


# In[746]:


MRank_gamma_top, __ = map(np.array, zip(*gamma_w_labels[:65]))


# In[792]:


Mrank_w_labels = sorted(list(zip(node_order, MRank)), key = lambda x: x[1], reverse = True)


# In[793]:


MRank_top, __ = map(np.array, zip(*Mrank_w_labels[:65]))


# ## Compare the top nodes selected

# In[783]:


top_rankings_agg = {'Betweenness': Betweenness_agg_top, 'Degree': Degree_agg_top, 'PageRank': PRank_agg_top}
#top_rankings_agg = {'Betweenness': Betweenness_agg_top, 'Degree': Degree_agg_top, 'MRank_Gamma': MRank_gamma_top}

# In[784]:


top_rankings_layer = {'Betweenness': Betweenness_by_layer_top, 'Degree': Degree_by_layer_top, 'PageRank': PRank_by_layer_top}


# In[785]:


index = ['Layer', 'Betweenness', 'Degree', 'PageRank', 'Any']


# In[800]:


set_MRank = set(MRank_top)


# In[801]:


common_nodes_detected = pd.DataFrame(columns = index)


# In[803]:



common_nodes_agg = dict()
agg_nodes_all = []
for i,(centrality, top_nodes) in enumerate(top_rankings_agg.items()):
    agg_nodes_all.extend(list(top_nodes))
    common = set(top_nodes) & set_MRank
    share_common = len(common)/len(set_MRank)
    common_nodes_agg.update({centrality: share_common})
    if i == 2:
        common_all =  set(agg_nodes_all) & set_MRank
        share_all = len(common_all)/len(set_MRank)
        common_nodes_agg.update({'Layer': 'Aggregate', 'Any': share_all})
common_nodes_detected = common_nodes_detected.append(common_nodes_agg, ignore_index = True)


# In[804]:


for layer in layer_order:
    common_nodes_agg = dict()
    agg_nodes_all = []
    for i,(centrality, top_nodes) in enumerate(top_rankings_layer.items()):
        agg_nodes_all.extend(list(top_nodes[layer]))
        common = set(top_nodes[layer]) & set_MRank
        share_common = len(common)/len(set_MRank)
        common_nodes_agg.update({centrality: share_common})
        if i == 2:
            common_all =  set(agg_nodes_all) & set_MRank
            share_all = len(common_all)/len(set_MRank)
            common_nodes_agg.update({'Layer': layer, 'Any': share_all})
    common_nodes_detected = common_nodes_detected.append(common_nodes_agg, ignore_index = True)


# In[805]:


common_nodes_detected


# #### with gamma ranks avg

# In[779]:


common_nodes_detected


# 
# 
# ## Plot the rankings for easier assessment of correlation among different rankings algos

# #### get the rank values in equal order

# In[1086]:


MRank_gamma_ranks = ranks_gamma_avg
MRank_gamma_ranks


# In[1087]:


MRank_ranks


# In[715]:


MRank_ranks_minus


# In[713]:


PRank_agg_ranks


# In[392]:


betweenness_ranks



# ### plot them against eachother

# In[428]:


length = len(MRank_ranks)


# #### 1. MRank - Mrank_gamma

# In[1342]:


fig, ax = plt.subplots()
ax.scatter(x = MRank_ranks, y = MRank_gamma_ranks, s = 1, c = 'royalblue')

line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle = '--')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.set_xlabel('MultiRank Ranks')
ax.set_ylabel('MutliRank Ranks (Avg. over Gamma)')

plt.show()


# ## 2. MRank - Prank_agg

# In[1343]:


fig, ax = plt.subplots()
ax.scatter(x = MRank_ranks, y = PRank_agg_ranks, s = 1, c = 'royalblue')

line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle = '--')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.set_xlabel('MultiRank Ranks')
ax.set_ylabel('PageRank Ranks over agg. Network')

plt.show()


# ## 6. Mrank - degree

# In[1394]:


fig, ax = plt.subplots()
ax.scatter(x = MRank_ranks, y = degree_ranks, s = 1, c = 'royalblue')

line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle = '--')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.set_xlabel('MultiRank Ranks')
ax.set_ylabel('Degree Centrality Ranks')

plt.show()


# ## MultiRank - Betweenness

# In[1345]:


fig, ax = plt.subplots()
ax.scatter(x = MRank_ranks, y = betweenness_ranks, s = 1, c = 'royalblue')

line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle = '--')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.set_xlabel('MultiRank Ranks')
ax.set_ylabel('Betweenness Centrality Ranks')

plt.show()


# # Selection of most central nodes

# ### get list of nodes ordered by centrality ranks for each measure

# #### MRank

# In[821]:


MRank_nodes, __ = list(zip(*sorted(list(zip(node_order, MRank_ranks)), key = lambda x: x[1], reverse = True)))


# #### PRank_agg

# In[822]:


PRank_agg_nodes, __ = list(zip(*sorted(list(zip(node_order, PRank_agg_ranks)), key = lambda x: x[1], reverse = True)))


# #### Degree

# In[823]:


Degree_nodes, __ = list(zip(*sorted(list(zip(node_order, degree_ranks)), key = lambda x: x[1], reverse = True)))


# #### Betweenness

# In[824]:


Betweenness_nodes, __ = list(zip(*sorted(list(zip(node_order, betweenness_ranks)), key = lambda x: x[1], reverse = True)))


# In[959]:


def total_weight_threshold(graph, nodes, total = True):
    total_weight = sum(dict(graph.degree(weight = 'exp')).values())
    node_weight = sum(dict(graph.degree(nodes, weight = 'exp')).values())
    weight_share = node_weight/total_weight
    if total:
        threshold = 0.80
    else:
        threshold = 0.60
        
    if weight_share >= threshold:
        return True, weight_share
    else:
        return False, weight_share


# In[960]:


node_lists = [MRank_nodes, PRank_agg_nodes, Degree_nodes, Betweenness_nodes]
market_share_total = []
market_share_layers = []
first = True

for x in range(1, 1000):
    layer_share = []
    thresh_ind_layers = []
    top_nodes = []
    
    for nodes in node_lists:
        top_nodes.extend(nodes[:x])
        
    union_nodes = list(set(top_nodes))
    thresh_ind_total, weight_share = total_weight_threshold(graph_sc, union_nodes)
    
    
    for key in ['CR', 'CU', 'EQ', 'IR']:
        thresh_ind, market_share = total_weight_threshold(layer_graphs_mp_adj[key], union_nodes, total = False)
        thresh_ind_layers.append(thresh_ind)
        layer_share.append(market_share)
        
    thresh_ind_layers = all(thresh_ind_layers)
    thresh_ind_all = all([thresh_ind_layers, thresh_ind_total])
    market_share_layers.append((x, len(union_nodes), *layer_share, thresh_ind_layers, thresh_ind_all))
    market_share_total.append((x, len(union_nodes), weight_share, thresh_ind_total, thresh_ind_all))
    
    if all([thresh_ind_layers, thresh_ind_total, first]):
           most_important_nodes = union_nodes.copy()
           first = False


# In[965]:


market_share_total_df = pd.DataFrame(market_share_total, columns = ['x', 'n_nodes', 'share_total', 'thresh_total', 'thresh_all'])


# In[966]:


market_share_layer_df = pd.DataFrame(market_share_layers, columns = ['x', 'n_nodes', 'share_CR', 'share_CU', 'share_EQ', 'share_IR','thresh_layers', 'thresh_all'])


# In[957]:


market_share_total_df.head()


# In[1351]:


fig, ax = plt.subplots()
ax.plot(market_share_total_df['x'][:300], market_share_total_df['share_total'][:300], c = 'red', linestyle = '--', label = 'Aggregate')
ax.set_xlabel('X')
ax.set_ylabel('Share of Weighted Degree')

ax2 = ax.twinx()
ax2.scatter(market_share_total_df['x'][:300], market_share_total_df['n_nodes'][:300], c = 'blue', marker = 'x', s = 15, linewidth = 0.4, label = 'Number of Nodes')
ax2.set_ylabel('Number of Nodes')
leg = ax.legend(loc = (0.5, 0.125))
ax2.legend(loc = (0.5, 0.2))
ax2.grid(None)
#ax2.set_ylim(0, 500)
line = mlines.Line2D([0, 1], [0.77, 0.77], color='green', linestyle = '-')
line2 =  mlines.Line2D([0.18, 0.18], [0, 1], color='green', linestyle = '-')

transform = ax.transAxes

line.set_transform(transform)
line2.set_transform(transform)

ax.add_line(line)
ax.add_line(line2)

plt.show()


# The dip at the end indicates inclusion of nodes with high volume but obviously low centrality

# In[1361]:


fig, ax = plt.subplots()

ax.plot(market_share_layer_df['x'][:300],market_share_layer_df['share_CR'][:300], c = 'maroon', linestyle = 'dotted', label = 'Credit')
ax.plot(market_share_layer_df['x'][:300],market_share_layer_df['share_CU'][:300], c = 'red', linestyle = '-', label = 'Currency')
ax.plot(market_share_layer_df['x'][:300],market_share_layer_df['share_EQ'][:300], c = 'purple', linestyle = '-.', label = 'Equity')
ax.plot(market_share_layer_df['x'][:300],market_share_layer_df['share_IR'][:300], c = 'gold', linestyle = '--', label = 'Interest Rate')

ax.set_xlabel('X')
ax.set_ylabel('Share of Weighted Degree')
leg = ax.legend(loc = (0.55, 0.025))

ax2 = ax.twinx()
ax2.scatter(market_share_total_df['x'][:300], market_share_total_df['n_nodes'][:300], c = 'blue', marker = 'x', s = 15, linewidth = 0.4, label = 'Number of Nodes')
ax2.set_ylabel('Number of Nodes')
ax2.legend(loc = (0.55, 0.325))
ax2.grid(None)
line = mlines.Line2D([0, 1], [0.59, 0.59], color='green', linestyle = '-')
line2 =  mlines.Line2D([0.18, 0.18], [0, 1], color='green', linestyle = '-')

transform = ax.transAxes

line.set_transform(transform)
line2.set_transform(transform)

ax.add_line(line)
ax.add_line(line2)

plt.show()


# ### layer activity for top nodes

# In[1008]:


layer_activity = collections.defaultdict(list)
for n, nbrs in graph_mp.adj.items():
    if n in most_important_nodes:
        for nbr, eattr in nbrs.items():
            for key, attr in eattr.items():
                if not key == 'CO':
                    layer_activity[n].append(key)
                    if nbr in most_important_nodes:
                        layer_activity[nbr].append(key)
                else:
                    continue
                #layer_activity[nbr].append(key)
    else:
        continue
print(len(layer_activity)/len(most_important_nodes)  )          
layer_activity_set = collections.defaultdict(list)
for key, value in layer_activity.items():
    for asset_class in list(set(value)):
        layer_activity_set[key].append(asset_class)
        
layer_activity_count = Counter([len(val) for key, val in layer_activity_set.items()])
{key: val/len(most_important_nodes) for key, val in layer_activity_count.items()}


# In[1014]:


scc = max(nx.strongly_connected_components(graph_sc),key=len)


# In[1016]:


in_scc = [node for node in most_important_nodes if node in scc]


# In[1017]:


len(in_scc)/len(most_important_nodes)


# # Interconnectedness of most relevant nodes

# In[1036]:


length = len(most_important_nodes)


# In[1037]:


most_important_nodes_sorted = [i for i in Mrank_w_labels if i[0] in most_important_nodes]
most_important_nodes_sorted, __ = zip(*most_important_nodes_sorted)


# In[1038]:


graph_sc_nd = graph_sc.to_undirected()


# In[1039]:


def get_jaccard_coefficient_nodes(nodes, graph):
    
    A = nodes[0]
    B = nodes[1]
    common_nbr = list(nx.common_neighbors(graph, A, B))
    if len(common_nbr) > 0:
        min_common_weight = 0
        max_common_weight = 0
        for nbr in common_nbr:
            weight_A = graph[A][nbr]['exp']
            weight_B = graph[B][nbr]['exp']
            min_common_weight += min(weight_A, weight_B)
            max_common_weight += max(weight_A, weight_B)

        jaccard = min_common_weight/max_common_weight
        #print(min_common_weight)
        #print(max_common_weight)
        return jaccard, len(common_nbr)
    else:
        return 0, 0


# In[1384]:


jaccard_matrix = np.empty((len(most_important_nodes), len(most_important_nodes)))
common_nbr_matrix = np.empty((len(most_important_nodes), len(most_important_nodes)))
for id_1, id_2 in itertools.combinations(range(len(most_important_nodes)), 2):
    node_A = most_important_nodes_sorted[id_1]
    node_B = most_important_nodes_sorted[id_2]
    jaccard_score, n_common_nbr = get_jaccard_coefficient_nodes((node_A, node_B), graph_sc_nd)
    jaccard_matrix[[id_1, id_2], [id_2, id_1]] = jaccard_score
    common_nbr_matrix[[id_1, id_2], [id_2, id_1]] = n_common_nbr
    jaccard_matrix[[id_1,id_2],[id_1, id_2]] = 1
    common_nbr_matrix[id_1,id_1] = 0#len(graph_sc_nd[node_A])/10
    common_nbr_matrix[id_2,id_2] = 0#len(graph_sc_nd[node_B])/10


# ### For individual layers

# In[1374]:


jaccard_by_layer = {'CR': np.empty((length,length)), 'CU': np.empty((length,length)),
                   'EQ': np.empty((length,length)), 'IR': np.empty((length,length))}

common_nbrs_by_layer = {'CR': np.empty((length,length)), 'CU': np.empty((length,length)),
                   'EQ': np.empty((length,length)), 'IR': np.empty((length,length))}


# In[1385]:


for key, graph_dir in layer_graphs_mp_adj.items():
    graph = graph_dir.to_undirected()
    #graph = graph_dir
    for id_1, id_2 in itertools.combinations(range(len(most_important_nodes_sorted)), 2):
        node_A = most_important_nodes_sorted[id_1]
        node_B = most_important_nodes_sorted[id_2]
        
        jaccard_score, n_common_nbr = get_jaccard_coefficient_nodes((node_A, node_B), graph)
        
        jaccard_by_layer[key][[id_1, id_2], [id_2, id_1]] = jaccard_score
        common_nbrs_by_layer[key][[id_1, id_2], [id_2, id_1]] = n_common_nbr
        
        jaccard_by_layer[key][[id_1,id_2],[id_1, id_2]] = 1
        
        common_nbrs_by_layer[key][id_1,id_1] = 0 #len(graph[node_A])/10
        common_nbrs_by_layer[key][id_2,id_2] = 0 #len(graph[node_B])/10
    
    


# ## Plot Heatmap

# In[1263]:


def heatmap(data, ax=None,
            cbar_kw={}, cbarlabel="", label = '', bar = True, top = True, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    
    

    # We want to show all ticks...
    #ax.set_xticks(np.arange(data.shape[1]))
    #ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    #ax.set_xticklabels(col_labels)
    #ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    if top:
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    else:
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linestyle('solid')
        spine.set_color('black')
        spine.set_linewidth(1)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    #ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.grid(None)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    if label:
        ax.set_title(label)
    
    if bar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        return im, cbar
    else:
        return im


# ### Jaccard Scores

# In[1369]:


fig, ax = plt.subplots()

im, cbar = heatmap(jaccard_matrix, ax=ax,
                   cmap="hot_r")
#texts = annotate_heatmap(im, valfmt="{x:.1f} t")

fig.tight_layout()

plt.show()


# ### Common Neighbors

# In[1387]:


max_nbrs = np.max(common_nbr_matrix)


# In[1389]:


fig, ax = plt.subplots()

im, cbar = heatmap(common_nbr_matrix, ax=ax, cbar_kw = {'ticks': [0,50,100,150,200, max_nbrs]},
                   cmap="hot_r")
#texts = annotate_heatmap(im, valfmt="{x:.1f} t")

fig.tight_layout()

plt.show()


# ### By Asset Classes

# In[1391]:


fig = plt.figure(figsize = (8,8))

grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 2),
                axes_pad=0.5,
                share_all=True,
                label_mode="L",
                cbar_location="right",
                cbar_mode="single",
                cbar_pad = 0.15
                )


im2 = heatmap(common_nbrs_by_layer['CU'], ax=grid[0],
                   cmap="hot_r", cbarlabel="Jaccard Score", bar = False, label = 'Currency', top = False)

im3 = heatmap(common_nbrs_by_layer['CR'], ax=grid[1],
                   cmap="hot_r", cbarlabel="Jaccard Score", bar = False, label = 'Credit', top = False)

im4 = heatmap(common_nbrs_by_layer['EQ'], ax=grid[2],
                   cmap="hot_r", cbarlabel="Jaccard Score", bar = False, label = 'Equity', top = False)

im5 = heatmap(common_nbrs_by_layer['IR'], ax=grid[3],
                   cmap="hot_r", cbarlabel="Jaccard Score", bar = False, label = 'Interest Rate', top = False)


ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 117]
grid.cbar_axes[0].colorbar(im5, ticks = ticks)
#fig.tight_layout()

plt.show()


# In[1257]:


np.max(common_nbrs_by_layer['IR'])


# In[1393]:


fig = plt.figure(figsize = (8,8))

grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 2),
                axes_pad=0.5,
                share_all=True,
                label_mode="L",
                cbar_location="right",
                cbar_mode="single",
                cbar_pad = 0.15
                )



im2 = heatmap(jaccard_by_layer['CU'], ax=grid[0],
                   cmap="hot_r", cbarlabel="Jaccard Score", bar = False, label = 'Currency', top = False)

im3 = heatmap(jaccard_by_layer['CR'], ax=grid[1],
                   cmap="hot_r", cbarlabel="Jaccard Score", bar = False, label = 'Credit', top = False)

im4 = heatmap(jaccard_by_layer['EQ'], ax=grid[2],
                   cmap="hot_r", cbarlabel="Jaccard Score", bar = False, label = 'Equity', top = False)

im5 = heatmap(jaccard_by_layer['IR'], ax=grid[3],
                   cmap="hot_r", cbarlabel="Jaccard Score", bar = False, label = 'Interest Rate', top = False)



grid.cbar_axes[0].colorbar(im5)

plt.show()

