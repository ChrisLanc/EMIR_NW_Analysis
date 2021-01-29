# EMIR_NW_Analysis

These files would reproduce the analysis of EMIR transaction data conducted in the context of my Master Thesis.

Note that due to the confidentiality of the Data and corresponding information, the script that was acctually run is sligtly different. While basic functionality of individual parts should be conserved, the script was striped of critical parts that would violate confidentiality, especially variable names and several sub-analyses might be missing/inconsistent. This is just to showcase the overall procedure.

# Abstract

This work analyzes the bilateral exposure network of the German uncleared OTC derivatives market. Based on the analysis of transaction data reported under the EMIR regulatory framework, a directed and weighted multiplex network representation is constructed from aggregate bilateral exposures where each layer consists of a distinct derivatives asset class.Subsequently, a detailed analysis of its topological features is conducted. Additionally, the recently proposed multiplex centrality measure MultiRank is applied to the exposure network in a procedure to identify the economically relevant core of institutions and to analyze their level of interconnectedness. The findings show that the derivatives network shares many common features previously identified for interbank networks, such as a clear core-periphery structure, non-linear degree distributions and ’small world’ properties. Furthermore, it is shown that the MultiRank centrality identifies central institutions that other centrality measures based on aggregated networks do not capture. The institutions in the identified core show a high degree of common relationships and a heterogeneous structure in the size of shared relationships. The framework presented in this work provides authorities with the means to monitor high- volume and velocity regulatory transactions data on a day-to-day basis.


# Steps
1. Data_Preperation.py shows the basic preprocessing steps undertaken to clean the data and aggregate bilateral exposures. Note that other crucial steps around matching reported observations and initial subsampling of relevant transactions was done on the SQL Server and cannot be shows here.
2. Initiate the different graph structures used in subsequent analysis. (Un-)directd, (Un-)weighted, Quasi-Multiplex etc...
3. Generate a set of different metrics to characterize the topology of the different network asset classes.
4. Use a state of the art Multiplex Network Centrality metric (MultiRank) to identify the most interconnected institutions in the multiplex network. Subsequently attempt to reconcile the behavior of the algorithm to break it down to more readily understandable metrics and also compare it to more established network centrality metrics in order to gauge potential benefits.
5. Based on the respective MultiRank centrality, visualize the interconnectedness in the identified core of institutions in a Heat Map. 