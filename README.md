# OTLETA (One Thing LEads To Another)

## Market Analysis and Visualization Tool

Key Features: 

* Wrote a Python-based tool to analyze historical market data and display time and price projections.
* Employed the usage of Plotly for data visualization.
* Came up with a clustering algorithm to show meaningful overlaps in price retracements.

The main working repo for this project was originally hosted [here](https://gitlab.com/tanlubin1986/otleta).

## Clustering Algorithms

Given a group of sets of price values (determined by various analyses to be significant price points), the following algorithms output clusters deemed notable.

### clusterAlgo2
#### parameters: `spaceFactor`

1. Find max spacing (between adjacent lines in a set.)

2. Compare each set (each set's lines need to be sorted.)
For each line in a set, compare with the lines in the other sets. 
Find the minimum spacing `minSpace` between 2 lines from different sets.

2a. If no overlaps: give the 2 lines above and below the latest close that are nearest to the close.

2b. If there are overlaps: `cutoff` = `spaceFactor` x `minSpace`

Again, for each line in a set, compare with the lines in the other sets. If the distance between 2 lines are less than `cutoff`, add to the cluster list.

3. For both above and below the latest close, if clusters exist, show clusters. Otherwise, show closest price to the latest close. 
(May end up with just 1 line: if there are no clusters, if all the lines are above or if all the lines are below the latest close.)

### clusterAlgo
#### parameters: `minCluster`, `minSpaceFactorConst`

1. Find set with the smallest range (max - min).

2. Find smallest spacing between adjacent lines of this smallest range set. `unitSpacing` = smallest spacing divided by `minSpaceFactorConst`.

3. Starting from the biggest range set, for each line in the set, find their distance from the other lines, and if distance is smaller than `unitSpacing`, add to the cluster.

4. If there are clusters where the main lines are close to each other (distance is less than `unitSpacing`), combine the clusters together.

5. If number of clusters is less than `minCluster`, multiply `unitSpacing` by 1.5, and repeat steps 3 to 4.

6. Return the average of each cluster.
