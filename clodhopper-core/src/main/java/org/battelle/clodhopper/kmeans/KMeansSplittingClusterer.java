package org.battelle.clodhopper.kmeans;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.battelle.clodhopper.AbstractClusterer;
import org.battelle.clodhopper.Cluster;
import org.battelle.clodhopper.ClusterSplitter;
import org.battelle.clodhopper.seeding.ClusterSeeder;
import org.battelle.clodhopper.seeding.PreassignedSeeder;
import org.battelle.clodhopper.task.ProgressHandler;
import org.battelle.clodhopper.task.TaskAdapter;
import org.battelle.clodhopper.task.TaskEvent;
import org.battelle.clodhopper.tuple.ArrayTupleList;
import org.battelle.clodhopper.tuple.TupleList;
import org.battelle.clodhopper.tuple.TupleMath;
import org.battelle.clodhopper.util.ArrayIntIterator;

/*=====================================================================
 * 
 *                       CLODHOPPER CLUSTERING API
 * 
 * -------------------------------------------------------------------- 
 * 
 * Copyright (C) 2013 Battelle Memorial Institute 
 * http://www.battelle.org
 * 
 * -------------------------------------------------------------------- 
 * 
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License.
 * 
 * -------------------------------------------------------------------- 
 * *
 * KMeansSplittingClusterer.java
 *
 *===================================================================*/

public abstract class KMeansSplittingClusterer extends AbstractClusterer {

	protected TupleList tuples;
	protected KMeansSplittingParams params;
	
	private TupleList initialClusterSeeds;
	
	private KMeansClusterer localKMeans;
	
	private Set<Cluster> unsplittables;
	
	private int maxClusters;
	
	private int splits;
	private List<Cluster> currentClusters;
	
	public KMeansSplittingClusterer(TupleList tuples, KMeansSplittingParams params) {
		if (tuples == null || params == null) {
			throw new NullPointerException();
		}
		this.tuples = tuples;
		this.params = params;
	}
	
	public TupleList getInitialClusterSeeds() {
		return initialClusterSeeds;
	}
	
	public void setInitialClusterSeeds(TupleList initialClusterSeeds) {
		this.initialClusterSeeds = initialClusterSeeds;
	}

	protected abstract void initializeIteration(List<Cluster> clusters);
	
	protected abstract ClusterSplitter createSplitter(List<Cluster> clusters, Cluster cluster);
	
	@Override
	protected List<Cluster> doTask() throws Exception {

		final int tupleCount = tuples.getTupleCount();
		final int tupleLength = tuples.getTupleLength();
        
        ProgressHandler ph = new ProgressHandler(this);
        ph.postBegin();
        
        ph.postIndeterminate();

        int[] allIDs = new int[tupleCount];
        for (int i = 0; i < tupleCount; i++) {
            allIDs[i] = i;
        }

        unsplittables = new HashSet<Cluster>();

        int numWorkerThreads = params.getWorkerThreadCount();
        if (numWorkerThreads <= 0) {
            numWorkerThreads = Runtime.getRuntime().availableProcessors();
        }

        List<Cluster> clusters = null;
        ExecutorService threadPool = null;

        try {
            
            if (numWorkerThreads > 1) {
                threadPool = Executors.newFixedThreadPool(numWorkerThreads);
            }

            int minClusters = Math.max(1, params.getMinClusters());
            if (minClusters > tupleCount) {
                minClusters = tupleCount;
            }
            
            this.maxClusters = params.getMaxClusters();
            if (this.maxClusters <= 0) {
                this.maxClusters = Integer.MAX_VALUE;
            }
            
            List<Cluster> workingList = null;
            
//            if (initialClusterSeeds != null || minClusters > 1) {
            if (initialClusterSeeds != null || minClusters >= 1) {

                int initialSeeds = initialClusterSeeds != null ? 
            			initialClusterSeeds.getTupleCount() : 0;
            	
                int nc = Math.max(minClusters, initialSeeds);
            	
                ClusterSeeder seeder = params.getClusterSeeder();
                if (initialClusterSeeds != null) {
            		seeder = new PreassignedSeeder(initialClusterSeeds);
            	}
            
                KMeansParams kparams = new KMeansParams.Builder()
                	.clusterCount(nc)
                	.maxIterations(Integer.MAX_VALUE)
                	.movesGoal(0)
                	.workerThreadCount(params.getWorkerThreadCount())
                	.distanceMetric(params.getDistanceMetric())
                	.clusterSeeder(seeder)
                	.build();
                
                localKMeans = new KMeansClusterer(tuples, kparams); 

                localKMeans.run();
            	workingList = localKMeans.get();
            	
            	localKMeans = null;

            } else { // mInitialClusterSeeds == null && minClusters == 1
                
            	workingList = new ArrayList<Cluster> ();
            	double[] center = TupleMath.average(tuples, new ArrayIntIterator(allIDs));           	
//            	workingList.add(new Cluster(allIDs, center));
            
            }
            
            int iteration = 0;

            double progress = 0.0;
            final double perIterationProgress = 0.95/10.0;
            
            do {

                initializeIteration(workingList);
                
                splits = 0;
                currentClusters = new ArrayList<Cluster>();

                final int numClusters = workingList.size();
                
                List<SplitCallable> splitterList = new ArrayList<SplitCallable>();
                
                for (int i=0; i<numClusters; i++) {
                    this.checkForCancel();
                    Cluster cluster = workingList.get(i);
                    if (!isUnsplittable(cluster)) {
//                        System.out.println("going to split " + cluster.getId());
                        splitterList.add(new SplitCallable(cluster, createSplitter(workingList, cluster)));
                    } else {
                        addToCurrentClusters(cluster);
                    }
                }

                if (splitterList.size() > 0) {
                    if (threadPool != null) {
                        List<Future<List<Cluster>>> results = threadPool.invokeAll(splitterList);
                        for (Future<List<Cluster>> result: results) {
                            List<Cluster> clist = result.get();
                            addToCurrentClusters(clist);
                            if (clist.size() > 1) {
                                incrementSplits();
                            } else if (clist.size() == 1) {
                                Cluster[] c = clist.toArray(new Cluster[1]);
                                addToUnsplittables(c[0]);
//                                System.out.println("unsplittable " + c[0].getId());
                            }
                        }
                    } else {
                        for (SplitCallable sc: splitterList) {
                            List<Cluster> clist = sc.call();
                            addToCurrentClusters(clist);
                            if (clist.size() > 1) {
                                incrementSplits();
                            } else if (clist.size() == 1) {
                                Cluster[] c = clist.toArray(new Cluster[1]);
                                addToUnsplittables(c[0]);
                            } 
                        }
                    }
                }

                int newNumClusters = currentClusters.size();

                workingList = new ArrayList<Cluster> (currentClusters);
                
                iteration++;
                
                int pctSplit = (int) (0.5 + 100.0 * ((double) splits)/numClusters);
                
                progress = Math.min(0.95, progress + perIterationProgress);
                
                if (pctSplit < 100 && progress < 0.5) {
                	progress = 0.5;
                }
                
                ph.postFraction(progress);
                
                ph.postMessage("loop " + iteration + ", percentage of clusters split = " + 
                		pctSplit + ", number of clusters = " + newNumClusters);

            } while (splits > 0 && 
                    workingList.size() < maxClusters);

            int numClusters = workingList.size();
                        
            TupleList finalSeeds = new ArrayTupleList(tupleLength, numClusters);
            for (int i=0; i<numClusters; i++) {
                finalSeeds.setTuple(i, workingList.get(i).getCenter());
            }
            
//            workingList = null;
            currentClusters = null;
            
            ph.postMessage("performing final round of k-means to polish up clusters");
            
            KMeansParams kparams = new KMeansParams.Builder()
            	.clusterCount(numClusters)
            	.maxIterations(Integer.MAX_VALUE)
            	.movesGoal(0)
            	.workerThreadCount(params.getWorkerThreadCount())
            	.distanceMetric(params.getDistanceMetric())
            	.clusterSeeder(new PreassignedSeeder(finalSeeds))
            	.build();
            localKMeans = new KMeansClusterer(tuples, kparams, workingList);
            localKMeans.addTaskListener(new TaskAdapter() {
                @Override
                public void taskMessage(TaskEvent e) {
                    postMessage("  (final k-means): " + e.getMessage());
                }
            });
            
            localKMeans.run();
            
            clusters = localKMeans.get();
            
            // In case k-means threw any away.
            numClusters = clusters.size();
            
            double minThreshold = params.getMinClusterToMeanThreshold();
            
            if (minThreshold > 0.0) {
                double avgSize = 0;
                int minSize = Integer.MAX_VALUE;
                for (int i=0; i<numClusters; i++) {
                    Cluster c = clusters.get(i);
                    int size = c.getMemberCount();
                    avgSize += size;
                    if (size < minSize) {
                        minSize = size;
                    }
                }
                avgSize /= numClusters;
                
                int intThreshold = (int) (0.5 + minThreshold * avgSize);
                if (minSize < intThreshold) {
                    // Some clusters were too small.
                    List<Cluster> bigEnough = new ArrayList<Cluster>(numClusters);
                    for (int i=0; i<numClusters; i++) {
                        Cluster c = clusters.get(i);
                        if (c.getMemberCount() >= intThreshold) {
                            bigEnough.add(c);
                        }
                    }
                    
                    int discard = numClusters - bigEnough.size();
                    numClusters = bigEnough.size();
      
                    finalSeeds = new ArrayTupleList(tupleLength, numClusters);
                    for (int i=0; i<numClusters; i++) {
                        finalSeeds.setTuple(i, bigEnough.get(i).getCenter());
                    }
                    
                    ph.postMessage(String.valueOf(discard) + " clusters will be discarded because of size");
                    
                    kparams.setClusterSeeder(new PreassignedSeeder(finalSeeds));
                    
                    localKMeans = new KMeansClusterer(tuples, kparams, bigEnough);
                    
                    localKMeans.run();
                    
                    clusters = localKMeans.get();                    
                }
            }
            
            ph.postMessage("final cluster count = " + numClusters);

            localKMeans = null;

            ph.postEnd();

        } finally {
            if (threadPool != null) {
                threadPool.shutdownNow();
            }
        }
        cleanID(clusters);
        return clusters;
	}

    private void cleanID(List<Cluster> clusters) {
        for (Cluster cluster : clusters) {
            String id = cluster.getId();
            int prefix = Integer.parseInt(id.substring(0,id.indexOf("-")));
            String prefixString = id.substring(0,id.indexOf("-"));
            if (prefix < 10)
                prefixString = '0' + prefixString;
//            char code = (char)Integer.parseInt(id.substring(0, id.indexOf("-")));
//            cluster.setId(Character.toString(code)+id.substring(id.indexOf("-")+1).replaceAll("-", ""));
            id = prefixString + '-' + id.substring(id.indexOf("-")).replaceAll("-", "");
            if (id.charAt(id.length()-1) == '-')
                id = id.substring(0, id.length()-1);
            cluster.setId(id);

            System.out.println(cluster.getId());
        }
    }

    public boolean cancel(boolean mayInterruptIfRunning) {
        if (super.cancel(mayInterruptIfRunning)) {
            if (localKMeans != null) {
                localKMeans.cancel(mayInterruptIfRunning);
            }
            synchronized (this) {
            	notifyAll();
            }
            return true;
        }
        return false;
    }
    
    private boolean isUnsplittable(Cluster cluster) {
        return unsplittables.contains(cluster);
    }

    private void addToUnsplittables(Cluster cluster) {
        unsplittables.add(cluster);
    }

    private void addToCurrentClusters(List<Cluster> clusters) {
        Iterator<Cluster> it = clusters.iterator();
        int numClusters = currentClusters.size();
        while(numClusters < maxClusters && it.hasNext()) {
            Cluster nextCluster = it.next();
            currentClusters.add(nextCluster);
            numClusters++;
        }
    }

    private void addToCurrentClusters(Cluster cluster) {
        if (currentClusters.size() < maxClusters) {
//            System.out.println("addToCurrentClusters " + cluster.getId());
            currentClusters.add(cluster);
        }
    }

    private void incrementSplits() {
        splits++;
    }

    class SplitCallable implements Callable<List<Cluster>> {

        private Cluster cluster;
        private ClusterSplitter splitter;
        
        SplitCallable(Cluster cluster, ClusterSplitter splitter) {
            this.cluster = cluster;
            this.splitter = splitter;
        }

        public List<Cluster> call() throws Exception {
            return splitter.split(cluster);
        }
    }
	
}
