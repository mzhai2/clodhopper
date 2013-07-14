package org.battelle.clodhopper.examples.kmeans;

import java.io.File;
import java.util.List;

import org.battelle.clodhopper.Cluster;
import org.battelle.clodhopper.distance.EuclideanDistanceMetric;
import org.battelle.clodhopper.kmeans.KMeansClusterer;
import org.battelle.clodhopper.kmeans.KMeansParams;
import org.battelle.clodhopper.seeding.KMeansPlusPlusSeeder;
import org.battelle.clodhopper.task.TaskEvent;
import org.battelle.clodhopper.task.TaskListener;
import org.battelle.clodhopper.task.TaskOutcome;
import org.battelle.clodhopper.tuple.ArrayTupleListFactory;
import org.battelle.clodhopper.tuple.TupleIO;
import org.battelle.clodhopper.tuple.TupleList;
import org.battelle.clodhopper.tuple.TupleListFactory;

public class SimpleKMeansDemo {

  public static void main(String[] args) {

    // This example expects you to provide the name of a csv file containing the data
    // you wish to cluster.
    if (args.length != 1) {
      System.err.printf("Usage: java %s <input csv file>\n", SimpleKMeansDemo.class.getName());
      System.exit(-1);
    }
    
    // Since this is a simple example, wrap it all in an all-emcompassing try-catch.
    //
    try {
      
      //
      // Step 1. Read the data in the file into a TupleList.
      //
    
      // First, create a TupleListFactory.  For this example, just use 
      // a simple ArrayTupleListFactory.  This factory produces ArrayTupleLists, which
      // use a 1-D array of doubles to house the data.
      TupleListFactory tupleListFactory = new ArrayTupleListFactory();
      // Call TupleIO to read it.  It doesn't matter what name is given to the TupleList.
      TupleList tuples = TupleIO.loadCSV(new File(args[0]), "myData", tupleListFactory);
      
      // 
      // Step 2. Configure the k-mean parameters object.
      //
      
      // For this example, arbitrarily choose the square root of the number 
      // of tuples as the number of clusters.  
      int numClusters = (int) Math.sqrt(tuples.getTupleCount());

      // Since k-means has many parameters, it's easiest to use the nested builder class.
      KMeansParams.Builder builder = new KMeansParams.Builder();
      
      // Generate the parameters object.
      //
      KMeansParams params = builder.clusterCount(numClusters)
        .clusterSeeder(new KMeansPlusPlusSeeder(new EuclideanDistanceMetric()))
        .distanceMetric(new EuclideanDistanceMetric())
        .workerThreadCount(Runtime.getRuntime().availableProcessors())
        .build();
      
      KMeansClusterer kMeans = new KMeansClusterer(tuples, params);
      
      // Register a TaskListener that will output messages to System.out.
      //
      kMeans.addTaskListener(new TaskListener() {

        @Override
        public void taskBegun(TaskEvent e) {
          // The initial event.
          System.out.println(e.getMessage());
        }

        @Override
        public void taskMessage(TaskEvent e) {
          // Status messages as the task proceeds.
          System.out.printf("\t ... %s\n", e.getMessage());
        }

        @Override
        public void taskProgress(TaskEvent e) {
          // This could be used to drive a progress bar, 
          // but we will ignore for this example.
        }

        @Override
        public void taskPaused(TaskEvent e) {
          // Ignore -- no provision in this example for pause 
          // to be triggered.
        }

        @Override
        public void taskResumed(TaskEvent e) {
          // Also ignore for this example.
        }

        @Override
        public void taskEnded(TaskEvent e) {
          // The final event.
          System.out.println(e.getMessage());
        }        
      });
      
      // Lauch k-means on a new thread.  (We could be REALLY simple and call k-means.run()...)
      Thread t = new Thread(kMeans);
      t.start();
      
      // Just join the thread to wait for it to finish.  
      t.join();
      
      // Check that k-means succeeded.
      if (kMeans.getTaskOutcome() == TaskOutcome.SUCCESS) {
        
        // All the clusterers extend AbstractClusterer, which is a 
        // Future<List<Cluster>> and has a blocking get().  Since kMeans has been
        // verified to have finished successfully, we can confidently call get().
        //
        List<Cluster> clusters = kMeans.get();
        
        // Print results to System.out.
        //
        final int clusterCount = clusters.size();
        
        System.out.printf("\nK-Means Generated %d Clusters\n\n", clusterCount);
        
        for (int i=0; i<clusterCount; i++) {
          Cluster c = clusters.get(i);
          StringBuilder sb = new StringBuilder("Cluster " + (i+1) + ": ");
          final int memberCount = c.getMemberCount();
          for (int j=0; j<memberCount; j++) {
            if (j > 0) {
              sb.append(", ");
            }
            sb.append(c.getMember(j));
          }
          System.out.println(sb.toString());
        }
        
      } else if (kMeans.getTaskOutcome() == TaskOutcome.ERROR) {
        
        System.out.printf("K-Means ended with the following error: %s\n", kMeans.getErrorMessage());
        
      }
    
    } catch (Throwable t) {
      
      t.printStackTrace();
      
    }
    
  }

}
