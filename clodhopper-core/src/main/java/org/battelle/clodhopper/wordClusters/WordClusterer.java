package org.battelle.clodhopper.wordClusters;

import org.battelle.clodhopper.Cluster;
import org.battelle.clodhopper.distance.ChebyshevDistanceMetric;
import org.battelle.clodhopper.distance.CosineDistanceMetric;
import org.battelle.clodhopper.distance.DistanceMetric;
import org.battelle.clodhopper.distance.EuclideanDistanceMetric;
import org.battelle.clodhopper.gmeans.GMeansClusterer;
import org.battelle.clodhopper.gmeans.GMeansParams;
import org.battelle.clodhopper.jarvispatrick.JarvisPatrickClusterer;
import org.battelle.clodhopper.jarvispatrick.JarvisPatrickParams;
import org.battelle.clodhopper.seeding.KMeansPlusPlusSeeder;
import org.battelle.clodhopper.task.TaskEvent;
import org.battelle.clodhopper.task.TaskListener;
import org.battelle.clodhopper.task.TaskOutcome;
import org.battelle.clodhopper.tuple.ArrayTupleListFactory;

import java.io.*;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.battelle.clodhopper.tuple.TupleIO;
import org.battelle.clodhopper.tuple.TupleList;
import org.battelle.clodhopper.tuple.TupleListFactory;
import org.battelle.clodhopper.xmeans.XMeansClusterer;
import org.battelle.clodhopper.xmeans.XMeansParams;

/**
 * Created by mike on 5/19/15.
 */
public class WordClusterer {

    public static void main(String[] args) throws UnsupportedEncodingException {
//        String path = URLDecoder.decode(WordClusterer.class.getProtectionDomain().getCodeSource().getLocation().getPath(), "UTF-8");
        String path = "/Users/mike/Desktop/cluster/";
        String[] matrices = {"10000x300"};
//        String path = "/home/mzhai/cluster/";
//        String[] matrices = {"glove.6B.300d"};

        List<DistanceMetric> metrics = new ArrayList<>();
        metrics.add(new CosineDistanceMetric());
        metrics.add(new EuclideanDistanceMetric());
        metrics.add(new ChebyshevDistanceMetric());

        BigDecimal threshold = new BigDecimal("0.05");
        List<BigDecimal> thresholds = new ArrayList<>();
        thresholds.add(threshold);
//        while (threshold.compareTo(new BigDecimal(".5")) == -1) {
//            thresholds.add(threshold);
//            threshold = threshold.add(new BigDecimal("0.1"));
//        }
        createCSV(matrices, path);
        clusterGmeans(matrices, path, metrics, thresholds);
        boolean[] booleans = {true, false};
        clusterXmeans(matrices, path, metrics, thresholds, booleans);
        clusterJarvisPatrick(matrices, path, metrics, thresholds, booleans);
    }

    private static Map<Integer, String> createDict(String matrix, String path) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(path+matrix));

            Map<Integer, String> dict = new HashMap<>();
            String line;
            for (int i=0; (line = br.readLine()) != null; i++) {
                line = line.substring(0, line.indexOf(" "));
                dict.put(i, line);
            }
            return dict;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    private static void createCSV(String[] matrices, String path) {
        for (String matrix : matrices) {
            if (new File(path + matrix + ".csv").exists())
                continue;
            try {
                BufferedReader br = new BufferedReader(new FileReader(path + matrix));
                BufferedWriter bw = new BufferedWriter(new FileWriter(path + matrix + ".csv"));
                String line;
                while ((line = br.readLine()) != null) {
                    line = line.substring(line.indexOf(" ") + 1);
                    String[] valuesStr = line.split(" ");
                    double sum = 0;
                    for (String valueStr : valuesStr) {
                        double value = Double.parseDouble(valueStr);
                        sum += value;
                    }
                    double mean = sum/valuesStr.length;
                    for (int i=0;i<valuesStr.length;i++) {
                        String valueStr = valuesStr[i];
                        bw.write(Double.toString(Double.parseDouble(valueStr)-mean));
                        if (i == valuesStr.length-1)
                            bw.write('\n');
                        else {
                            bw.write(", ");
                        }
                    }
                    bw.flush();
                }
                bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
    }

    private static void clusterGmeans(String[] matrices, String path, List<DistanceMetric> metrics, List<BigDecimal> thresholds) {
        try {
            for (String matrix : matrices) {
                for (DistanceMetric metric : metrics) {
                    for (BigDecimal threshold : thresholds) {
                        Map<Integer, String> dict = createDict(matrix, path);
                        TupleListFactory factory = new ArrayTupleListFactory();
                        TupleList tuples = TupleIO.loadCSV(new File(path + matrix + ".csv"), "myData", factory);
                        GMeansParams.Builder builder = new GMeansParams.Builder();
                        GMeansParams params = builder.clusterSeeder(new KMeansPlusPlusSeeder(metric))
                                .maxClusters(tuples.getTupleCount())
                                .distanceMetric(metric)
                                .minClusters(200)
                                .minClusterToMeanThreshold(threshold.doubleValue())
                                .workerThreadCount(Runtime.getRuntime().availableProcessors())
                                .build();
                        GMeansClusterer gMeans = new GMeansClusterer(tuples, params);
                        gMeans.addTaskListener(new TaskListener() {
                            @Override
                            public void taskBegun(TaskEvent e) {
                                System.out.println(e.getMessage());
                            }

                            @Override
                            public void taskMessage(TaskEvent e) {
                                System.out.println(e.getMessage());
                            }

                            @Override
                            public void taskProgress(TaskEvent e) {

                            }

                            @Override
                            public void taskPaused(TaskEvent e) {

                            }

                            @Override
                            public void taskResumed(TaskEvent e) {

                            }

                            @Override
                            public void taskEnded(TaskEvent e) {
                                System.out.println(e.getMessage());
                            }
                        });
                        Thread t = new Thread(gMeans);
                        System.out.println("Starting with " + Runtime.getRuntime().availableProcessors() + " threads");
                        t.start();
                        t.join();
                        if (gMeans.getTaskOutcome() == TaskOutcome.SUCCESS) {
                            List<Cluster> clusters = gMeans.get();
                            final int clusterCount = clusters.size();
                            System.out.printf("\nG-Means Generated %d Clusters\n\n", clusterCount);
                            BufferedWriter bw = new BufferedWriter(new FileWriter(path + matrix + metric + threshold + ".cluster"));
                            for (int i = 0; i < clusterCount; i++) {
                                Cluster c = clusters.get(i);
                                StringBuilder sb = new StringBuilder();
                                final int memberCount = c.getMemberCount();
                                for (int j = 0; j < memberCount; j++) {
                                    if (j > 0) {
                                        sb.append(",");
                                    }
                                    sb.append(c.getMember(j));
                                }
                                toCluster(sb.toString(), dict, bw);
                            }
                        } else if (gMeans.getTaskOutcome() == TaskOutcome.ERROR) {
                            System.out.printf("G-Means ended with the following error: %s\n", gMeans.getErrorMessage());
                        } else {
                            System.out.printf("G-Means ended with the unexpected outcome of: %s\n", gMeans.getTaskOutcome());
                        }
                    }
                }
            }
        }   catch (Throwable t) {
            t.printStackTrace();
        }
    }
    private static void clusterXmeans(String[] matrices, String path, List<DistanceMetric> metrics, List<BigDecimal> thresholds, boolean[] bic) {
        try {
            for (String matrix : matrices) {
                for (DistanceMetric metric : metrics) {
                    for (BigDecimal threshold : thresholds) {
                        for (boolean bo : bic) {
                            Map<Integer, String> dict = createDict(matrix, path);
                            TupleListFactory factory = new ArrayTupleListFactory();
                            TupleList tuples = TupleIO.loadCSV(new File(path + matrix + ".csv"), "myData", factory);
                            XMeansParams.Builder builder = new XMeansParams.Builder();
                            XMeansParams params = builder.clusterSeeder(new KMeansPlusPlusSeeder(metric))
                                    .maxClusters(tuples.getTupleCount())
                                    .distanceMetric(metric)
                                    .minClusters(200)
                                    .minClusterToMeanThreshold(threshold.doubleValue())
                                    .workerThreadCount(Runtime.getRuntime().availableProcessors())
                                    .userOverallBIC(bo)
                                    .build();
                            XMeansClusterer xMeans = new XMeansClusterer(tuples, params);
                            xMeans.addTaskListener(new TaskListener() {
                                @Override
                                public void taskBegun(TaskEvent e) {
                                    System.out.println(e.getMessage());
                                }

                                @Override
                                public void taskMessage(TaskEvent e) {
                                    System.out.println(e.getMessage());
                                }

                                @Override
                                public void taskProgress(TaskEvent e) {

                                }

                                @Override
                                public void taskPaused(TaskEvent e) {

                                }

                                @Override
                                public void taskResumed(TaskEvent e) {

                                }

                                @Override
                                public void taskEnded(TaskEvent e) {
                                    System.out.println(e.getMessage());
                                }
                            });
                            Thread t = new Thread(xMeans);
                            t.start();
                            t.join();
                            if (xMeans.getTaskOutcome() == TaskOutcome.SUCCESS) {
                                List<Cluster> clusters = xMeans.get();
                                final int clusterCount = clusters.size();
                                System.out.printf("\nG-Means Generated %d Clusters\n\n", clusterCount);
                                BufferedWriter bw = new BufferedWriter(new FileWriter(path + matrix + metric + threshold + bic + ".cluster"));
                                for (int i = 0; i < clusterCount; i++) {
                                    Cluster c = clusters.get(i);
                                    StringBuilder sb = new StringBuilder();
                                    final int memberCount = c.getMemberCount();
                                    for (int j = 0; j < memberCount; j++) {
                                        if (j > 0) {
                                            sb.append(",");
                                        }
                                        sb.append(c.getMember(j));
                                    }
                                    toCluster(sb.toString(), dict, bw);
                                }
                            } else if (xMeans.getTaskOutcome() == TaskOutcome.ERROR) {
                                System.out.printf("G-Means ended with the following error: %s\n", xMeans.getErrorMessage());
                            } else {
                                System.out.printf("G-Means ended with the unexpected outcome of: %s\n", xMeans.getTaskOutcome());
                            }
                        }
                    }
                }
            }
        }   catch (Throwable t) {
            t.printStackTrace();
        }
    }

    private static void clusterJarvisPatrick(String[] matrices, String path, List<DistanceMetric> metrics, List<BigDecimal> thresholds, boolean[] mutual) {
        try {
            for (String matrix : matrices) {
                for (DistanceMetric metric : metrics) {
                    for (BigDecimal threshold : thresholds) {
                        for (boolean bo : mutual) {
                            Map<Integer, String> dict = createDict(matrix, path);
                            TupleListFactory factory = new ArrayTupleListFactory();
                            TupleList tuples = TupleIO.loadCSV(new File(path + matrix + ".csv"), "myData", factory);
                            JarvisPatrickParams.Builder builder = new JarvisPatrickParams.Builder();
                            JarvisPatrickParams params = builder.mutualNearestNeighbors(bo)
                                    .nearestNeighborOverlap(20)
                                    .nearestNeighborsToExamine(2)
                                    .distanceMetric(metric)
                                    .workerThreadCount(Runtime.getRuntime().availableProcessors())
                                    .build();
                            JarvisPatrickClusterer jarvisPatrick = new JarvisPatrickClusterer(tuples, params);
                            jarvisPatrick.addTaskListener(new TaskListener() {
                                @Override
                                public void taskBegun(TaskEvent e) {
                                    System.out.println(e.getMessage());
                                }

                                @Override
                                public void taskMessage(TaskEvent e) {
                                    System.out.println(e.getMessage());
                                }

                                @Override
                                public void taskProgress(TaskEvent e) {

                                }

                                @Override
                                public void taskPaused(TaskEvent e) {

                                }

                                @Override
                                public void taskResumed(TaskEvent e) {

                                }

                                @Override
                                public void taskEnded(TaskEvent e) {
                                    System.out.println(e.getMessage());
                                }
                            });
                            Thread t = new Thread(jarvisPatrick);
                            t.start();
                            t.join();
                            if (jarvisPatrick.getTaskOutcome() == TaskOutcome.SUCCESS) {
                                List<Cluster> clusters = jarvisPatrick.get();
                                final int clusterCount = clusters.size();
                                System.out.printf("\nG-Means Generated %d Clusters\n\n", clusterCount);
                                BufferedWriter bw = new BufferedWriter(new FileWriter(path + matrix + metric + threshold + mutual + ".cluster"));
                                for (int i = 0; i < clusterCount; i++) {
                                    Cluster c = clusters.get(i);
                                    StringBuilder sb = new StringBuilder();
                                    final int memberCount = c.getMemberCount();
                                    for (int j = 0; j < memberCount; j++) {
                                        if (j > 0) {
                                            sb.append(",");
                                        }
                                        sb.append(c.getMember(j));
                                    }
                                    toCluster(sb.toString(), dict, bw);
                                }
                            } else if (jarvisPatrick.getTaskOutcome() == TaskOutcome.ERROR) {
                                System.out.printf("G-Means ended with the following error: %s\n", jarvisPatrick.getErrorMessage());
                            } else {
                                System.out.printf("G-Means ended with the unexpected outcome of: %s\n", jarvisPatrick.getTaskOutcome());
                            }
                        }
                    }
                }
            }
        }   catch (Throwable t) {
            t.printStackTrace();
        }
    }


    private static void toCluster(String s, Map<Integer, String> dict, BufferedWriter bw) {
        try {
            for (String l : s.split("\n")) {
                for (String key : l.split(",")) {
                    bw.write(dict.get(Integer.parseInt(key)) + " ");
                }
                bw.write("\n");
                bw.flush();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
