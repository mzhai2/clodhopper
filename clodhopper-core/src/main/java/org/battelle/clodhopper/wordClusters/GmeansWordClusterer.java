package org.battelle.clodhopper.wordClusters;

import org.battelle.clodhopper.Cluster;
import org.battelle.clodhopper.distance.CosineDistanceMetric;
import org.battelle.clodhopper.gmeans.GMeansClusterer;
import org.battelle.clodhopper.gmeans.GMeansParams;
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

/**
 * Created by mike on 5/19/15.
 */
public class GmeansWordClusterer {

    public static void main(String[] args) throws UnsupportedEncodingException {
//        String path = URLDecoder.decode(GmeansWordClusterer.class.getProtectionDomain().getCodeSource().getLocation().getPath(), "UTF-8");
//        String path = "/Users/mike/Desktop/cluster/";
//        String[] matrices = {"10000x300"};
        String path = "/home/mzhai/cluster/";
        String[] matrices = {"glove.6B.300d"};
        BigDecimal threshold = new BigDecimal("0.05");
        List<BigDecimal> thresholds = new ArrayList<>();
        while (threshold.compareTo(new BigDecimal(".5")) == -1) {
            thresholds.add(threshold);
            threshold = threshold.add(new BigDecimal("0.1"));
        }
        System.out.println(thresholds);
        createCSV(matrices, path);
        clusterMatrices(matrices, path, thresholds);
    }

    private static Map<Integer, String> createDict(String matrix, String path) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(path+matrix+".data"));

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
                BufferedReader br = new BufferedReader(new FileReader(path + matrix + ".data"));
                BufferedWriter bw = new BufferedWriter(new FileWriter(path + matrix + ".csv"));
                String line;
                while ((line = br.readLine()) != null) {
                    line = line.substring(line.indexOf(" ") + 1).replace(' ', ',');
                    bw.write(line);
                    bw.flush();
                    bw.write('\n');
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
    }

    private static void clusterMatrices(String[] matrices, String path, List<BigDecimal> thresholds) {
        try {
            for (String matrix : matrices) {
                for (BigDecimal threshold : thresholds) {
                    Map<Integer, String> dict = createDict(matrix, path);
                    TupleListFactory factory = new ArrayTupleListFactory();
                    TupleList tuples = TupleIO.loadCSV(new File(path + matrix + ".csv"), "myData", factory);
                    GMeansParams.Builder builder = new GMeansParams.Builder();
                    GMeansParams params = builder.clusterSeeder(new KMeansPlusPlusSeeder(new CosineDistanceMetric()))
                            .maxClusters(tuples.getTupleCount() / 50)
                            .distanceMetric(new CosineDistanceMetric())
                            .minClusters(1)
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
                    t.start();
                    t.join();
                    if (gMeans.getTaskOutcome() == TaskOutcome.SUCCESS) {
                        List<Cluster> clusters = gMeans.get();
                        final int clusterCount = clusters.size();
                        System.out.printf("\nG-Means Generated %d Clusters\n\n", clusterCount);
                        BufferedWriter bw = new BufferedWriter(new FileWriter(path + matrix +"."+ threshold + ".cluster"));
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
                        System.out.printf("K-Means ended with the following error: %s\n", gMeans.getErrorMessage());
                    } else {
                        System.out.printf("K-Means ended with the unexpected outcome of: %s\n", gMeans.getTaskOutcome());
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
