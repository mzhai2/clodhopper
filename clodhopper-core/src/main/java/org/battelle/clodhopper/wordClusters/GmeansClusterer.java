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
import java.util.*;
import java.util.concurrent.*;

import org.battelle.clodhopper.tuple.TupleIO;
import org.battelle.clodhopper.tuple.TupleList;
import org.battelle.clodhopper.tuple.TupleListFactory;

/**
 * Created by mike on 5/19/15.
 */
public class GmeansClusterer {

    public static void main(String[] args) {
//        String path = "/home/mzhai/cluster/";
        String path = "/Users/mike/Desktop/cluster/";
//        String matrix = "glove.6B.300d";
//        String matrix = "10000x300";
//        String matrix = "glove.840B.300d";
        long start_time = System.nanoTime();
        createCSV(args[0], path);
        long end_time = System.nanoTime();
        double difference = (end_time - start_time)/1e9;
        System.out.println("csv processing complete in " + difference + "seconds");
        System.out.println(661001572/difference + " tokens/second");
        clusterGmeans(args[0], path, Integer.parseInt(args[1]), Integer.parseInt(args[2]));
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

    private static void createCSV(String matrix, String path) {
        if (new File(path + matrix + ".csv").exists())
            return;
        try {
            BufferedReader bf = new BufferedReader(new FileReader(path + matrix));
            BufferedWriter bw = new BufferedWriter(new FileWriter(path + matrix + ".csv"));
            int availProc = Runtime.getRuntime().availableProcessors();
            ExecutorService pool =  Executors.newFixedThreadPool(availProc);
            String line;
            BlockingQueue<StringBuilder> sbPool = new ArrayBlockingQueue<>(2*availProc);
            BlockingQueue<List<String>> listPool = new ArrayBlockingQueue<>(2*availProc);
            for (int i=0; i<2*availProc;i++) {
                sbPool.put(new StringBuilder());
                listPool.put(new ArrayList<>(300));
            }
            BlockingQueue<Future<String>> normalizedValues = new ArrayBlockingQueue<>(availProc);
            Writer writer = new Writer(normalizedValues, bw);
            Thread writerThread = new Thread(writer);
            writerThread.start();
            int lineCount = 0;
            List<String> lines = listPool.take();
            while (true) {
                if ((lineCount != 0) && ((lineCount % 6000) == 0)) {
                    normalizedValues.put(pool.submit(new Normalizer(lines, listPool, sbPool)));
                    lines = listPool.take();
                }
                line = bf.readLine();
                lineCount++;
                if (line == null) {
                    if (lines.size() > 0) {
                        normalizedValues.put(pool.submit(new Normalizer(lines, listPool, sbPool)));
                    }
                    break;
                }
                lines.add(line);
            }
            if (writer != null) {
                writer.terminate();
                writerThread.join();
            }
            pool.shutdown();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static class Writer implements Runnable {
        private volatile boolean running;
        private BlockingQueue<Future<String>> queue;
        private BufferedWriter bw;

        public Writer(BlockingQueue<Future<String>> queue, BufferedWriter bw) {
            super();
            this.queue = queue;
            this.bw = bw;
            running = true;
        }

        public void terminate() {
            running = false;
        }

        @Override
        public void run() {
            while (running) {
                try {
                    while (!queue.isEmpty()) {
                        bw.write(queue.take().get());
                        bw.flush();
                    }
                } catch (InterruptedException e) {
                    running = false;
                } catch (ExecutionException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
    public static class Normalizer implements Callable<String> {
        private final BlockingQueue<List<String>> listPool;
        private BlockingQueue<StringBuilder> sbPool;
        private List<String> lines;

        public Normalizer(List<String> lines, BlockingQueue<List<String>> listPool, BlockingQueue<StringBuilder> sbPool) {
            this.listPool = listPool;
            this.sbPool = sbPool;
            this.lines = lines;
        }

        @Override
        public String call() throws Exception
        {
            StringBuilder sb = sbPool.remove();
            for (String line : lines) {
                line = line.substring(line.indexOf(" ") + 1);
                String[] valuesStr = line.split(" ");
                float[] values = new float[valuesStr.length];
                float sum = 0;
                for (int i = 0; i < valuesStr.length; i++) {
                    float value = Float.parseFloat(valuesStr[i]);
                    values[i] = value;
                    sum += value;
                }
                double mean = sum / valuesStr.length;
                for (float value : values) {
                    sb.append(String.format("%.8f", value-mean));
                    sb.append(", ");
                }
                sb.replace(sb.length()-2, sb.length(), "\n");
            }
            String result = sb.toString();
            lines.clear();
            listPool.put(lines);
            sb.setLength(0);
            sbPool.put(sb);
            return result;
        }
    }



    private static void clusterGmeans(String matrix, String path, int min, int max) {
        try {
            Map<Integer, String> dict = createDict(matrix, path);
            TupleListFactory factory = new ArrayTupleListFactory();
            TupleList tuples = TupleIO.loadCSV(new File(path + matrix + ".csv"), "myData", factory);
            GMeansParams.Builder builder = new GMeansParams.Builder();
            GMeansParams params = builder.clusterSeeder(new KMeansPlusPlusSeeder(1, new Random(), new CosineDistanceMetric()))
                    .maxClusters(max)
                    .distanceMetric(new CosineDistanceMetric())
                    .minClusters(min)
                    .minClusterToMeanThreshold(0.05)
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
                BufferedWriter resultWriter = new BufferedWriter(new FileWriter(path + matrix + '_' + min + '_' + max + ".keys"));
                BufferedWriter readableWriter = new BufferedWriter(new FileWriter(path + matrix + '_' + min + '_' + max + ".readable"));
                BufferedWriter clusterWriter = new BufferedWriter(new FileWriter(path + matrix + '_' + min + '_' + max + ".cluster"));
                for (int i = 0; i < clusterCount; i++) {
                    Cluster c = clusters.get(i);
                    StringBuilder sb = new StringBuilder();
                    final int memberCount = c.getMemberCount();
                    String id = null;
                    for (int j = 0; j < memberCount; j++) {
                        if (j > 0) {
                            sb.append(",");
                        }
                        else {
                            if (min == 1)
                                 id = c.getId().substring(1);
                            else
                                id = c.getId();}
                        sb.append(c.getMember(j));
                    }
                    resultWriter.write(sb.toString());
                    toReadable(id, sb.toString(), dict, readableWriter);
                    toCluster(id, sb.toString(), dict, clusterWriter);
                    if (i != clusterCount-1) {
                        resultWriter.write('\n');
                        readableWriter.write('\n');
                    }
                }
                resultWriter.close();

            } else if (gMeans.getTaskOutcome() == TaskOutcome.ERROR) {
                System.out.printf("G-Means ended with the following error: %s\n", gMeans.getErrorMessage());
            } else {
                System.out.printf("G-Means ended with the unexpected outcome of: %s\n", gMeans.getTaskOutcome());
            }

        }   catch (Throwable t) {
            t.printStackTrace();
        }
    }

    private static void toCluster(String id, String s, Map<Integer, String> dict, BufferedWriter bw) {
        try {
            String[] keys = s.split(",");
            for (int i=0; i<keys.length; i++) {
                String key = keys[i];
                bw.write(id + ' ' + dict.get(Integer.parseInt(key)));
                if (i != keys.length-1)
                    bw.write('\n');
                }
            bw.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void toReadable(String id, String s, Map<Integer, String> dict, BufferedWriter bw) {
        try {
            bw.write(id + ": ");
            String[] keys = s.split(",");
            for (int i=0; i< keys.length; i++) {
                String key = keys[i];
                bw.write(dict.get(Integer.parseInt(key)) + ' ');
            }
            bw.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}