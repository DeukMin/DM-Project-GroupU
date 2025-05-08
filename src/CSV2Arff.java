package src;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;

public class CSV2Arff{
    public static void main(String[]args) throws Exception{
        CSVLoader loader= new CSVLoader();
        ArffSaver saver = new ArffSaver();

        loader.setSource(new File("dataset/nursery_after_proc_case0.csv"));
        Instances data1= loader.getDataSet();

        saver.setInstances(data1);
        saver.setFile(new File("dataset/nursery_case0.arff"));
        saver.writeBatch();

        loader.setSource(new File("dataset/nursery_after_proc_case1.csv"));
        Instances data2= loader.getDataSet();

        saver.setInstances(data2);
        saver.setFile(new File("dataset/nursery_case1.arff"));
        saver.writeBatch();
    }
}