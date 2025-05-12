package src;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.util.Random;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import javax.swing.*;

public class DecisionTree {

    public static void main(String[] args) {
        try {

            String[] datasetPaths = {
                    "Datasets/nursery_case0.arff",  // All attributes
                    "Datasets/nursery_case1.arff"   // Selected attributes
            };

            // Model output directory
            File modelDir = new File("Models");
            if (!modelDir.exists()) {
                modelDir.mkdirs();
            }

            for (int i = 0; i < datasetPaths.length; i++) {
                String datasetPath = datasetPaths[i];
                System.out.println("\n===== Processing: " + datasetPath + " =====");

                // 1. Load dataset
                DataSource source = new DataSource(datasetPath);
                Instances dataset = source.getDataSet();

                // Set class index to last attribute (target)
                dataset.setClassIndex(dataset.numAttributes() - 1);

                // 2. Configure Decision Tree with pruning
                J48 tree = new J48();
//                String[] options = {
//                        "-C", "0.25",  // Confidence threshold for pruning
//                        "-M", "2"       // Minimum number of instances per leaf
//                };
//                tree.setOptions(options);

                // 3. Build and save model
                tree.buildClassifier(dataset);
                String modelPath = "Models/DecisionTree_" + (i+1) + ".model";
                weka.core.SerializationHelper.write(modelPath, tree);

                // 4. Load model
                J48 loadedTree = (J48) weka.core.SerializationHelper.read(modelPath);

                // 5. Evaluate with 10-fold cross-validation
                long startTime = System.currentTimeMillis();
                Evaluation eval = new Evaluation(dataset);
                eval.crossValidateModel(loadedTree, dataset, 10, new Random(1));
                long endTime = System.currentTimeMillis();
                double runtimeSeconds = (endTime - startTime) / 1000.0;

                // 6. Evaluation metrics
                System.out.println(eval.toSummaryString("\n=== Evaluation Results ===\n", false));
                System.out.println("Runtime (seconds): " + runtimeSeconds);
                System.out.println("AUC = " + eval.areaUnderROC(1));  // Using class index 1
                System.out.println("fMeasure = " + eval.fMeasure(0));
                System.out.println("Error Rate = " + eval.errorRate());
                System.out.println(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n"));
                System.out.println(eval.toMatrixString("\n=== Confusion Matrix ===\n"));

                // 7. Print the tree model
                System.out.println("\n=== Decision Tree Model ===");
                System.out.println(loadedTree);

                // 8. Visualize the tree
                visualizeTree(loadedTree, "Nursery Decision Tree - Case " + (i+1));
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void visualizeTree(J48 tree, String title) throws Exception {
        TreeVisualizer tv = new TreeVisualizer(null, tree.graph(), new PlaceNode2());
        JFrame frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(1200, 800);
        frame.getContentPane().add(tv);
        frame.setVisible(true);
        tv.fitToScreen();
    }
}
