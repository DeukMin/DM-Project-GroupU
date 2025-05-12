package src;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.util.Random;

public class RandomForestClassifier {

    public static void main(String[] args) {
        try {
            String[] datasetPaths = {
                    "Datasets/nursery_case0.arff",
                    "Datasets/nursery_case1.arff"
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

                // 2. Configure Random Forest
                RandomForest forest = new RandomForest();
                forest.setOptions(weka.core.Utils.splitOptions("-I 100 -K 0 -S 1")); // Set number of trees and other params
                forest.setMaxDepth(10);  // Set the maximum depth of the trees

                // 3. Build and save model
                forest.buildClassifier(dataset);
                String modelPath = "Models/RandomForest_" + (i + 1) + ".model";
                weka.core.SerializationHelper.write(modelPath, forest);

                // 4. Load model
                RandomForest loadedForest = (RandomForest) weka.core.SerializationHelper.read(modelPath);

                // 5. Evaluate with 10-fold cross-validation
                long startTime = System.currentTimeMillis();
                Evaluation eval = new Evaluation(dataset);
                eval.crossValidateModel(loadedForest, dataset, 10, new Random(1));
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

                // 7. Print the Random Forest model
                System.out.println("\n=== Random Forest Model ===");
                System.out.println(loadedForest);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}