package src;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.util.Random;

public class NaiveBayesClassifier {

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

                // 2. Configure Naive Bayes
                NaiveBayes bayes = new NaiveBayes();

                // 3. Build and save model
                bayes.buildClassifier(dataset);
                String modelPath = "Models/NaiveBayes_" + (i + 1) + ".model";
                weka.core.SerializationHelper.write(modelPath, bayes);

                // 4. Load model
                NaiveBayes loadedBayes = (NaiveBayes) weka.core.SerializationHelper.read(modelPath);

                // 5. Evaluate with 10-fold cross-validation
                long startTime = System.currentTimeMillis();
                Evaluation eval = new Evaluation(dataset);
                eval.crossValidateModel(loadedBayes, dataset, 10, new Random(1));
                long endTime = System.currentTimeMillis();
                double runtimeSeconds = (endTime - startTime) / 1000.0;

                // 6. Evaluation metrics
                System.out.println(eval.toSummaryString("\n=== Evaluation Results ===\n", false));
                System.out.println("Runtime (seconds): " + runtimeSeconds);
                System.out.println("AUC = " + eval.areaUnderROC(1));  // Using class index 1
                System.out.println("Kappa = " + eval.kappa());
                System.out.println("MAE = " + eval.meanAbsoluteError());
                System.out.println("RMSE = " + eval.rootMeanSquaredError());
                System.out.println("RAE = " + eval.relativeAbsoluteError());
                System.out.println("RRSE = " + eval.rootRelativeSquaredError());
                System.out.println("fMeasure = " + eval.fMeasure(0));
                System.out.println("Error Rate = " + eval.errorRate());
                System.out.println(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n"));
                System.out.println(eval.toMatrixString("\n=== Confusion Matrix ===\n"));

                // 7. Print the model
                System.out.println("\n=== Naive Bayes Model ===");
                System.out.println(loadedBayes);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}