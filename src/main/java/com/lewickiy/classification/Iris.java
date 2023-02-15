package com.lewickiy.classification;

import com.lewickiy.neuronnetwork.Network;
import com.lewickiy.util.Activation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.lewickiy.util.LoadCSV.loadCSV;
import static com.lewickiy.util.LoadCSV.max;
import static com.lewickiy.util.Normalize.normalizeByFeatureScaling;

public class Iris {
    public static final String IRIS_SETOSA = "Iris-setosa";
    public static final String IRIS_VERSICOLOR = "Iris-versicolor";
    public static final String IRIS_VERGINICA = "Iris-virginica";

    private final List<double[]> irisParameters = new ArrayList<>();
    private final List<double[]> irisClassifications = new ArrayList<>();
    private final List<String> irisSpecies = new ArrayList<>();

    public Iris() throws IOException {
        List<String[]> irisDataset = loadCSV("iris/");
        Collections.shuffle(irisDataset);

        for (String[] iris : irisDataset) {
            double[] parameters = Arrays.stream(iris).limit(4).mapToDouble(Double::parseDouble).toArray();
            irisParameters.add(parameters);

            String species = iris[4];
            switch (species) {
                case IRIS_SETOSA -> irisClassifications.add(new double[]{1.0, 0.0, 0.0});
                case IRIS_VERSICOLOR -> irisClassifications.add(new double[]{0.0, 1.0, 0.0});
                default -> irisClassifications.add(new double[]{0.0, 0.0, 1.0});
            }
            irisSpecies.add(species);
        }
        normalizeByFeatureScaling(irisParameters);
    }

    public String irisInterpretOutput(double[] output) {
        double max = max(output);
        if(max == output[0]) {
            return IRIS_SETOSA;
        } else if (max == output[1]) {
            return IRIS_VERSICOLOR;
        } else {
            return IRIS_VERGINICA;
        }
    }

    public Network<String>.Results classify() {

        Network<String> irisNetwork = new Network<>(
                new int[] {4, 12, 3},
                10.7,
                Activation::sigmoid,
                Activation::derivativeSigmoid
        );

        List<double[]> irisTrainers = irisParameters.subList(0, 120);
        List<double[]> irisTrainersCorrects = irisClassifications.subList(0, 120);
        int trainingIterations = 3;
        for (int i = 0; i < trainingIterations; i++) {
            irisNetwork.train(irisTrainers, irisTrainersCorrects);
        }

        List<double[]> irisTesters = irisParameters.subList(120, 150);
        List<String> irisTestersCorrects = irisSpecies.subList(120,150);
        return irisNetwork.validate(irisTesters, irisTestersCorrects, this::irisInterpretOutput);
    }
}
