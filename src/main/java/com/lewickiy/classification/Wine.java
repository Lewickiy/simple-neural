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

public class Wine {
    public List<double[]> wineParameters = new ArrayList<>();
    public List<double[]> wineCategories = new ArrayList<>();
    public List<Integer> wineSpecies = new ArrayList<>();

    public Wine() throws IOException {
        List<String[]> wineDataset = loadCSV("wine/");
        Collections.shuffle(wineDataset);
        for (String[] wineData : wineDataset) {
            double[] parameters = Arrays.stream(wineData).skip(1).mapToDouble(Double::parseDouble).toArray();
            wineParameters.add(parameters);
            int species = Integer.parseInt(wineData[0]);
            switch (species) {
                case 1:
                    wineCategories.add(new double[] {1.0, 0.0, 0.0});
                    break;
                case 2:
                    wineCategories.add(new double[] {0.0, 1.0, 0.0});
                    break;
                default:
                    wineCategories.add(new double[] {0.0, 0.0, 1.0});
            }
            wineSpecies.add(species);
        }
        normalizeByFeatureScaling(wineParameters);
    }

    public Integer wineInterpretOutput(double[] output) {
        double max = max(output);
        if(max == output[0]) {
            return 1;
        } else if (max == output[1]) {
            return 2;
        } else {
            return 3;
        }
    }

    public Network<Integer>.Results classify() {

        Network<Integer> wineNetwork = new Network<>(
                new int[] {13, 7, 3},
                0.79,
                Activation::sigmoid,
                Activation::derivativeSigmoid
        );

        List<double[]> wineTrainers = wineParameters.subList(0, 150);
        List<double[]> wineTrainersCorrects = wineCategories.subList(0, 150);
        int trainingIterations = 13;
        for (int i = 0; i < trainingIterations; i++) {
            wineNetwork.train(wineTrainers, wineTrainersCorrects);
        }

        List<double[]> wineTesters = wineParameters.subList(150, 177);
        List<Integer> wineTestersCorrects = wineSpecies.subList(150, 177);
        return wineNetwork.validate(wineTesters, wineTestersCorrects, this::wineInterpretOutput);
    }
}
