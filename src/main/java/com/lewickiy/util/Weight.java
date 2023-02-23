package com.lewickiy.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;

public class Weight {
    private static final String FILE_PATH = "learnedWeights.csv";
    private static ArrayList<String> learnedWeightsArray = new ArrayList<>();
    private static PrintWriter writer;

    public static void saveWeights(String neuronWeights) {
        learnedWeightsArray.add(neuronWeights);
        System.out.println(learnedWeightsArray.size() + " learnedWeightsArray SIZE");
    }

    public static void saveWeightsToCSV() {
/*        for (int i1 = 0; i1 < randomWeights.length; i1++) {
            buildWeightsString.append(randomWeights[i1]);
            if (i1 < (randomWeights.length - 1)) {
                buildWeightsString.append(',');
            } else {
                buildWeightsString.append('\n');
            }
        }*/
    }

    public static void clearWeights() {
        //TODO clear array
    }

    /**
     * Проверка наличия файла с весами нейронов при первом запуске.
     * Также может использоваться при обработке исключений
     *
     * @return false/true
     */
    public static boolean isLearningWeightsExist() {
        File file = new File(FILE_PATH);
        return file.exists();
    }

    /**
     * Создание пустого файла для сохранения весов нейронов при первом запуске.
     */
    public static void createLearningWeightsCSV() {
        try {
            writer = new PrintWriter(FILE_PATH);
        } catch (FileNotFoundException e) {
            //TODO обработать
            throw new RuntimeException(e);
        }
        writer.close();
    }
}
