package com.lewickiy.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Normalize {
    public static void normalizeByFeatureScaling(List<double[]> dataset) {
        for (int colNum = 0; colNum < dataset.get(0).length; colNum++) {
            List<Double> column = new ArrayList<>();

            for (double[] row: dataset) {
                column.add(row[colNum]);
            }

            double maximum = Collections.max(column);
            double minimum = Collections.min(column);
            double difference = maximum - minimum;

            for (double[] row : dataset) {
                row[colNum] = (row[colNum] - minimum) / difference;
            }
        }
    }
}
