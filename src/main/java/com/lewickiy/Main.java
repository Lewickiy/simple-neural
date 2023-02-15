package com.lewickiy;

import com.lewickiy.classification.Wine;
import com.lewickiy.neuronnetwork.Network;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {

/* IRIS
        int iteration = 1;
        double sum = 0.0;
        Iris iris = new Iris();
        Network<String>.Results results = iris.classify();
        sum = sum + results.percentage;

        while (iteration < 1000) {
            Iris irisNext = new Iris();
            Network<String>.Results resultsNext = irisNext.classify();
            sum = sum + resultsNext.percentage;

            iteration++;
        }
        System.out.println(
                (sum * 100) / iteration + " средний балл на "
                        + iteration + " итераций"
        );*/

        //WINE
        int iteration = 1;
        double sum = 0.0;
        Wine wine = new Wine();
        Network<Integer>.Results results = wine.classify();
        sum = sum + results.percentage;

        while (iteration < 1000) {
            Wine wineNext = new Wine();
            Network<Integer>.Results resultsNext = wineNext.classify();
            sum = sum + resultsNext.percentage;
            if (resultsNext.percentage < 0.78) {
                System.out.println(
                        (resultsNext.percentage * 100) + "% на " + iteration + " прохрде");
            }

            iteration++;
        }
        System.out.println(
                (sum * 100) / iteration + " средний балл на "
                        + iteration + " итераций"
        );
    }
}
