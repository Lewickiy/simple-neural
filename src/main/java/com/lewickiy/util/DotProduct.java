package com.lewickiy.util;

/**
 * Скалярное произведение требуется нам на этапе прямой и обратной связи.Si
 */
public class DotProduct {
    public static double dotProduct(double[] xs, double[] ys) {
        double sum = 0.0;
        for (int i = 0; i < xs.length; i++) {
            sum += xs[i] * ys[i];
        }

        return sum;
    }
}
