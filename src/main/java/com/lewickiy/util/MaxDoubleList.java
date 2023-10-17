package com.lewickiy.util;

import java.util.Arrays;

public class MaxDoubleList {
    public static double max(double[] numbers) {
        return Arrays.stream(numbers).max().orElse(Double.MAX_VALUE);
    }
}
