package com.lewickiy.util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;



public class LoadCSV {
    public static List<String[]> loadCSV(String filename) throws IOException {
        List<String[]> strings = new ArrayList<>();
        FileReader fr = new FileReader(filename);
        BufferedReader br = new BufferedReader(fr);
        String line = br.readLine();
        while (line != null) {
            strings.add(line.split(","));
            line = br.readLine();
        }
        return strings;
    }

    public static double max(double[] numbers) {
        return Arrays.stream(numbers).max().orElse(Double.MAX_VALUE);
    }
}
