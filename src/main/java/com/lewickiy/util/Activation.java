package com.lewickiy.util;

/**
 * Функция активации преобразует выходные данные нейрона
 * перед тем как они попадут на следующий слой
 * Функция активации имеет цели:
 * - позволяет нейронной сети представить решения,
 *   которые не являются всего лишь линейным преобразованием
 *   (если сама функция активации сама больше чем просто линейное преобразование)
 * - не позволяет чтобы входные данные каждого нейрона выходили
 *   за пределы заданного диапазона.
 * Функция активации должна иметь вычислимую производную,
 * чтобы её можно было использовать для обратного распространения
 * Популярным множеством функций активации являются сигмоидные функции
 * Результатом сигмоидной функции всегда будет значение в диапазоне от 0.0 да 1.0
 */
public class Activation {
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double derivativeSigmoid(double x) {
        double sig = sigmoid(x);
        return sig * (1.0 - sig);
    }
}
