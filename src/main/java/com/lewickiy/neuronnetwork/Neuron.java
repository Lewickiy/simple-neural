package com.lewickiy.neuronnetwork;
import java.util.function.DoubleUnaryOperator;

import static com.lewickiy.util.Counter.countLayer;
import static com.lewickiy.util.Counter.countNeuron;
import static com.lewickiy.util.DotProduct.dotProduct;

/**
 * Каждый нейрон должен хранить в себе множество элементов состояния:
 *  - вес;
 *  - дельту;
 *  - скорость обучения;
 *  - кэш последних выходных данных;
 *  - функцию активации и её производную.
 * В обучающем материале сказано, что некоторые из этих объектов
 * лучше хранить в слое, но для наглядности они отнесены к нейрону.
 */
public class Neuron {
    public String neuronAddress;
    public double[] weights;
    public double delta;
    public final double learningRate;
    public double outputCache;
    public final DoubleUnaryOperator activationFunction;
    public final DoubleUnaryOperator derivativeActivationFunction;

    /**
     * Большинство параметров инициализируются в конструкторе.
     * @param weights ...
     * @param learningRate - выглядит предустановленным, но всё же
     *                     есть причина сделать его изменяемым.
     *                     Если класс будет использоваться для других типов нейронных сетей,
     *                     то, возможно, значение будет изменяться в процессе выполнения программы.
     *                     Поэтому его можно настраивать для максимальной гибкости.
     *                     Существуют нейронные сети которые изменяют скорость обучения по мере
     *                     приближения к решению и автоматически пробуют разные функции активации.
     *                     Поскольку у нас данная переменная final она не может быть изменена
     *                     в середине потока, но чтобы сделать её не окончательной, просто меняем код.
     * @param activationFunction - выглядит предустановленным, но всё же
     *                           есть причина сделать его изменяемым.
     *                           Если класс будет использоваться для других типов нейронных сетей,
     *                           то, возможно, значение будет изменяться в процессе выполнения программы.
     *                           Поэтому его можно настраивать для максимальной гибкости.
     *                           Существуют нейронные сети которые изменяют скорость обучения по мере
     *                           приближения к решению и автоматически пробуют разные функции активации.
     *                           Поскольку у нас данная переменная final она не может быть изменена
     *                           в середине потока, но чтобы сделать её не окончательной, просто меняем код.
     * @param derivativeActivationFunction - выглядит предустановленным, но всё же
     *                           есть причина сделать его изменяемым.
     *                           Если класс будет использоваться для других типов нейронных сетей,
     *                           то, возможно, значение будет изменяться в процессе выполнения программы.
     *                           Поэтому его можно настраивать для максимальной гибкости.
     *                           Существуют нейронные сети которые изменяют скорость обучения по мере
     *                           приближения к решению и автоматически пробуют разные функции активации.
     *                           Поскольку у нас данная переменная final она не может быть изменена
     *                           в середине потока, но чтобы сделать её не окончательной, просто меняем код.
     */
    public Neuron(double[] weights
            , double learningRate
            , DoubleUnaryOperator activationFunction
            , DoubleUnaryOperator derivativeActivationFunction
    ) {
        neuronAddress = countNeuron + "-" + (countLayer - 1);
        countNeuron++; //TODO count neuron
        System.out.println(neuronAddress + " neuron created!");

        this.weights = weights;
        this.delta = 0.0;
        this.learningRate = learningRate;
        this.outputCache = 0.0;
        this.activationFunction = activationFunction;
        this.derivativeActivationFunction = derivativeActivationFunction;
    }

    public String getNeuronAddress() {
        return neuronAddress;
    }

    public void setNeuronAddress(String neuronAddress) {
        this.neuronAddress = neuronAddress;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getOutputCache() {
        return outputCache;
    }

    public void setOutputCache(double outputCache) {
        this.outputCache = outputCache;
    }

    public DoubleUnaryOperator getActivationFunction() {
        return activationFunction;
    }

    public DoubleUnaryOperator getDerivativeActivationFunction() {
        return derivativeActivationFunction;
    }

    /**
     * Единственный метод класса Neuron
     * @param inputs - принимает входные сигналы (входные данные) и применяет к ним формулу (DotProduct)
     *               Входные сигналы объединяются с весами посредством
     *               скалярного произведения и результат кэшируется в outputCache.
     *               Это значение, которое было получено до того как была задействована функция активации,
     *               используется для вычисления дельты.
     * @return - Прежде чем сигнал будет отправлен на следующий слой, к нему применяется функция активации.
     */
    public double output(double[] inputs) {
        outputCache = dotProduct(inputs, weights);
        return activationFunction.applyAsDouble(dotProduct(inputs, weights));
    }
}
