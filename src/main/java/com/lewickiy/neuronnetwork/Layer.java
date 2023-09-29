package com.lewickiy.neuronnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

import static com.lewickiy.util.Counter.countLayer;
import static com.lewickiy.util.Counter.countNeuron;
import static com.lewickiy.util.DotProduct.dotProduct;

/**
 * Слой в нашей сети должен поддерживать три элемента состояния:
 * - свои нейроны (neurons);
 * - предшествующий слой (previousLayer);
 * - и выходной кэш (outputCache).
 * Выходной кэш похож на кэш нейрона, но на один слой выше. Он кэширует выходные данные каждого нейрона
 * в данном слое после применения функции активации.
 * В момент создания слоя основной задачей является инициализация его нейронов.
 * Поэтому конструктору класса необходимо знать, сколько нейронов требуется инициализировать,
 * какими должны быть их функции активации и какова скорость обучения.
 * В нашей простой сети у всех нейронов слоя функция активации и скорость обучения одинаковы, что видно из кода.
 */
public class Layer {
    int layerAddress;
    public Optional<Layer> previousLayer;
    public List<Neuron> neurons = new ArrayList<>();
    public double[] outputCache;

    public Layer(Optional<Layer> previousLayer,
                 int numNeurons,
                 double learningRate,
                 DoubleUnaryOperator activationFunction,
                 DoubleUnaryOperator derivativeActivationFunction

    ) {
        layerAddress = countLayer++;
        countNeuron = 1;
        System.out.println(layerAddress + " layer created");
        this.previousLayer = previousLayer;

                Random random = new Random();

                for (int i = 0; i < numNeurons; i++) {
                    double[] randomWeights = null;
                    if (previousLayer.isPresent()) {
                        //TODO build string with random weights of neuron for load to csv file
                        randomWeights = random
                                .doubles(previousLayer.get().neurons.size())
                                .toArray();



                    }
                    Neuron neuron = new Neuron(randomWeights,
                            learningRate,
                            activationFunction,
                            derivativeActivationFunction
                    );
                    neurons.add(neuron);
                }

        outputCache = new double[numNeurons];
    }

    /**
     * По мере того как сигналы передаются через сеть, их должен обрабатывать каждый нейрон Layer (слоя),
     * именно это делает метод. (Каждый нейрон слоя, получает сигналы от каждого нейрона предыдущего слоя)
     *
     * @param inputs - данные полученные от прошлого слоя
     * @return outputCache - выходные данные, являющиеся inputs для следующего слоя
     */
    public double[] outputs(double[] inputs) {
        if (previousLayer.isPresent()) {
            outputCache = neurons.stream().mapToDouble(n -> n.output(inputs)).toArray();
        } else {
            outputCache = inputs;
        }
        return outputCache;
    }

    /**
     * Существует два типа дельт для вычисления в обратном распространении:
     * данный тип для выходного слоя (188,7.4).
     * Впоследствии этот метод будет вызываться сетью во время обратного распространения.
     *
     * @param expected ...
     */
    public void calculateDeltasForOutputLayer(double[] expected) {
        for (int i = 0; i < neurons.size(); i++) {
            neurons.get(i).delta = neurons.get(i).derivativeActivationFunction
                    .applyAsDouble(neurons.get(i).outputCache)
                    * (expected[i] - outputCache[i]);
        }
    }

    /**
     * Данный тип для скрытого слоя (188,7.5).
     * @param nextLayer - в качестве параметра принимается объект "Следующий слой".
     */
    public void calculateDeltasForHiddenLayer(Layer nextLayer) {
        for (int i = 0; i < neurons.size(); i++) {
            int index = i;
            double[] nextWeights = nextLayer.neurons.stream().mapToDouble(n -> n.weights[index]).toArray();
            double[] nextDeltas = nextLayer.neurons.stream().mapToDouble(n -> n.delta).toArray();
            double sumWeightsAndDeltas = dotProduct(nextWeights, nextDeltas);
            neurons.get(i).delta = neurons.get(i).derivativeActivationFunction
                    .applyAsDouble(neurons.get(i).outputCache) * sumWeightsAndDeltas;
        }
    }
}
