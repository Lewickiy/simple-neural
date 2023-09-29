package com.lewickiy.neuronnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;

import static com.lewickiy.util.Weight.createLearningWeightsCSV;
import static com.lewickiy.util.Weight.isLearningWeightsExist;

/**
 * Сама сеть хранит только один элемент состояния - слои, которыми она управляет.
 * Данный класс отвечает за инициализацию составляющих его слоёв.
 * Общий тип T связывает сеть с типом окончательных категорий классификации из набора данных.
 * Он используется только в последнем методе класса validate()
 */
public class Network<T>{
    private final List<Layer> layers = new ArrayList<>();

    /**
     * Работая над этой простой сетью, предполагаем, что все слои сети используют
     * одну и ту же функцию активации для своих нейронов и имеют одинаковую скорость обучения.
     * @param layerStructure список элементов типа int (Например {2, 4, 3} описывает сеть,
     *                       имеющую два нейрона во входном слое, четыре нейрона в скрытом,
     *                       и три нейрона на выходном).
     * @param learningRate скорость обучения.
     * @param activationFunction -функция активации
     * @param derivativeActivationFunction ...
     */
    public Network(int[] layerStructure,
                   double learningRate,
                   DoubleUnaryOperator activationFunction,
                   DoubleUnaryOperator derivativeActivationFunction
    ) {

        if (isLearningWeightsExist()) { //Проверка наличия csv с весами (он может быть и пустой. Это не имеет значения)
            createLearningWeightsCSV();
        }
        System.out.println("Network created");

        if (layerStructure.length > 3) {
            throw new IllegalArgumentException("Error: Should be at least 3 layers (1 input, 1 hidden, 1 output)");
        }
        Layer inputLayer = new Layer(
                Optional.empty(),
                layerStructure[0],
                learningRate,
                activationFunction,
                derivativeActivationFunction);
        layers.add(inputLayer); //в сеть добавляем первый слой (входной)

        for (int i = 1; i < layerStructure.length; i++) {
            Layer nextLayer = new Layer(
                    Optional.of(layers.get(i - 1)),
                    layerStructure[i],
                    learningRate,
                    activationFunction,
                    derivativeActivationFunction
            );
            layers.add(nextLayer);
        }
        //TODO save loaded to array weights to CSV
    }

    /**
     * Выходные данные нейронной сети.
     * @param input - входящий сигнал
     * @return - выходные данные - результат обработки сигналов,
     * проходящих через все слои сети.
     */
    private double[] outputs(double[] input) {
        double[] result = input;
        for (Layer layer : layers) {
            result = layer.outputs(result);
        }
        return result;
    }

    /**
     * Данный метод отвечает за вычисление дельт для каждого нейрона в сети.
     * В этом методе последовательно задействуются методы
     * calculateDeltasForOutputLayer() и calculateDeltasForHiddenLayer().
     * Не стоит забывать что при обратном распространении дельты вычисляются в обратном порядке.
     * Данный метод передаёт ожидаемые значения выходных данных для заданного набора входных данных
     * в функцию calculateDeltasForOutputLayer(). Этот метод использует ожидаемые значения, чтобы найти ошибку,
     * с помощью которой вычисляется дельта.
     * @param expected ...
     */
    private void backPropagate(double[] expected) {
        //вычисление дельты для нейронов выходного слоя.
        int lastLayer = layers.size() - 1;
        layers.get(lastLayer).calculateDeltasForOutputLayer(expected);

        //вычисление дельты для скрытых слоёв в обратном порядке
        for (int i = lastLayer - 1; i >= 0; i--) {
            layers.get(i).calculateDeltasForHiddenLayer(layers.get(i + 1));
        }
    }

    /**
     * Так как метод backPropagate() отвечает только за изменение всех дельт, но не изменяет веса элементов сети,
     * после неё вызывается данный метод, поскольку изменение веса зависит от дельт.
     */
    private void updateWeights() {
        for (Layer layer : layers.subList(1, layers.size())) {
            for (Neuron neuron : layer.neurons) {
//                System.out.print(neuron.getNeuronAddress() + " neuron address for UPDATE wright | ");
                for (int i = 0; i < neuron.weights.length; i++) {
                    neuron.weights[i] = neuron.weights[i] + (
                            neuron.learningRate
                                    * layer.previousLayer.get().outputCache[i]
                                    * neuron.delta
                    );
                }
            }
        }
    }

    /**
     * Веса нейронов изменяются в конце каждого этапа обучения.
     * Для этого в сеть должны быть помещены обучающие наборы данных (входные данные и ожидаемые результаты)
     * Данный метод принимает список массивов входных данных и ожидаемых результатов.
     * Каждый набор входных данных пропускается через сеть,
     * после чего её веса обновляются посредством вызова backPropagate() для ожидаемого результата и последующего вызова
     * updateWeights().
     * TODO ввести сюда код, который позволит вывести на печать частоту ошибок когда через сеть проходит обучающий набор данных.
     * TODO Тогда можно будет отследить как постепенно уменьшается количество ошибок сети по
     * TODO мере продвижения вниз по склону в процессе градиентного спуска.
     * @param inputs - принимаемые входные данные
     * @param expected - принимаемые ожидаемые результаты.
     */
    public void train(List<double[]> inputs, List<double[]> expected) {
        //метод использует результаты выполнения функции outputs()
        //для нескольких входных данных, сравнивает их с окончательным результатом
        //и передаёт полученное функциям backPropagate() и updateWeights()
        for (int i = 0; i < inputs.size(); i++) {
            double[] xs = inputs.get(i);
            double[] ys = expected.get(i);
            outputs(xs);
            backPropagate(ys);
            updateWeights();
        }
    }

    public class Results {
        public final int correct;
        public final int trials;
        public final double percentage;

        public Results(int correct, int trials, double percentage) {
            this.correct = correct;
            this.trials = trials;
            this.percentage = percentage;
        }
    }

    public Results validate(List<double[]> inputs, List<T> expectedList, Function<double[], T> interpret) {
        int correct = 0;
        for (int i = 0; i < inputs.size(); i++) {
            double[] input = inputs.get(i);
            T expected = expectedList.get(i);
            T result = interpret.apply(outputs(input));

            if (result.equals(expected)) {
                correct++;
            }
        }
        double percentage = (double) correct / (double) inputs.size();
        return new Results(correct, inputs.size(), percentage);
    }
}
