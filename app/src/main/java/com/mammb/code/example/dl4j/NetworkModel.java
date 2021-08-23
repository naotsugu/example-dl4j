package com.mammb.code.example.dl4j;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.stream.Collectors;

public class NetworkModel {

    private static final Logger log = LoggerFactory.getLogger(NetworkModel.class);
    private static final String MODEL = "model.zip";

    private final MnistSet mnistSet;
    private final MultiLayerNetwork model;

    public NetworkModel(MnistSet mnistSet) {
        this.mnistSet = mnistSet;
        this.model = Paths.get(MODEL).toFile().exists()
                ? restoreModel()
                : buildNetwork(mnistSet.nIn(), mnistSet.outcomes());
    }

    public NetworkModel init() {
        if (Paths.get(MODEL).toFile().exists()) {
            return this;
        }
        int nEpochs = 2; // Number of training epochs
        model.fit(mnistSet.iterator(), nEpochs);
        Evaluation eval = model.evaluate(mnistSet.iteratorTesting());
        System.out.println(eval.stats());
        writeModel();
        return this;
    }

    public String outputAsString(byte[] bytes) {
        INDArray ans = model.output(mnistSet.as(bytes));
        return String.format("{ans:%s, \routput: %s}", ans.argMax(1), ans);
    }

    public void writeModel() {
        try {
            ModelSerializer.writeModel(model, MODEL, true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public MultiLayerNetwork restoreModel() {
        try {
            return ModelSerializer.restoreMultiLayerNetwork(MODEL);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static MultiLayerNetwork buildNetwork(int nIn, int nOut) {

        //create the first, input layer with xavier initialization
        DenseLayer denseLayer = new DenseLayer.Builder()
                .nIn(nIn)
                .nOut(1000)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build();

        // create hidden layer
        OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) // loss function
                .nIn(1000)
                .nOut(nOut)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4)
                .list()
                .layer(denseLayer)
                .layer(outputLayer)
                .backpropType(BackpropType.Standard)
                .build();

//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//            .seed(123)
//            .l2(0.0005)
//            .weightInit(WeightInit.XAVIER)
//            .updater(new Adam(1e-3))
//            .list()
//            .layer(new ConvolutionLayer.Builder(5, 5)
//                .nIn(1)
//                .stride(1, 1)
//                .nOut(20)
//                .activation(Activation.IDENTITY)
//                .build())
//            .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
//                .kernelSize(2, 2)
//                .stride(2, 2)
//                .build())
//            .layer(new ConvolutionLayer.Builder(5, 5)
//                .stride(1,1)
//                .nOut(50)
//                .activation(Activation.IDENTITY)
//                .build())
//            .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
//                .kernelSize(2, 2)
//                .stride(2, 2)
//                .build())
//            .layer(new DenseLayer.Builder().activation(Activation.RELU)
//                .nOut(500).build())
//            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                .nOut(nOut)
//                .activation(Activation.SOFTMAX)
//                .build())
//            .setInputType(InputType.convolutionalFlat(28, 28, 1))
//            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(500));

        return model;
    }

}
