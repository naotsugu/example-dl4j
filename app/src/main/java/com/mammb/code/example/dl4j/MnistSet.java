package com.mammb.code.example.dl4j;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

public class MnistSet {

    private static final Logger log = LoggerFactory.getLogger(MnistSet.class);

    private final Mnist mnist;
    private final NativeImageLoader imageLoader;
    private final ImagePreProcessingScaler imageScaler;

    public MnistSet() {
        this.mnist = new Mnist();
        this.imageLoader = new NativeImageLoader(mnist.imgHeight, mnist.imgWidth);
        this.imageScaler = new ImagePreProcessingScaler(0, 1);
    }

    public DataSetIterator iterator() {
        return iterator(mnist.trainingImages(),
                Long.valueOf(mnist.countTrainingImage()).intValue());
    }

    public DataSetIterator iteratorTesting() {
        return iterator(mnist.testingImages(),
                Long.valueOf(mnist.countTestingImage()).intValue());
    }

    public int nIn() {
        return mnist.nIn();
    }

    public int outcomes() {
        return mnist.outcomes;
    }

    public DataSetIterator iterator(Stream<File> images, int samples) {
        final INDArray in  = Nd4j.create(samples, mnist.nIn());
        final INDArray out = Nd4j.create(samples, mnist.outcomes);

        final AtomicInteger index = new AtomicInteger();
        images.forEach(file -> {
            try {
                int n = index.getAndIncrement();

                INDArray img = imageLoader.asRowVector(file);
                imageScaler.transform(img);
                in.putRow(n, img);

                int label = Integer.parseInt(file.toPath().getParent().getFileName().toString());
                out.put(n, label, 1.0); // one-hot

            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        List<DataSet> list = new DataSet(in, out).asList();
        Collections.shuffle(list, new Random(System.currentTimeMillis()));

        int batchSize = 10;
        return new ListDataSetIterator<>(list, batchSize);
    }

    public INDArray as(byte[] bytes) {
        try {
            final byte[] scaled = toGrayScale(bytes);

            ImageIO.write(ImageIO.read(new ByteArrayInputStream(scaled)),
                    "png", new File("image.png"));

            INDArray img = imageLoader.asRowVector(new ByteArrayInputStream(scaled));
            imageScaler.transform(img);

            INDArray in = Nd4j.create(1, mnist.nIn());
            in.putRow(0, img);

            return in;

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public INDArray selectAny() {
        try {
            File selected = mnist.selectAny();
            log.info("selected[{}]", selected);
            INDArray img = imageLoader.asRowVector(selected);
            imageScaler.transform(img);

            INDArray in = Nd4j.create(1, mnist.nIn());
            in.putRow(0, img);

            return in;
        } catch(IOException e) {
            throw new RuntimeException(e);
        }
    }

    private byte[] toGrayScale(byte[] bytes) {

        try (InputStream in = new ByteArrayInputStream(bytes);
             ByteArrayOutputStream out = new ByteArrayOutputStream()) {

            BufferedImage garyScaled = new BufferedImage(
                    mnist.imgWidth, mnist.imgHeight,
                    BufferedImage.TYPE_BYTE_GRAY);

            garyScaled.getGraphics().drawImage(
                    ImageIO.read(in),
                    0, 0,
                    mnist.imgWidth, mnist.imgHeight, null);

            ImageIO.write(garyScaled, "png", out);
            return out.toByteArray();

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

}
