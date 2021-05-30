package com.mammb.code.example.dl4j;

import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.ArchiveInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.compress.utils.IOUtils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Mnist {

    private final URL url;

    private final Path baseDir;
    private final Path trainingDir;
    private final Path testingDir;

    final int imgHeight;
    final int imgWidth;
    final int outcomes;

    public Mnist() {
        try {
            this.url = new URL("https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz");
            this.baseDir = Paths.get("./mnist");
            this.trainingDir = baseDir.resolve("mnist_png/training");
            this.testingDir = baseDir.resolve("mnist_png/testing");
            this.imgHeight = 28;
            this.imgWidth  = 28;
            this.outcomes = 10;
        } catch (MalformedURLException e) {
            throw new RuntimeException(e);
        }
    }

    public int nIn() {
        return imgHeight * imgWidth;
    }

    public long countTrainingImage() {
        return images(trainingDir).count();
    }


    public long countTestingImage() {
        return images(testingDir).count();
    }


    public Stream<File> trainingImages() {
        return images(trainingDir);
    }


    public Stream<File> testingImages() {
        return images(testingDir);
    }


    private Stream<File> images(Path path) {
        if (!exists()) {
            fetch();
        }
        return IntStream.range(0, outcomes)
                .mapToObj(Integer::toString)
                .map(path::resolve)
                .map(Path::toFile)
                .map(File::listFiles)
                .flatMap(Stream::of)
                .filter(File::isFile);
    }


    private boolean exists() {
        File file = baseDir.toFile();
        return file.exists() && file.list().length > 0;
    }


    private void fetch() {

        baseDir.toFile().mkdirs();

        try (InputStream gzi = new GzipCompressorInputStream(url.openStream());
             ArchiveInputStream in = new TarArchiveInputStream(gzi)) {

            ArchiveEntry entry;
            while ((entry = in.getNextEntry()) != null) {
                if (!in.canReadEntryData(entry)) {
                    continue;
                }
                File file = baseDir.resolve(entry.getName()).toFile();
                if (entry.isDirectory()) {
                    if (!file.isDirectory() && !file.mkdirs()) {
                        throw new IOException("failed to create directory " + file);
                    }
                } else {
                    File parent = file.getParentFile();
                    if (!parent.isDirectory() && !parent.mkdirs()) {
                        throw new IOException("failed to create directory " + parent);
                    }
                    try (OutputStream o = Files.newOutputStream(file.toPath())) {
                        IOUtils.copy(in, o);
                    }
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public File selectAny() {
        File[] files = testingDir.resolve(
                Integer.toString(new Random().nextInt(10)))
            .toFile()
            .listFiles();
        return files[new Random().nextInt(files.length)];
    }
}
