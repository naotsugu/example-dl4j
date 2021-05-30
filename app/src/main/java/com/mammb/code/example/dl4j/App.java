package com.mammb.code.example.dl4j;

public class App {

    public static void main(String[] args) {
        var dataSet = new MnistSet();
        var model = new NetworkModel(dataSet).init();
        var server = new Server(model::outputAsString);
        server.start();
        server.browse();
    }

}
