package com.mammb.code.example.dl4j;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.Desktop;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.Objects;
import java.util.function.Function;

public class Server {

    private static final Logger log = LoggerFactory.getLogger(Server.class);

    private final HttpServer server;
    private final String contextRoot;
    private final int port;
    private final Function<byte[], String> fn;

    private Server(String contextRoot, int port, Function<byte[], String> fn) {
        try {
            this.contextRoot = contextRoot;
            this.port = port;
            this.fn = fn;
            this.server = HttpServer.create(new InetSocketAddress(port), 0);
            this.server.createContext(contextRoot, this::handle);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public Server(Function<byte[], String> fn) {
        this("/dl4j", 8080, fn);
    }

    public void start() {
        server.start();
    }

    public void stop() {
        if (Objects.nonNull(server)) {
            server.stop(0);
        }
    }

    void handle(HttpExchange exchange) {
        String path = exchange.getRequestURI().normalize().getPath();
        if (path.endsWith(".html")) {
            writePage("/index.html", exchange);
        } else {
            try (InputStream in = exchange.getRequestBody()) {
                var body = new String(in.readAllBytes(), StandardCharsets.UTF_8)
                        .replace("\"data:image/png;base64,", "")
                        .replace("\"", "");
                byte[] bytes = Base64.getDecoder().decode(body);
                String res = fn.apply(bytes);
                writeJson(res, exchange);
            } catch (Exception e) {
                log.error(e.getMessage(), e);
                write(500, e.getMessage().getBytes(), "text/html", exchange);
            }
        }
    }

    private void writeJson(String res, HttpExchange exchange) {
        write(200, res.getBytes(StandardCharsets.UTF_8), "application/json", exchange);
    }

    private void writePage(String path, HttpExchange exchange) {
        try (InputStream is = getClass().getResourceAsStream(path)) {
            write(200, is.readAllBytes(), "text/html", exchange);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void write(int rCode, byte[] bytes, String contentType, HttpExchange exchange) {
        try (OutputStream os = exchange.getResponseBody()) {
            var header = String.format("%s; charset=%s", contentType, StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", header);
            exchange.sendResponseHeaders(rCode, bytes.length);
            os.write(bytes);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void browse() {
        try {
            Object headless = System.getProperties().getOrDefault("java.awt.headless", false);
            if (headless.toString().equals("true")) {
                return;
            }
            var url = String.format("http://localhost:%s/%s/index.html",
                    port,
                    contextRoot.replaceFirst("/", ""));
            Desktop.getDesktop().browse(new URI(url));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

}
