package com.example.hotdogsru_maven.Service;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtException;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NeiroModel implements AutoCloseable {
    private OrtEnvironment env;
    private OrtSession session;
    private String[] classes;

    public NeiroModel() throws OrtException, IOException {
        // Создаем ONNX Runtime environment
        env = OrtEnvironment.getEnvironment();

        // Загружаем модель ONNX
        session = env.createSession("./src/main/resources/model/best_model.onnx", new OrtSession.SessionOptions());

        // Загружаем список классов
        loadClasses();

        // Выводим информацию о входах модели
        System.out.println("Model inputs:");
        for (var input : session.getInputInfo().entrySet()) {
            System.out.println("  " + input.getKey() + ": " + input.getValue().getInfo());
        }
    }

    public String getAnswer(String path) throws OrtException, IOException {
        // Загружаем и обрабатываем изображение
        float[] imageData = preprocessImageFlat(path);

        // Создаем input tensor
        long[] shape = {1, 224, 224, 3};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imageData), shape);

        // Создаем input map
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input", inputTensor); // Правильное имя входа для нашей модели

        // Запускаем inference
        OrtSession.Result result = session.run(inputs);

        // Получаем результат
        Object outputObj = result.get(0).getValue();
        float[] output;
        if (outputObj instanceof float[][]) {
            output = ((float[][]) outputObj)[0];
        } else if (outputObj instanceof float[]) {
            output = (float[]) outputObj;
        } else if (outputObj instanceof float[][][][]) {
            output = ((float[][][][]) outputObj)[0][0][0];
        } else {
            throw new RuntimeException("Unexpected output type: " + outputObj.getClass());
        }

        // Освобождаем ресурсы
        inputTensor.close();
        result.close();

        // Находим класс с максимальной вероятностью
        int maxIndex = 0;
        float maxValue = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }

        return mapIndexToClass(maxIndex);
    }

    private float[] preprocessImageFlat(String path) throws IOException {
        BufferedImage img = ImageIO.read(new File(path));
        BufferedImage resizedImg = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        resizedImg.getGraphics().drawImage(img.getScaledInstance(224, 224, java.awt.Image.SCALE_SMOOTH), 0, 0, null);

        float[] data = new float[224 * 224 * 3];
        int index = 0;
        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                int rgb = resizedImg.getRGB(x, y);
                data[index++] = ((rgb >> 16) & 0xFF) / 255.0f; // R
                data[index++] = ((rgb >> 8) & 0xFF) / 255.0f;  // G
                data[index++] = (rgb & 0xFF) / 255.0f;         // B
            }
        }
        return data;
    }

    private void loadClasses() throws IOException {
        List<String> classList = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader("./src/main/resources/model/classes.txt"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                // Извлекаем имя породы из формата n02116738-African_hunting_dog
                String[] parts = line.split("-");
                if (parts.length >= 2) {
                    String breedName = parts[1].replace("_", " ");
                    classList.add(breedName);
                }
            }
        }
        classes = classList.toArray(new String[0]);
        System.out.println("Loaded " + classes.length + " dog breeds");
    }

    private String mapIndexToClass(int index) {
        if (index >= 0 && index < classes.length) {
            return classes[index];
        }
        return "Unknown";
    }

    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
        if (env != null) {
            env.close();
        }
    }
}