package com.example.hotdogsru_maven;

import com.example.hotdogsru_maven.Service.NeiroModel;
import ai.onnxruntime.OrtException;

import java.io.IOException;

public class ModelTest {
    public static void main(String[] args) {
        try {
            System.out.println("Testing ONNX model loading...");
            NeiroModel model = new NeiroModel();
            System.out.println("Model loaded successfully!");

            // Test with a sample image if available
            String testImagePath = "./src/main/resources/images/1.jpg";
            if (new java.io.File(testImagePath).exists()) {
                System.out.println("Testing prediction...");
                String result = model.getAnswer(testImagePath);
                System.out.println("Prediction result: " + result);
            } else {
                System.out.println("Test image not found at: " + testImagePath);
            }

            model.close();
            System.out.println("Test completed successfully!");

        } catch (OrtException | IOException e) {
            System.err.println("Error during model test: " + e.getMessage());
            e.printStackTrace();
        }
    }
}