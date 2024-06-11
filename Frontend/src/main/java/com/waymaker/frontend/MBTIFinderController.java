package com.waymaker.frontend;

import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.util.Map;

public class MBTIFinderController {

    @FXML
    private ImageView imageView;

    @FXML
    private Label resultLabel;

    private FileChooser fileChooser;
    private boolean isFileChooserOpen = false;
    private static final String API_URL = "http://localhost:8000/predict/?temperature=1.0";

    public MBTIFinderController() {
        fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg"));
    }

    @FXML
    public void handleUpload(ActionEvent event) {
        if (!isFileChooserOpen) {
            isFileChooserOpen = true;
            Stage stage = (Stage) imageView.getScene().getWindow();
            File file = fileChooser.showOpenDialog(stage);
            isFileChooserOpen = false;

            if (file != null) {
                try {
                    Image image = new Image(file.toURI().toString());
                    imageView.setImage(image);

                    byte[] fileContent = Files.readAllBytes(file.toPath());
                    String response = uploadImage(fileContent, file.getName());
                    displayResult(response);
                } catch (Exception e) {
                    e.printStackTrace();
                    resultLabel.setText("Failed to upload image and get MBTI result.");
                }
            }
        }
    }

    private String uploadImage(byte[] fileContent, String fileName) throws IOException {
        URL url = new URL(API_URL);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setDoOutput(true);
        connection.setRequestMethod("POST");
        String boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
        connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

        try (DataOutputStream request = new DataOutputStream(connection.getOutputStream())) {
            request.writeBytes("--" + boundary + "\r\n");
            request.writeBytes("Content-Disposition: form-data; name=\"file\"; filename=\"" + fileName + "\"\r\n");

            // Content-Type 설정
            if (fileName.toLowerCase().endsWith(".png")) {
                request.writeBytes("Content-Type: image/png\r\n\r\n");
            } else if (fileName.toLowerCase().endsWith(".jpg") || fileName.toLowerCase().endsWith(".jpeg")) {
                request.writeBytes("Content-Type: image/jpeg\r\n\r\n");
            } else {
                request.writeBytes("Content-Type: application/octet-stream\r\n\r\n");
            }

            request.write(fileContent);
            request.writeBytes("\r\n--" + boundary + "--\r\n");
        }

        int responseCode = connection.getResponseCode();
        if (responseCode == HttpURLConnection.HTTP_OK) {
            try (BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()))) {
                String inputLine;
                StringBuilder response = new StringBuilder();
                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                return response.toString();
            }
        } else {
            throw new IOException("Failed to upload image: " + responseCode);
        }
    }

    private void displayResult(String response) throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        Map<String, Object> result = objectMapper.readValue(response, Map.class);
        Map<String, Double> probabilities = (Map<String, Double>) result.get("predicted_probabilities");

        if (probabilities != null && !probabilities.isEmpty()) {
            StringBuilder resultText = new StringBuilder("MBTI Result:\n");
            probabilities.forEach((mbti, probability) ->
                    resultText.append(String.format("%s: %.2f%%\n", mbti, probability))
            );
            resultLabel.setText(resultText.toString());
        } else {
            resultLabel.setText("No MBTI result found.");
        }
    }
}
