package com.waymaker.frontend;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;

import java.io.File;

public class MBTIFinderController {

    @FXML
    private ImageView imageView;

    @FXML
    private Label resultLabel;

    public void handleUpload(ActionEvent event) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg"));
        File file = fileChooser.showOpenDialog(null);

        if (file != null) {
            Image image = new Image(file.toURI().toString());
            imageView.setImage(image);
            // Dummy MBTI processing logic (to be replaced with actual logic)
            String mbtiResult = processImage(file);
            resultLabel.setText("MBTI Result: " + mbtiResult);
        }
    }

    private String processImage(File file) {
        // Dummy implementation
        return "INFP"; // This should be replaced with actual MBTI detection logic
    }
}
