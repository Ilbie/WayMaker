package com.waymaker;

import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;

import java.io.File;

public class Controller {
    @FXML
    private ImageView imageView;

    @FXML
    private Label mbtiLabel;

    @FXML
    private Label rankLabel;

    @FXML
    private void handleUpload() {
        FileChooser fileChooser = new FileChooser();
        File file = fileChooser.showOpenDialog(null);
        if (file != null) {
            Image image = new Image(file.toURI().toString());
            imageView.setImage(image);
            analyzeImage(image);
        }
    }

    private void analyzeImage(Image image) {
        // 여기에 이미지 분석 로직을 추가합니다.
        // 예시로 MBTI 유형을 랜덤으로 설정합니다.
        String mbtiType = MBTIAnalyzer.analyze(image);
        mbtiLabel.setText("MBTI Type: " + mbtiType);
        rankLabel.setText("Rank: " + (int)(Math.random() * 100 + 1));
    }
}

