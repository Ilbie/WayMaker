package com.waymaker.frontend;

import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.geometry.Pos;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ListCell;
import javafx.scene.control.ListView;
import javafx.scene.control.ProgressBar;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.util.Callback;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class MBTIFinderController {

    @FXML
    private ImageView imageView;

    @FXML
    private VBox resultContainer;

    @FXML
    private ComboBox<String> modelSelector;

    @FXML
    private Label highestMbtiLabel;

    @FXML
    private Label highestMbtiDescription;

    @FXML
    private ListView<Map<String, String>> songListView;

    @FXML
    private VBox musicPlayerContainer;

    private FileChooser fileChooser;
    private boolean isFileChooserOpen = false;
    private MusicPlayerController musicPlayerController;

    public MBTIFinderController() {
        fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("이미지 파일", "*.png", "*.jpg", "*.jpeg"));
    }

    @FXML
    public void initialize() {
        modelSelector.getItems().addAll("일반 모델", "고급 모델");
        modelSelector.setValue("일반 모델");

        // 초기 이미지 설정
        try {
            String initialImagePath = "/com/waymaker/frontend/미리보기.png";
            Image initialImage = new Image(getClass().getResource(initialImagePath).toExternalForm());
            imageView.setImage(initialImage);
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("초기 이미지를 설정할 수 없습니다.");
        }

        // ListView의 셀 팩토리 설정
        songListView.setCellFactory(new Callback<ListView<Map<String, String>>, ListCell<Map<String, String>>>() {
            @Override
            public ListCell<Map<String, String>> call(ListView<Map<String, String>> listView) {
                return new SongListCell();
            }
        });

        // 음악 플레이어 UI 로드
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/com/waymaker/frontend/music_player.fxml"));
            VBox musicPlayer = loader.load();
            musicPlayerController = loader.getController();
            musicPlayerController.setMBTIFinderController(this);
            musicPlayerContainer.getChildren().add(musicPlayer);
        } catch (IOException e) {
            e.printStackTrace();
        }
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
                }
            }
        }
    }

    private String uploadImage(byte[] fileContent, String fileName) throws IOException {
        String apiUrl = "http://localhost:8000/predict/";

        URL url = new URL(apiUrl);
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
            throw new IOException("이미지 업로드 실패: " + responseCode);
        }
    }

    private void displayResult(String response) throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        Map<String, Object> result = objectMapper.readValue(response, Map.class);
        Map<String, Double> probabilities = (Map<String, Double>) result.get("predicted_probabilities");
        List<Map<String, String>> songRecommendations = (List<Map<String, String>>) result.get("song_recommendations");

        if (probabilities != null && !probabilities.isEmpty()) {
            resultContainer.getChildren().clear();
            probabilities = probabilities.entrySet().stream()
                    .sorted((e1, e2) -> Double.compare(e2.getValue(), e1.getValue()))
                    .collect(Collectors.toMap(
                            Map.Entry::getKey,
                            Map.Entry::getValue,
                            (e1, e2) -> e1,
                            java.util.LinkedHashMap::new));

            probabilities.forEach((mbti, probability) -> {
                HBox hbox = new HBox(10);
                hbox.setAlignment(Pos.CENTER);
                Label label = new Label(mbti + ": " + String.format("%.2f%%", probability));
                label.setPrefWidth(100);
                label.setStyle("-fx-text-fill: " + getColorForMbti(mbti) + ";");
                ProgressBar progressBar = new ProgressBar(probability / 100);
                progressBar.setPrefWidth(200);
                progressBar.setStyle("-fx-accent: " + getColorForMbti(mbti) + ";");
                hbox.getChildren().addAll(label, progressBar);
                resultContainer.getChildren().add(hbox);
            });

            // 가장 높은 MBTI 결과를 가져옴
            Map.Entry<String, Double> highestEntry = probabilities.entrySet().iterator().next();
            String highestMbti = highestEntry.getKey();
            double highestProbability = highestEntry.getValue();

            // 부제목과 설명 매핑
            String subtitle;
            String description;
            switch (highestMbti) {
                case "INTP":
                    subtitle = "논리술사";
                    description = "논리술사(INTP)는 독특한 관점과 지성을 가진 성격입니다.";
                    break;
                case "INTJ":
                    subtitle = "전략가";
                    description = "전략가(INTJ)는 이성적이고 두뇌 회전이 빠른 성격입니다.";
                    break;
                case "ENTJ":
                    subtitle = "통솔자";
                    description = "통솔자(ENTJ)는 타고난 리더입니다.";
                    break;
                case "ENTP":
                    subtitle = "변론가";
                    description = "변론가(ENTP)는 논쟁을 즐기는 성격입니다.";
                    break;
                case "INFJ":
                    subtitle = "옹호자";
                    description = "옹호자(INFJ)는 이상주의적이고 원칙주의적입니다.";
                    break;
                case "INFP":
                    subtitle = "중재자";
                    description = "중재자(INFP)는 창의적이고 공감 능력이 높습니다.";
                    break;
                case "ENFJ":
                    subtitle = "선도자";
                    description = "선도자(ENFJ)는 타고난 지도자입니다.";
                    break;
                case "ENFP":
                    subtitle = "활동가";
                    description = "활동가(ENFP)는 낙관적인 태도를 지닌 성격입니다.";
                    break;
                case "ISTJ":
                    subtitle = "현실주의자";
                    description = "현실주의자(ISTJ)는 진솔하고 안정된 성격입니다.";
                    break;
                case "ISFJ":
                    subtitle = "수호자";
                    description = "수호자(ISFJ)는 겸손하고 헌신적인 성격입니다.";
                    break;
                case "ESTJ":
                    subtitle = "경영자";
                    description = "경영자(ESTJ)는 전통과 질서를 중시합니다.";
                    break;
                case "ESFJ":
                    subtitle = "집정관";
                    description = "집정관(ESFJ)은 책임감이 강한 성격입니다.";
                    break;
                case "ISTP":
                    subtitle = "장인";
                    description = "장인(ISTP)은 손기술이 뛰어난 성격입니다.";
                    break;
                case "ISFP":
                    subtitle = "모험가";
                    description = "모험가(ISFP)는 진정한 예술가입니다.";
                    break;
                case "ESTP":
                    subtitle = "사업가";
                    description = "사업가(ESTP)는 영향력을 행사하는 성격입니다.";
                    break;
                case "ESFP":
                    subtitle = "연예인";
                    description = "연예인(ESFP)은 즉흥적이고 즐거움을 주는 성격입니다.";
                    break;
                default:
                    subtitle = "";
                    description = "";
                    break;
            }

            highestMbtiLabel.setText(subtitle + " (" + highestMbti + ")");
            highestMbtiLabel.setStyle("-fx-text-fill: " + getColorForMbti(highestMbti) + "; -fx-font-size: 32px; -fx-font-weight: bold;");
            highestMbtiDescription.setText(description);
            highestMbtiDescription.setStyle("-fx-font-size: 18px;");

            // 추천 노래 목록 표시
            songListView.getItems().clear();
            songListView.getItems().addAll(songRecommendations);

            // 음악 플레이어에 노래 목록 설정
            musicPlayerController.setSongList(songRecommendations);
        }
    }

    public void updateSongListView(int currentSongIndex) {
        final int finalCurrentSongIndex = currentSongIndex;  // 'currentSongIndex'를 effectively final로 만듭니다.
        songListView.setCellFactory(listView -> new SongListCell());
        for (int i = 0; i < songListView.getItems().size(); i++) {
            final int index = i;  // 'i'를 effectively final로 만듭니다.
            ListCell<Map<String, String>> cell = (ListCell<Map<String, String>>) songListView.lookupAll(".cell").stream()
                    .filter(node -> ListCell.class.isAssignableFrom(node.getClass()))
                    .map(node -> (ListCell<Map<String, String>>) node)
                    .filter(listCell -> listCell.getItem() != null && listCell.getItem().equals(songListView.getItems().get(index)))
                    .findFirst().orElse(null);
            if (cell != null) {
                if (index == finalCurrentSongIndex) {
                    cell.setStyle("-fx-background-color: lightblue;");
                } else {
                    cell.setStyle("");
                }
            }
        }
    }

    private String getColorForMbti(String mbti) {
        switch (mbti) {
            case "INTP": return "blue";
            case "INTJ": return "purple";
            case "ENTJ": return "red";
            case "ENTP": return "orange";
            case "INFJ": return "green";
            case "INFP": return "pink";
            case "ENFJ": return "yellow";
            case "ENFP": return "cyan";
            case "ISTJ": return "brown";
            case "ISFJ": return "darkgreen";
            case "ESTJ": return "darkblue";
            case "ESFJ": return "gold";
            case "ISTP": return "gray";
            case "ISFP": return "lightgreen";
            case "ESTP": return "darkred";
            case "ESFP": return "magenta";
            default: return "black";
        }
    }

    private class SongListCell extends ListCell<Map<String, String>> {
        @Override
        protected void updateItem(Map<String, String> song, boolean empty) {
            super.updateItem(song, empty);
            if (empty || song == null) {
                setText(null);
                setGraphic(null);
            } else {
                String songInfo = song.get("title") + " - " + song.get("artist");
                Label songLabel = new Label(songInfo);
                songLabel.setOnMouseClicked(event -> {
                    if (event.getClickCount() == 2) {
                        int index = getIndex();
                        musicPlayerController.playSong(index);
                    }
                });

                HBox hbox = new HBox(10, songLabel);
                hbox.setAlignment(Pos.CENTER_LEFT);
                setGraphic(hbox);

                // 현재 재생 중인 곡은 배경색을 변경
                updateStyle();
            }
        }

        public void updateStyle() {
            if (getIndex() == musicPlayerController.getCurrentSongIndex()) {
                setStyle("-fx-background-color: lightblue;");
            } else {
                setStyle("");
            }
        }
    }
}
