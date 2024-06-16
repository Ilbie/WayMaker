package com.waymaker.frontend;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class MBTIFinderApp extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        FXMLLoader loader = new FXMLLoader(getClass().getResource("/com/waymaker/frontend/main.fxml"));
        Scene scene = new Scene(loader.load(), 800, 1300);
        scene.getStylesheets().add(getClass().getResource("/com/waymaker/frontend/style.css").toExternalForm());
        primaryStage.setScene(scene);
        primaryStage.setTitle("WonderWall");
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
