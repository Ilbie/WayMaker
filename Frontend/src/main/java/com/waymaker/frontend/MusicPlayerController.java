package com.waymaker.frontend;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;

import java.util.List;
import java.util.Map;
import java.util.Timer;
import java.util.TimerTask;

public class MusicPlayerController {

    @FXML
    private Button shuffleButton;

    @FXML
    private Button prevButton;

    @FXML
    private Button nextButton;

    @FXML
    private Button repeatButton;

    @FXML
    private Label currentSongLabel;

    @FXML
    private Button playButton;

    private MP3Player mp3Player;
    private boolean isPlaying = false;
    private List<Map<String, String>> songList;
    private int currentSongIndex = 0;
    private boolean isRepeat = false;
    private boolean isShuffle = false;
    private Timer timer;
    private MBTIFinderController mbtiFinderController;

    public void initialize() {
        mp3Player = new MP3Player();
        timer = new Timer();

        nextButton.setOnAction(event -> playNextSong());
        prevButton.setOnAction(event -> playPreviousSong());
        repeatButton.setOnAction(event -> toggleRepeat());
        shuffleButton.setOnAction(event -> toggleShuffle());
        playButton.setOnAction(event -> togglePlayPause());
    }

    public void setSongList(List<Map<String, String>> songList) {
        this.songList = songList;
    }

    public void setMBTIFinderController(MBTIFinderController controller) {
        this.mbtiFinderController = controller;
    }

    private void playNextSong() {
        if (isShuffle) {
            currentSongIndex = (int) (Math.random() * songList.size());
        } else {
            currentSongIndex = (currentSongIndex + 1) % songList.size();
        }
        playCurrentSong();
    }

    private void playPreviousSong() {
        currentSongIndex = (currentSongIndex - 1 + songList.size()) % songList.size();
        playCurrentSong();
    }

    private void toggleRepeat() {
        isRepeat = !isRepeat;
        repeatButton.setStyle(isRepeat ? "-fx-background-color: lightblue;" : "");
    }

    private void toggleShuffle() {
        isShuffle = !isShuffle;
        shuffleButton.setStyle(isShuffle ? "-fx-background-color: lightblue;" : "");
    }

    private void togglePlayPause() {
        if (isPlaying) {
            mp3Player.pause();
            isPlaying = false;
            playButton.setText("재생");
        } else {
            mp3Player.resume();
            isPlaying = true;
            playButton.setText("일시 정지");
        }
    }

    public void playSong(int index) {
        currentSongIndex = index;
        playCurrentSong();
    }

    private void playCurrentSong() {
        if (songList != null && !songList.isEmpty()) {
            String previewUrl = songList.get(currentSongIndex).get("preview_url");
            if (previewUrl == null) {
                System.err.println("미리보기 URL이 null입니다: " + currentSongIndex);
                return;
            }

            String songTitle = songList.get(currentSongIndex).get("title");
            String songArtist = songList.get(currentSongIndex).get("artist");
            currentSongLabel.setText("현재 재생 중: " + songTitle + " - " + songArtist);
            mp3Player.stop();
            mp3Player.play(previewUrl);
            isPlaying = true;
            playButton.setText("일시 정지");
            mbtiFinderController.updateSongListView(currentSongIndex);
        }
    }

    public int getCurrentSongIndex() {
        return currentSongIndex;
    }
}
