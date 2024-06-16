package com.waymaker.frontend;

import javazoom.jl.decoder.JavaLayerException;
import javazoom.jl.player.advanced.AdvancedPlayer;

import java.io.BufferedInputStream;
import java.io.InputStream;
import java.net.URL;

public class MP3Player {

    private AdvancedPlayer player;
    private Thread playbackThread;
    private boolean isPaused;
    private String currentUrl;

    public void play(String url) {
        try {
            currentUrl = url;
            InputStream inputStream = new BufferedInputStream(new URL(url).openStream());
            player = new AdvancedPlayer(inputStream);

            playbackThread = new Thread(() -> {
                try {
                    player.play();
                } catch (JavaLayerException e) {
                    e.printStackTrace();
                }
            });

            playbackThread.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void stop() {
        if (player != null) {
            player.close();
        }
        if (playbackThread != null && playbackThread.isAlive()) {
            playbackThread.interrupt();
        }
    }

    public void pause() {
        if (player != null) {
            player.close();
            isPaused = true;
        }
    }

    public void resume() {
        if (isPaused) {
            play(currentUrl);
            isPaused = false;
        }
    }

    public double getProgress() {
        // Implement logic to get the current progress of the playback if possible
        return 0;
    }

    public int getTotalDuration() {
        // Implement logic to get the total duration of the playback if possible
        return 0;
    }
}
