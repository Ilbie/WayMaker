package com.waymaker;

import javafx.scene.image.Image;

public class MBTIAnalyzer {
    public static String analyze(Image image) {
        // 실제 이미지 분석 로직을 여기에 추가합니다.
        // 예시로 임의의 MBTI 유형을 반환합니다.
        String[] mbtiTypes = {"INTJ", "ENTP", "INFJ", "ENFP", "ISTJ", "ESTJ", "ISFJ", "ESFJ", "INTP", "ENTJ", "INFP", "ENFJ", "ISTP", "ESTP", "ISFP", "ESFP"};
        int randomIndex = (int) (Math.random() * mbtiTypes.length);
        return mbtiTypes[randomIndex];
    }
}

