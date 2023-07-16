# Nonverbal_Query_Based_Scene_Detection

## Introduction
### Model Pipline
![이미지](/img/pipline.png)

### Dataset
![이미지](/img/dataset_ovsd.png)
![이미지](/img/dataset_qvh.png)
![이미지](/img/dataset_youtube.png)


## Shot-Scene Detection
### 순서
1. Raw Videos --> ILSD Shot Detection --> Shot Boundary Return (txt file)
2. Shot boundary 각 shot 별 중간 frame을 가져옴
- For each video, we perform shot boundary detection and extract a center image for the visual representation fed into the Inception network.
![이미지](/img/example_shot_scene_detection.png)
4. 가져온 frame을 가지고 video에서 frame 추출 -> 추출한 frame을 가지고 inception v3 network에 넣는다.
5. 각 frame 별 2048개의 feature 존재 -> hs file로 저장
6. hs file을 이용해서 05G 수행 -> 최종 SCENE Boundary 추촌
6-1. video clip metadata 수집 ( clip 길이(~= frame 수))--> EDA 동해서 전체 최종 clip별 추출할 frame 수 선정.
6-2. Scene boundary 추출한 것을 바탕으로 video clip을 지움
7. Video clip별 scene boundary

### 

## MMV
### 

## Moment-DETR
