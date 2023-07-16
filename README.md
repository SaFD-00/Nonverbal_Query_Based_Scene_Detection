# Nonverbal_Query_Based_Scene_Detection
## 1. Introduction
### (1) 파이프라인
![이미지](/img/pipline.png)
<br>

### (2) Dataset
![이미지](/img/dataset_ovsd.png)
![이미지](/img/dataset_qvh.png)
![이미지](/img/dataset_youtube.png)
<br>

## 2. Shot-Scene Detection
### (1) 파이프라인
1. Raw Videos --> ILSD Shot Detection --> Shot Boundary Return (txt file)
2. Shot boundary 각 shot 별 중간 frame을 가져옴
- For each video, we perform shot boundary detection and extract a center image for the visual representation fed into the Inception network.
![이미지](/img/example_shot_scene_detection_1.png)
4. 가져온 frame을 가지고 video에서 frame 추출 -> 추출한 frame을 가지고 inception v3 network에 넣는다.
5. 각 frame 별 2048개의 feature 존재 -> hs file로 저장
6. hs file을 이용해서 05G 수행 -> 최종 SCENE Boundary 추촌<br>
  6-1. video clip metadata 수집 ( clip 길이(~= frame 수))--> EDA 동해서 전체 최종 clip별 추출할 frame 수 선정.
![이미지](/img/example_shot_scene_detection_2.png)
  6-2. Scene boundary 추출한 것을 바탕으로 video clip을 지움
7. Video clip별 scene boundary
<br>

### (2) 설정사항
- ShotDetector Debug안에 video 폴더가 존재
- 파라미터의 video 경로를 video가 있는 폴더로 지정
<br>

### (3) 과정
#### 1) shot detection 결과 전처리
- video -> shot detection -> middle shot.txt 파일  
- video의 shot별 middle frame을 inception v3 모델에 넣어서 feature 추출  
- Feature 추출 -> h5 파일로 만들어서 저장하기

#### 2) scene detection 수행
- h5 file(x) -> scene detection -> h5 file(x,t)

#### 3) scene detection 후, raw video -> scene 별로 자르기
- mmv model video input 전처리과정
<br>




## MMV
### 







## Moment-DETR
