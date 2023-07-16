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


## 3. MMV
### (1) 데이터셋 준비(Data Setup)
Detecting Moments and Highlights in Videos via Natural Language Queries 논문에서 제공하는 Open Dataset인 QVGHIGHLIGHTS이라는 데이터셋으로 비디오와 비디오 장면의 상황적 특징을 설명하는 텍스트가 짝을 지어 연결되어 annotation되어 이으므로, MMV multimodal proj_head를 학습시키기에 적합하다고 판단되어 사용
- Query : 10,310
- Video : 10,148
- Moment : 18,367
- Category : 일상 블로그, 여행 블로그, 뉴스 이벤트 등

### (2) 데이터 전처리 (Data Preprocessing)
#### 1) Video
- MMV 논문에서 제안한 Proj_head에 들어갈 input_data의 데이터 전처리를 진행한다.
- qvhighlights dataset의 raw video
- MMV module을 불러와 preprocessing을 진행
#### 2) Audio
- MMV 논문에서 제안한 Proj_head에 들어갈 input_data의 데이터 전처리를 진행한다.
- qvhighlights dataset의 raw video에서 추출한 raw audio
- MMV module을 불러와 preprocessing을 진행
#### 3) Text
- MMV 논문에서 제안한 Proj_head에 들어갈 input_data의 데이터 전처리를 진행한다.
- lower, tokenization, rm stop words, pad to 16을 진행
- qvhighlights dataset의 json 파일에서 query와 vid를 dict화 하여 진행
#### 4) Feature Extract
- 앞서 정의한 class와 def을 바탕으로, feature 추출
#### 5) projection_head를 위한 feature_dataset
- Backbone을 통해 추출한 feature vector npy파일들을 훈련에 사용할 수 있는 형태로 바꾸고자 한다.
- 각 npy 파일을 하나의 numpy array로 합쳐준다(concatenate)

### (3) 학습 모델 훈련 (Train Model)
- Common project space에서 proj_head를 학습하기 위해 Torch를 이용하여 모델을 구축하고자 한다.
- proj_head는 mmv 논문에서 제안한 구조를 참고하여 구성함
- proj_head를 NCELoss로 학습하기 위해 Loss 함수를 구축하고자 한다.
- 정답 레이블은 본 프로젝트에서 제안한 같은 인덱스에서 나온 비디오, 오디오, 텍스트의 쌍을 통해 구성한다.

### (4) 추론(Inference)
- 훈련시킨 모델을 통해 순위를 매겨 사용자에게 제공하고자 한다. SceneDetection을 통해 비디오, 오디오 특징 벡터를 추출하고 사용자의 query를 받아 텍스트 특징 벡터를 추출하여 MMV 모델을 통해 사용자의 쿼리를 고려한 영상의 인덱스를 순위로 반환하여 준다.
- 실험 결과를 통해 비디오-오디오는 영향이 적어 비디오-텍스트의 유사도만 반영
