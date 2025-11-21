# Nonverbal Query Based Scene Detection

멀티모달 정보(비디오, 오디오, 텍스트)를 활용하여 사용자의 질의(Query)에 맞는 비디오 **장면(Scene)** 을 탐색하는 프로젝트입니다.  
본 레포지토리는 다음과 같은 세부 단계를 포함합니다.

- **Shot / Scene Detection**: ILSD 기반 샷 검출 및 Inception v3 특징 추출을 통한 씬 경계 탐지
- **MMV 기반 멀티모달 표현 학습**: QVHighlights 데이터셋을 이용해 비디오·오디오·텍스트 공통 임베딩 공간 학습
- **Query-based Retrieval**: 사용자의 자연어 질의에 따라 관련 장면을 순위화하여 반환

---

## Table of Contents

1. [Introduction](#1-introduction)
   - [Pipeline](#11-pipeline)
   - [Datasets](#12-datasets)
2. [Shot & Scene Detection](#2-shot--scene-detection)
   - [Pipeline](#21-pipeline)
   - [Configuration](#22-configuration)
   - [Processing Steps](#23-processing-steps)
3. [MMV (Multimodal Video)](#3-mmv)
   - [Data Setup](#31-data-setup)
   - [Data Preprocessing](#32-data-preprocessing)
   - [Training](#33-학습-모델-훈련-train-model)
   - [Inference](#34-추론inference)

---

## 1. Introduction

### 1.1. Pipeline

본 프로젝트의 전체 파이프라인은 다음과 같습니다.

![Pipeline](/img/pipline.png)

1. **Raw Video 입력**
2. **ILSD Shot Detection** 으로 샷 경계 검출
3. **Inception v3** 를 이용한 샷별 프레임 특징 추출
4. 샷 단위 특징을 이용한 **Scene Boundary Detection**
5. Scene 단위로 잘린 비디오 클립에 대해 **MMV 모델** 학습 및 추론
6. 최종적으로 **텍스트 질의에 기반한 장면 검색 결과** 제공

---

### 1.2. Datasets

본 프로젝트에서는 서로 다른 특성을 가진 여러 비디오 데이터셋을 사용합니다.

#### (1) OVSD Dataset

![OVSD](/img/dataset_ovsd.png)

- 샷 및 씬 구조를 실험하기 위해 사용되는 비디오 데이터셋
- 다양한 장면 전환(컷, 페이드 등)을 포함

#### (2) QVHighlights Dataset

![QVH](/img/dataset_qvh.png)

**논문**: *Detecting Moments and Highlights in Videos via Natural Language Queries* 에서 공개한 오픈 데이터셋 QVHighlights를 사용합니다.

- Query: **10,310**
- Video: **10,148**
- Moment: **18,367**
- Category: 일상 브이로그, 여행, 뉴스 이벤트 등 다양한 상황 포함
- 각 비디오 구간과 해당 구간을 설명하는 **텍스트 질의가 짝을 이뤄(annotation)** 제공되므로,  
  멀티모달 공통 임베딩 공간(projection head)을 학습하기에 적합합니다.

#### (3) YouTube Dataset

![YouTube](/img/dataset_youtube.png)

- 추가적인 일반 도메인 비디오 데이터를 통해
  - Scene Detection 일반화
  - Query 기반 검색 성능 향상
- 실제 서비스 환경에 가까운 다양한 컨텐츠 포함

---

## 2. Shot & Scene Detection

### 2.1. Pipeline

Shot / Scene Detection 단계는 다음 순서로 진행됩니다.

1. **샷 경계 검출 (ILSD)**
   - Raw videos → **ILSD Shot Detection**
   - 샷 경계 정보가 담긴 **텍스트 파일**(예: `shot_boundaries.txt`) 생성

2. **샷 대표 프레임 추출**
   - 각 샷(shot)의 **중간 프레임(center frame)** 을 추출
   - 해당 프레임을 이후 시각 특징 추출에 사용
   - README 원문:  
     > For each video, we perform shot boundary detection and extract a center image for the visual representation fed into the Inception network.

3. **프레임 단위 특징 추출 (Inception v3)**
   - 추출한 중간 프레임 이미지들을 **Inception v3 네트워크**에 입력
   - 각 프레임마다 **2048-D feature vector** 추출
   - 모든 프레임 특징을 모아 **`.h5` 파일**로 저장

4. **Scene Boundary Detection**
   - `.h5` 파일을 입력으로 하여 **Scene Detection 알고리즘(05G 기반)** 수행
   - 최종적으로 **Scene Boundary** 도출
   - (원문 표현: *05G 수행 -> 최종 SCENE Boundary 추출*)

5. **클립 단위 EDA 및 최종 Scene 클립 구성**
   - 각 비디오 클립의 메타데이터 (예: 프레임 수, 길이 등)를 수집
   - EDA를 통해 **클립별로 사용할 프레임 수**를 결정
   - Scene boundary 정보를 바탕으로 원본 비디오를 **Scene 단위로 자르기**

6. **최종 결과**
   - 비디오마다 **Scene 단위 클립**과 해당 boundary 정보가 정리됨
   - 이후 MMV 모듈의 입력으로 사용

예시 이미지:

![Shot & Scene Detection Example 1](/img/example_shot_scene_detection_1.png)  
![Shot & Scene Detection Example 2](/img/example_shot_scene_detection_2.png)

---

### 2.2. Configuration

Shot / Scene Detection 모듈을 사용할 때 필요한 기본 설정입니다.

- `ShotDetector/Debug` 디렉토리 안에 **video 폴더**가 존재해야 합니다.
- 파라미터(예: config 파일 또는 스크립트 인자)의 **video 경로**를 실제 비디오가 위치한 폴더로 지정해야 합니다.

---

### 2.3. Processing Steps

정리하면 Shot / Scene Detection은 다음 세 단계로 구성됩니다.

#### 1) Shot Detection 결과 전처리

1. `video` → **Shot Detection** → `middle_shot.txt` 생성  
2. `middle_shot.txt` 에 기록된 샷별 중간 프레임을 기반으로 Inception v3에 입력  
3. 각 프레임의 특징을 추출하고, 이를 **`.h5` 파일**로 저장

#### 2) Scene Detection 수행

- 입력: `h5` 파일 (특징 벡터)
- 출력: Scene 단위로 구분된 경계 정보가 포함된 `h5` 혹은 별도 결과 파일
- 이 결과를 이용해 **Scene 단위의 비디오 클립**을 정의

#### 3) Scene Detection 이후 Raw Video 자르기

- Scene boundary 정보를 이용하여 **원본 비디오를 Scene 단위로 자르기**
- 잘린 Scene 단위 비디오는 **MMV 모델 입력 전처리용** 클립으로 사용

---

## 3. MMV

MMV(Multimodal Video) 모듈은 비디오, 오디오, 텍스트를 공통 임베딩 공간으로 투영하여,  
사용자의 쿼리(텍스트)에 가장 잘 맞는 비디오/장면을 검색하는 역할을 합니다.

### 3.1. 데이터셋 준비 (Data Setup)

- **QVHighlights** 데이터셋을 사용합니다.
  - 자연어 쿼리와 비디오 장면의 쌍으로 annotation되어 있어,
  - 멀티모달 projection head를 학습하기에 적합합니다.

요약:

- Query 수: **10,310**
- Video 수: **10,148**
- Moment 수: **18,367**
- Category: 일상 브이로그, 여행 블로그, 뉴스 이벤트 등

---

### 3.2. 데이터 전처리 (Data Preprocessing)

#### 1) Video

- MMV 논문에서 제안한 **projection head** 입력 포맷에 맞게 비디오 전처리를 수행합니다.
- QVHighlights 데이터셋의 **raw video**를 사용합니다.
- 본 레포지토리 내 **MMV 모듈**을 불러와 해당 스크립트로 전처리 진행

#### 2) Audio

- 비디오에서 **raw audio**를 추출한 뒤,  
  projection head 입력 포맷에 맞도록 전처리합니다.
- 역시 MMV 모듈의 전처리 함수를 사용합니다.

#### 3) Text

- QVHighlights의 JSON 어노테이션에서 **query와 vid**를 읽어와 dict 형태로 구성합니다.
- 이후 텍스트 전처리를 수행합니다.
  - 소문자 변환 (`lower`)
  - tokenization
  - stop words 제거
  - 길이를 16 토큰 기준으로 padding (`pad to 16`)

#### 4) Feature Extract

- 위에서 정의한 전처리 클래스/함수를 이용해
  - 비디오, 오디오, 텍스트 각각에 대한 **backbone feature**를 추출합니다.
- 결과는 **`.npy`** 등으로 저장합니다.

#### 5) Projection Head용 Feature Dataset 구성

- Backbone에서 추출한 feature vector `.npy` 파일들을  
  **훈련에 사용 가능한 하나의 데이터셋**으로 합칩니다.
- 여러 `.npy` 파일을 **하나의 큰 `numpy array`로 concatenate** 하여  
  학습 시 빠르게 로딩할 수 있도록 구성합니다.

---

### 3.3. 학습 모델 훈련 (Train Model)

- 멀티모달 **공통 임베딩 공간(common projection space)** 에서  
  비디오 / 오디오 / 텍스트를 정렬(alignment)하기 위해,  
  PyTorch 기반의 **projection head (proj\_head)** 를 구현합니다.
- proj\_head 구조는 MMV 논문에서 제안한 구조를 참고하여 설계합니다.
- 학습에는 **NCE Loss (Noise Contrastive Estimation Loss)** 를 사용합니다.
- 정답 레이블은 본 프로젝트에서 제안한 방식에 따라,
  - 같은 인덱스에서 나온 **비디오-오디오-텍스트** 쌍을 **positive pair** 로 사용합니다.

---

### 3.4. 추론(Inference)

- 학습된 MMV 모델을 사용하여 사용자에게 **쿼리 기반 장면 검색 결과**를 제공합니다.
- 전체 흐름:
  1. Scene Detection을 통해 얻은 Scene 단위 비디오 클립에서  
     비디오·오디오 특징 벡터를 추출
  2. 사용자의 **자연어 query**를 입력받아 텍스트 특징 벡터 추출
  3. MMV 모델의 공통 임베딩 공간에서 텍스트와 비디오/오디오 임베딩 간 유사도 계산
  4. 유사도가 높은 순서대로 비디오 또는 Scene 인덱스를 **랭킹 형태로 반환**

- 실험 결과:
  - 비디오-오디오 유사도는 상대적으로 영향이 적었고,
  - 최종 랭킹에서는 **비디오-텍스트 유사도**를 중심으로 반영합니다.
