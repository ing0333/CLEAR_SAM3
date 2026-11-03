# CLEAR: SAM 3의 말을 위에서 본 세상에 맞추기

**C**oncept trans**L**ation, **E**nsembling **A**nd **R**ebalancing.
SAM 3를 학습 없이 원격탐사 장면 인식과 토지피복 분할에 맞추는 프롬프트 모듈과, 그 근거가 되는 진단이다.

> **English summary.** SAM 3's presence head is a poor aerial scene classifier out of the box
> (AID 37%). Diagnosis shows the bottleneck is mostly *language*, not vision: presence
> uncertainty is not a useful "look again" trigger (AUROC 0.55), 43% of errors are classes SAM 3
> never grounds under their functional name, and predictions pile onto a few concepts. Three
> training-free modules fix this: **OCT** translates class names into overhead-visible noun
> phrases, **MPV** averages several phrases, and **PPR** rebalances class priors without labels.
> Over 4 datasets (3 overhead, 1 ground-level) × 3 models (SAM 3, CLIP, RemoteCLIP) × 5 splits,
> CLEAR raises mean accuracy from 42.9→71.1 (SAM 3), 66.8→85.3 (CLIP), and 79.4→89.1 (RemoteCLIP).
> It beats the official RS-TransCLIP in 52 of 60 splits and is complementary to it
> (CLEAR + RS-TransCLIP: 74.5 / 89.2 / 91.6). The same translation lifts SAM 3 *segmentation*
> by +11.5 / +8.1 mIoU over official class names on OpenEarthMap / LoveDA. It matches the
> hand-written SegEarth-OV3 name lists on OpenEarthMap and exceeds them by +1.8 on LoveDA.
> A few-shot extension, **PSC**, reaches 85.8% mean with SAM 3. For CLIP models a plain linear
> probe on image embeddings is better.

---

## 한눈에 보기

**장면 분류**: 데이터셋 4개 평균, test 정확도 %, 5개 split 평균

| 모델 | 원래 이름 | + PPR | 공식 RS-TransCLIP | **CLEAR** | **CLEAR + RS-TransCLIP** | CLEAR-FS (클래스당 20장) |
| --- | --- | --- | --- | --- | --- | --- |
| SAM 3 | 42.9 | 57.2 | 51.1 | **71.1** | **74.5** | 85.8 |
| CLIP ViT-L/14 | 66.8 | 74.1 | 76.7 | **85.3** | **89.2** | 90.1 |
| RemoteCLIP ViT-L/14 | 79.4 | 87.4 | 87.4 | **89.1** | **91.6** | 91.5 |

**토지피복 분할**: SAM 3, test mIoU %, 5개 split 평균 ± 표준편차

| 데이터셋 | 원래 이름 | SegEarth-OV3 손작업 목록 | OCT (라벨 없음) | **OCT-S1 (100장으로 선택)** |
| --- | --- | --- | --- | --- |
| OpenEarthMap | 34.0 ± 0.3 | 45.5 ± 0.3 | 44.6 ± 0.3 | **45.5 ± 0.4** |
| LoveDA | 38.9 ± 0.2 | 45.2 ± 0.2 | 44.9 ± 0.3 | **47.0 ± 0.3** |

---

## 1. 문제

SAM 3는 이미지와 짧은 명사구를 받아 그 개념에 해당하는 물체를 모두 찾는다. 점수는 두 항의 곱이다.

```
p(query가 그 개념) = p(query가 그 개념 | 개념이 있음)  ×  p(개념이 사진에 있음)
                     └─ localization (query 200개) ─┘     └──── presence ────┘
```

이 저장소는 presence를 장면 분류기로 썼을 때 무엇이 잘못되는지 진단하고, 고치는 모듈을 만든 뒤, 다른 데이터셋·다른 모델·분할에서도 통하는지 확인한다. 처음 주제는 "presence가 애매할 때만 가까이 다시 보기(look-closer)"였다. 그러나 진단 결과 문제의 중심은 이미지가 아니라 프롬프트(말)였고, 그래서 방향을 바꿨다.

## 2. 진단 (AID, seed 0, val 600 / test 2,400)

**기준선이 재현된다.** presence만으로 37.1%, 둘의 곱으로 39.1%다. 원격탐사 평가 논문의 37.3% / 38.7%와 같다.

**"애매할 때 다시 보기"는 좋은 트리거가 아니다.** 틀린 사진의 대부분은 presence 0.8 이상으로 확신하면서 틀린다.

| 트리거 신호 (test) | AUROC | 사진 30%를 다시 볼 때 잡히는 오류 |
| --- | --- | --- |
| 낮은 확신 (원래 아이디어) | 0.550 | 31.5% |
| 1·2등 점수 차가 작음 | 0.669 | 35.9% |
| presence 1등 ≠ loc 1등 | 0.672 | 39.6% |
| 무작위 | 0.449 | 26.8% |

**오류 1,516장의 43%는 정답 개념을 아예 보지 못한 경우다.** 이 사진들은 정답 클래스의 presence가 0.1 미만이다. 32%는 애매하고, 25%는 다른 개념도 실제로 있는 경우다.

**예측이 몇 개 개념으로 쏠린다.** 클래스마다 정답이 80장인데 Parking이 408번, BareLand가 278번 예측되고, 상위 5개 클래스가 예측의 51%를 가져간다. Church, Commercial, School, Viaduct는 한 번도 예측되지 않는다.

**대부분은 말의 문제다.** 정확도 0%였던 클래스의 이름을 위에서 보이는 말로 바꾸면 presence가 살아난다.

| 클래스 | 원래 이름 → 정답 사진 평균 presence | 바꾼 표현 → presence / 클래스 정확도 (val) |
| --- | --- | --- |
| Port | "port" 8.0% | "harbor" 82.6% / 80% |
| Playground | "playground" 10.0% | "athletic field" 90.6% / 100% |
| Viaduct | "viaduct" 0.1% | "highway interchange" 88.8% / 95% |
| Commercial | "commercial" 7.5% | "office building" 80.0% / 70% |
| Church | "church" 1.5% | 어떤 표현도 2% 미만 (이미지 쪽 문제) |
| School | "school" 0.8% | 최고 "running track" 15.8%, 정확도 0% |

**실패한 시도: 덮는 면적으로 "장면의 주인공" 고르기.** presence 상위 후보 중 사진을 가장 넓게 덮는 개념을 고르면 모든 규칙에서 정확도가 떨어진다(36.8% → 34.5%). 교회·저장탱크·야구장처럼 장면을 정의하는 물체는 작고, 분할 헤드는 없는 개념도 넓게 칠한다(`experiments/area_verifier.py`).

## 3. 모듈

| 모듈 | 하는 일 | 쓰는 라벨 |
| --- | --- | --- |
| **OCT** Overhead Concept Translation | 클래스 이름을 위에서 보이는 물체·표면 명사구로 바꾼다. 표현은 val 정확도로 고르거나(S1), 라벨 없이 "일부 사진에서만 강하게 반응하는가"로 고른다(S2). | S1: val / S2: 없음 |
| **MPV** Multi-Phrase Voting | 같은 클래스를 표현 r개로 따로 묻고 점수를 평균(또는 최댓값)한다. SAM 3는 짧은 명사구 하나만 받으므로 바깥에서 합친다. | 순위에 val |
| **PPR** Presence Prior Rebalancing | 클래스별 보정값을 더해 test 묶음의 예측이 클래스마다 고르게 나뉘게 한다. 가정은 "클래스 비율이 비슷하다" 하나다. | 없음 (온도 T만 val) |
| **PSC** Part-to-Scene Composer | 모든 표현의 presence를 부분으로 보고 장면마다 가중치를 학습한다(로지스틱 회귀). | val 클래스당 20장 |

- **CLEAR** = OCT + MPV + PPR (학습 없음)
- **CLEAR-FS** = PSC + PPR (few-shot)

분할에서는 OCT와 MPV만 쓴다. PPR의 "클래스 비율이 비슷하다"는 가정은 픽셀 수준에서 성립하지 않는다.

## 4. 실험 설정

**장면 분류 데이터셋**

| 데이터셋 | 시점 | 클래스 | val / test | 표현 |
| --- | --- | --- | --- | --- |
| AID | 위 | 30 | 600 / 2,400 | `prompts/phrases.json` (149개) |
| UC Merced | 위 | 21 | 420 / 630 | `prompts/phrases_ucm.json` (105개) |
| NWPU-RESISC45 | 위 | 45 | 900 / 900 | `prompts/phrases_resisc45.json` (222개) |
| SUN-AID | **앞** | 26 | 520 / 520 | AID 표현 그대로 (129개) |

- SUN-AID는 SUN397에서 AID와 같은 개념의 장면(church, harbor, stadium 등)을 골라 만든 앞에서 찍은 사진 세트다(`prompts/sun_aid_classes.json`). 질문은 AID와 똑같이 두고 시점만 바꾼다.
- 모델은 SAM 3(presence), CLIP ViT-L/14(OpenAI), RemoteCLIP ViT-L/14다. CLIP 계열은 `"a satellite photo of {}."`(위) / `"a photo of {}."`(앞) 템플릿의 점수를 쓴다.
- 비교 대상은 다음과 같다.
  - **공식 RS-TransCLIP**: [elkhouryk/RS-TransCLIP](https://github.com/elkhouryk/RS-TransCLIP)의 solver를 그대로 쓴다. 이미지 특징에서 클래스 중심을 다시 추정하는 transductive 방법이다.
    - 고친 곳은 두 군데다. 정확도 함수는 numpy 2에서 실패하므로 바꿨다. `initial_predictions` 경로에서는 사전확률과 할당이 같은 텐서를 가리켜 사전확률이 덮어써지므로, 둘을 복사해 분리했다.
    - 고친 뒤 zero-shot 경로와 결과가 정확히 같은지 테스트로 확인했다(`tests/test_benchmark_modules.py`).
    - SAM 3는 이미지 임베딩을 쓰지 않으므로, 모든 표현의 presence 벡터를 특징으로 넣는다.
  - **TransLP / TransCLIP-lite**: RS-TransCLIP을 직접 단순화한 두 버전이다. k-NN 라벨 전파와, 텍스트 사전확률을 둔 EM 중심 재추정이다.
  - **선형 분류기**: CLIP 계열 이미지 특징에 붙인다. 클래스당 20장을 쓴다.
- 규칙
  - 표현·T·r 등 모든 선택은 val에서만 하고, test는 한 번만 본다.
  - seed 0은 원래 split이다. seed 1~4는 같은 사진 묶음을 클래스별로 다시 나눈다(val 개수는 같다). 따라서 표준편차는 split 분산이다.

**토지피복 분할**
- 데이터셋
  - OpenEarthMap val 500장, 8클래스. 라벨 0(unknown)은 무시한다.
  - LoveDA val 1,669장 중 seed 0으로 고른 500장, 7클래스. 라벨 0(no data)은 무시하고, background는 실제 클래스로 둔다.
- split마다 100장으로 표현을 고르고 400장으로 mIoU를 잰다. 이를 5번 반복한다.
- 표현별 점수 지도는 SegEarth-OV3와 같은 방식으로 만든다: `max(인스턴스 마스크 × 점수, semantic head 확률) × presence`.
  - 인스턴스 임계값은 LoveDA 0.5, OpenEarthMap 0.1이다.
  - LoveDA의 background 임계값은 0.5다. 둘 다 SegEarth-OV3 설정값이다.
  - 클래스 지도는 고른 표현들의 최댓값이고, 픽셀 라벨은 argmax다.
  - 이미지는 통째로 넣고(슬라이딩 윈도우 없음), 256×256에서 비교한다.
  - 따라서 수치는 SegEarth-OV3 논문과 직접 비교할 수 없다. **같은 파이프라인에서 프롬프트만 바꾼 비교**다.
- 비교하는 프롬프트 세트는 다음과 같다(`prompts/seg_phrases.json`).
  - **원래 이름**: 데이터셋 공식 클래스 이름
  - **SegEarth-OV3**: 논문 저장소의 손작업 동의어 목록
  - **OCT-all**: 우리 표현 5개 전부의 최댓값(라벨 없음)
  - **OCT-S1**: 클래스당 표현 1개를 선택 이미지로 고른 것
  - **OCT+MPV**: 상위 r개를 합친 것

## 5. 결과

### 5-1. 장면 분류 (test 정확도 %, 5개 split 평균 ± 표준편차)

| 데이터셋 | 모델 | 원래 이름 | + PPR | 공식 RS-TransCLIP | **CLEAR** | CLEAR + RS-TransCLIP | CLEAR-FS | 선형 분류기 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AID | SAM 3 | 37.1 ± 0.2 | 45.0 ± 0.3 | 44.0 ± 0.4 | **59.9 ± 1.8** | 66.6 ± 1.7 | 87.0 ± 0.5 | – |
| | CLIP | 63.8 ± 0.2 | 71.6 ± 0.3 | 77.1 ± 0.3 | **85.0 ± 0.7** | 92.3 ± 0.6 | 92.8 ± 0.2 | 95.5 ± 0.2 |
| | RemoteCLIP | 81.3 ± 0.3 | 90.4 ± 0.2 | 92.5 ± 0.4 | **90.2 ± 1.1** | 93.1 ± 1.1 | 93.7 ± 0.3 | 95.5 ± 0.3 |
| UC Merced | SAM 3 | 41.7 ± 1.1 | 65.4 ± 1.2 | 51.7 ± 0.9 | **80.2 ± 1.2** | 85.5 ± 2.0 | 94.0 ± 0.7 | – |
| | CLIP | 63.9 ± 0.9 | 75.4 ± 0.7 | 77.3 ± 4.8 | **88.7 ± 1.6** | 90.6 ± 2.0 | 93.1 ± 1.1 | 95.5 ± 0.6 |
| | RemoteCLIP | 91.2 ± 0.4 | 96.6 ± 0.3 | 96.3 ± 0.3 | **96.9 ± 0.5** | 97.3 ± 0.1 | 97.0 ± 0.5 | 98.3 ± 0.4 |
| NWPU-RESISC45 | SAM 3 | 37.9 ± 0.3 | 51.8 ± 0.7 | 47.6 ± 1.2 | **66.5 ± 1.0** | 68.0 ± 1.6 | 82.6 ± 1.2 | – |
| | CLIP | 62.0 ± 1.0 | 70.5 ± 0.9 | 71.8 ± 1.3 | **80.0 ± 1.4** | 84.1 ± 1.1 | 85.8 ± 0.8 | 87.9 ± 0.7 |
| | RemoteCLIP | 76.2 ± 0.4 | 85.1 ± 0.5 | 85.2 ± 0.9 | **87.4 ± 0.9** | 90.4 ± 0.7 | 89.2 ± 0.5 | 91.9 ± 0.3 |
| SUN-AID (앞) | SAM 3 | 55.0 ± 0.8 | 66.5 ± 1.6 | 60.8 ± 1.0 | **77.8 ± 0.8** | 77.8 ± 0.8 | 79.8 ± 0.6 | – |
| | CLIP | 77.7 ± 1.1 | 79.0 ± 1.2 | 80.5 ± 0.9 | **87.7 ± 1.3** | 89.9 ± 1.2 | 88.8 ± 0.6 | 90.0 ± 0.3 |
| | RemoteCLIP | 68.8 ± 1.3 | 77.5 ± 1.3 | 75.8 ± 1.2 | **81.7 ± 0.7** | 85.8 ± 0.5 | 86.1 ± 1.5 | 87.8 ± 1.4 |

모든 모듈의 ablation(OCT-S1, OCT-S2, MPV, TransLP, TransCLIP-lite 등)과 val에서 고른 설정은 [`results/seeds.md`](results/seeds.md)에 있다.

**결과 해석**
1. **CLEAR는 모델과 데이터셋을 가리지 않는다.** 12개 조합 전부에서 원래 이름보다 오른다(+5.7 ~ +38.5%p). 원래 점수가 낮을수록 더 많이 오른다(SAM 3 > CLIP > RemoteCLIP). 이는 SAM 3 전용 요령이 아니라 "클래스 이름이 모델이 아는 말이 아니다"와 "예측이 쏠린다"는 공통 문제를 고치는 것이다.
2. **UC Merced와 RESISC45의 표현은 그 데이터셋에 모델을 돌리기 전에 같은 지시문으로 썼다.** 표현을 튜닝한 데이터셋 밖에서도 통한다.
3. **RemoteCLIP은 이미 클래스 이름을 잘 안다.** AID에서는 OCT/MPV가 거의 보탬이 되지 않고(CLEAR 90.2 vs PPR 90.4), 이득의 대부분이 PPR에서 온다.

### 5-2. 공식 RS-TransCLIP과의 관계

| | CLEAR > RS-TransCLIP | CLEAR + RS-TransCLIP > CLEAR |
| --- | --- | --- |
| SAM 3 (20 split) | 20 / 20 | 17 / 20 |
| CLIP (20 split) | 20 / 20 | 20 / 20 |
| RemoteCLIP (20 split) | 12 / 20 | 19 / 20 |

- CLEAR는 60개 split 중 52개에서 공식 RS-TransCLIP보다 높다. 진 8개는 모두 RemoteCLIP이다(AID 5개, UC Merced 2개, RESISC45 1개). 이름이 이미 맞는 모델에서는 이미지 특징의 군집 구조를 쓰는 RS-TransCLIP이 더 낫다.
- 둘은 고치는 것이 다르다. CLEAR는 텍스트 쪽(말, 클래스 사전확률)을 고치고, RS-TransCLIP은 이미지 쪽(특징 공간의 군집)을 쓴다. 그래서 CLEAR의 예측을 RS-TransCLIP의 초기값으로 넣으면 56 / 60 split에서 더 오른다. 학습 없는 설정에서 가장 높은 조합이다.
- SAM 3에서는 이미지 임베딩이 없어서 RS-TransCLIP이 PPR보다 약하다(51.1 vs 57.2).
- 단순화한 TransLP는 거의 효과가 없다(+0.2 ~ +5.8%p). 오류가 "이웃끼리 엇갈림"이 아니라 "클래스 쏠림"이기 때문이다.

### 5-3. 시점: 같은 개념, 같은 질문, 다른 시점

| | AID (위, 26개 공통 개념 포함 30클래스) | SUN-AID (앞, 26클래스) |
| --- | --- | --- |
| SAM 3 원래 이름 | 37.1 | 55.0 |
| CLIP 원래 이름 | 63.8 | 77.7 |
| RemoteCLIP 원래 이름 | 81.3 | 68.8 |

- SAM 3는 같은 개념이라도 앞에서 찍은 사진에서 훨씬 잘한다. "SAM 3의 약점은 위에서 본 장면이다"라는 진단과 맞는다.
- 반대로 RemoteCLIP은 앞에서 찍은 사진에서 일반 CLIP보다 낮다. 위성 쪽으로 특화한 대가다.
- CLEAR는 위에서 본 모습을 묘사한 AID 표현을 그대로 써도 SUN-AID에서 오른다(SAM 3 +22.8, CLIP +10.0, RemoteCLIP +12.9).

### 5-4. 분할

| 데이터셋 | 원래 이름 | SegEarth-OV3 | OCT-all | **OCT-S1** | OCT+MPV |
| --- | --- | --- | --- | --- | --- |
| OpenEarthMap | 34.0 ± 0.3 | 45.5 ± 0.3 | 44.6 ± 0.3 | **45.5 ± 0.4** | 45.1 ± 0.5 |
| LoveDA | 38.9 ± 0.2 | 45.2 ± 0.2 | 44.9 ± 0.3 | **47.0 ± 0.3** | 46.3 ± 0.5 |

클래스별 IoU에서 이득이 몰리는 곳은 추상적인 이름이다.

| 클래스 | 원래 이름 | SegEarth-OV3 | OCT-S1 | OCT-S1이 주로 고른 표현 |
| --- | --- | --- | --- | --- |
| OEM bareland | 0.4 | 8.7 | 18.2 | sand and gravel surface |
| OEM rangeland | 4.2 | 38.3 | 35.2 | low green vegetation |
| OEM agriculture land | 28.8 | 42.8 | 49.6 | farmland |
| OEM developed space | 13.3 | 26.5 | 17.2 | paved parking lot |
| LoveDA barren | 0.5 | 32.9 | 36.7 | exposed dirt ground |
| LoveDA forest | 22.2 | 32.7 | 40.2 | group of trees |

- 장면 분류에서 찾은 "말의 문제"는 분할에도 그대로 있다. 원래 이름보다 OpenEarthMap +11.5, LoveDA +8.1 mIoU 오른다.
- 사람이 손으로 만든 SegEarth-OV3 목록과는 OpenEarthMap에서 같고, LoveDA에서 +1.8 높다. 표준편차가 0.2~0.4라 LoveDA의 차이는 split을 바꿔도 유지된다.
- 라벨을 전혀 쓰지 않는 OCT-all도 손작업 목록과 0.3~0.9 차이다.
- MPV는 분할에서 추가 이득이 없다. 잘 맞는 표현 하나를 고르는 것이 여러 개를 합치는 것보다 낫다.
- developed space처럼 손작업 이름("pavement")이 더 나은 클래스도 있다.

### 5-5. few-shot

- SAM 3에서는 PSC가 의미 있다. 원래 이름 30개로만 학습한 대조군(AID 78.8)보다 표현 전체를 부분으로 쓴 PSC(86.5)가 높다.
- CLIP 계열에서는 이미지 특징에 선형 분류기를 붙이는 쪽이 같거나 높다. 따라서 PSC의 가치는 SAM 3처럼 쓸 만한 이미지 임베딩이 없는 모델에서 "개념 점수로 장면 만들기"로 쓸 때에 한정된다.

## 6. 기여와 한계

**기여**
- SAM 3를 원격탐사 장면 인식에 쓸 때 병목이 이미지보다 말에 있다는 진단이다. 트리거 실험, 오류 유형, 쏠림, 표현 바꾸기, 같은 개념의 시점 비교로 뒷받침한다.
- 그 진단에서 나온 학습 없는 모듈(OCT, MPV, PPR)을 묶었다. 이 모듈들이 SAM 3뿐 아니라 CLIP 계열, 다른 데이터셋, 앞에서 찍은 사진, 분할까지 통한다는 검증이다.
- 공식 RS-TransCLIP과 경쟁이 아니라 보완 관계라는 결과다.

**이미 알려진 것**
- 기법 하나하나(LLM으로 만든 클래스 설명, 프롬프트 앙상블, 클래스 사전확률 보정, 개념 점수 위의 선형 분류기)는 CLIP 계열에서 알려져 있다. 분할에서 클래스 이름을 동의어 목록으로 바꾸는 것도 SegEarth-OV 계열이 이미 한다.
- 이 저장소의 새로운 점은 SAM 3에 맞춘 적용, 그것이 왜 통하는지에 대한 진단, 그리고 조건을 맞춘 교차 검증이다.

**한계**
- 표준편차는 같은 사진 묶음을 다시 나눈 split 분산이다. 다른 사진을 새로 뽑은 분산은 아니다.
- OCT 표현은 사람과 LLM이 썼다. AID 8개 클래스는 진단 중에 먼저 썼다. 분할용 표현은 SegEarth-OV3 목록을 읽은 뒤에 썼으므로 일부 겹친다.
- PPR은 test 묶음의 클래스 비율이 비슷하다고 가정한다. 비율이 크게 다르면 `balance(..., prior=...)`로 비율을 넣어야 한다.
- RemoteCLIP의 학습 데이터에는 UC Merced·RSICD·RSITMD 캡션이 들어 있다. UC Merced의 RemoteCLIP 점수는 학습 분포와 겹친다.
- 공식 RS-TransCLIP은 논문의 평균 템플릿 대신 이 저장소의 단일 템플릿으로 돌렸다. SAM 3에는 이미지 임베딩 대신 presence 벡터를 넣었다. 따라서 논문 수치와 직접 비교할 수 없다.
- 분할은 val 일부(500장)에서, 256×256 해상도로, 슬라이딩 윈도우 없이 쟀다. OCT-S1은 라벨이 있는 100장을 쓴다.
- SUN-AID는 직접 만든 세트다. AID 30개 중 4개는 맞는 SUN397 장면이 없어서 뺐다.
- Church와 School은 어떤 표현으로도 위에서 본 모습을 찾지 못한다. 이 둘은 말이 아니라 이미지 쪽 한계다.

## 7. 재현 방법 (Colab, GPU)

1. Hugging Face에서 [`facebook/sam3`](https://huggingface.co/facebook/sam3) 접근을 신청하고, Colab Secrets에 `HF_TOKEN`을 등록한다.
2. 설치한다.
   ```bash
   git clone https://github.com/facebookresearch/sam3 && pip install --no-deps -e sam3
   pip install -r requirements.txt
   git clone https://github.com/elkhouryk/RS-TransCLIP      # 공식 비교용
   ```
3. 실행한다. 아래 `WORK`는 점수 캐시를 둘 폴더다.

| 단계 | 명령 | L4 기준 시간 |
| --- | --- | --- |
| Part 1: AID 진단과 모듈 | `python experiments/run_pipeline.py --work_dir $WORK --data_dir /content/aid` | 약 3시간 |
| Part 2: 4 데이터셋 × 3 모델 | `python experiments/multi_benchmark.py --work_dir $WORK` | CLIP 40분, SAM 3 약 4시간 |
| 5개 split + 공식 RS-TransCLIP | `python experiments/seeds_transclip.py --work_dir $WORK --rstc_dir RS-TransCLIP` | CPU 약 30분 (점수 캐시 사용) |
| Part 3: 분할 | `python experiments/seg_benchmark.py --work_dir $WORK --stage all` | 데이터셋당 약 20분 |
| 표 만들기 | `python experiments/make_seed_report.py results/seeds.json results/seg_results.json` | 수 초 |

- Colab에서는 `notebooks/colab_demo.ipynb`(Part 1)와 `notebooks/benchmarks.ipynb`(Part 2, 3)를 위에서부터 실행하면 된다.
- 점수는 모두 캐시되므로 셀이 끊겨도 이어서 돌아간다. 캐시가 있으면 모델을 불러오지 않고 CPU로 평가만 다시 한다.
- CPU 여러 개로 평가를 병렬로 돌릴 때는 `OMP_NUM_THREADS`를 나눠서 줘야 한다. 그렇지 않으면 numpy가 코어를 서로 빼앗아 크게 느려진다.
- CPU 테스트는 `pytest -q tests`로 돌린다. RS-TransCLIP 동등성 테스트는 `RSTC_DIR`에 클론이 있을 때만 돈다.
- Part 1 숫자는 개발 노트북(셀 단위로 캐시하며 실행)에서 나왔다. `run_pipeline.py`는 같은 절차를 한 번에 돌리지만, bf16 계산의 작은 차이 때문에 소수점 아래가 조금 다를 수 있다.

## 8. 저장소 구조

```
sam3_rs/
  scoring.py              SAM 3 presence·loc 분리 추출 (프롬프트 여러 개를 한 번에 디코딩)
  segmentation.py         SAM 3 표현별 분할 지도, mIoU, 표현 선택
  clip_scoring.py         CLIP / RemoteCLIP 텍스트·이미지 특징
  data.py, benchmarks.py  AID 로딩, UC Merced·RESISC45·SUN-AID 준비
  concept_translation.py  OCT (S1, S2)
  ensemble.py             MPV
  balance.py              PPR
  composition.py          PSC
  transductive.py         TransLP, TransCLIP-lite
  metrics.py              정확도, 오류 유형, 트리거 신호
prompts/
  phrases.json                    AID 표현 (SUN-AID도 사용)
  phrases_ucm.json, phrases_resisc45.json
  sun_aid_classes.json            AID 개념 → SUN397 장면
  seg_phrases.json                분할용 표현 + SegEarth-OV3 목록
experiments/
  run_pipeline.py         Part 1 재현
  area_verifier.py        실패한 "주인공" 확인 실험
  multi_benchmark.py      Part 2 (seed 0)
  seeds_transclip.py      5개 split + 공식 RS-TransCLIP
  seg_benchmark.py        Part 3 분할
  make_report.py, make_seed_report.py
notebooks/                colab_demo.ipynb (Part 1), benchmarks.ipynb (Part 2, 3)
results/
  notebook_results.md     Part 1 상세
  multi_benchmark.{json,md}  Part 2 seed 0
  seeds.{json,md}         Part 2 5개 split + RS-TransCLIP
  seg_results.json        Part 3
tests/                    합성 데이터로 하는 CPU 테스트
```

## 9. 참고

- SAM 3: Segment Anything with Concepts, [arXiv:2511.16719](https://arxiv.org/abs/2511.16719) · [code](https://github.com/facebookresearch/sam3)
- Promptable Concept Segmentation from Above: Evaluating SAM 3 in Remote Sensing, [arXiv:2607.09583](https://arxiv.org/abs/2607.09583)
- SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote Sensing Images, [arXiv:2512.08730](https://arxiv.org/abs/2512.08730) · [code](https://github.com/earth-insights/SegEarth-OV-3)
- RS-TransCLIP: Enhancing Remote Sensing Vision-Language Models for Zero-Shot Scene Classification, ICASSP 2025, [arXiv:2409.00698](https://arxiv.org/abs/2409.00698) · [code](https://github.com/elkhouryk/RS-TransCLIP)
- RemoteCLIP: [arXiv:2306.11029](https://arxiv.org/abs/2306.11029) · CLIP: Radford et al., 2021
- 데이터셋: AID (Xia et al., TGRS 2017), UC Merced Land Use (Yang & Newsam, 2010), NWPU-RESISC45 (Cheng et al., Proc. IEEE 2017), SUN397 (Xiao et al., CVPR 2010), OpenEarthMap (Xia et al., WACV 2023), LoveDA (Wang et al., NeurIPS 2021 D&B)
- 관련: CuPL, DCLIP, Waffle (LLM 기반 클래스 설명) · Meta-prompting for zero-shot RS scene classification (ISPRS 2026)

## 라이선스

이 저장소의 코드는 MIT 라이선스다. SAM 3 코드와 체크포인트는 Meta의 SAM License를 따르며 이 저장소에 들어 있지 않다. 연구 결과를 발표할 때는 SAM 사용을 밝혀야 한다. RS-TransCLIP(MIT)은 따로 클론해서 쓰며 이 저장소에 포함하지 않는다. 데이터셋 이미지는 포함하지 않으며, 각 데이터셋의 라이선스를 따른다.
