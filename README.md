# CLEAR: SAM 3로 위성 장면 알아보기

**C**oncept trans**L**ation, **E**nsembling **A**nd **R**ebalancing — SAM 3를 학습 없이 위성/항공 장면 인식에 맞추는 모듈 3개와, few-shot 확장 모듈 1개.

> **English summary.** SAM 3's presence head ("is concept X anywhere in the image?") is a poor
> aerial scene classifier out of the box (AID: 37%, reproducing prior work). We diagnose why:
> presence uncertainty is *not* a useful "look again" trigger (AUROC 0.55), 43% of errors are
> concepts SAM 3 does not ground under their class name, and predictions pile up on concepts
> that appear in many scenes (Parking predicted 408x for 80 true images). Three training-free
> modules — **OCT** (translate function names into overhead-visible noun phrases), **MPV**
> (average several phrases), **PPR** (label-free class-prior rebalancing) — raise AID test
> accuracy from **36.8% to 58.7%**. A few-shot extension, **PSC** (linear "parts-to-scene"
> classifier over phrase presences, 20 labels/class), reaches **86.1%**; a control without parts
> already gives 77.1%, so parts add +8.3%p and partly encode context. A coverage-based
> "main-subject verifier" is reported as a negative result.
>
> **Part 2 — does it transfer?** On 4 datasets (AID, UC Merced, NWPU-RESISC45 from above;
> SUN-AID, 26 AID concepts photographed from the ground) × 3 models (SAM 3, CLIP ViT-L/14,
> RemoteCLIP ViT-L/14), CLEAR improves every one of the 12 settings, by +5.6 to +37.3%p
> (mean: SAM 3 43.1→70.7, CLIP 66.9→85.5, RemoteCLIP 79.2→89.1). Label-free PPR beats two
> simplified RS-TransCLIP-style transductive baselines (k-NN label propagation; EM prototype
> refinement) in 10 of 12 settings, consistent with the errors being a class-prior skew rather
> than noisy neighbours. For CLIP models, the few-shot PSC is close to but below a linear probe
> on image embeddings, so PSC's value is specific to models without a usable image embedding.

---

## 1. 무엇을 했나

SAM 3는 이미지와 짧은 명사구를 받아, 그 개념에 해당하는 물체를 전부 찾아요. 내부에서는 두 점수를 곱해요.

```
p(query가 그 개념) = p(query가 그 개념 | 개념이 있음)  ×  p(개념이 사진에 있음)
                     └─ localization (query 200개) ─┘     └─ presence (1개) ─┘
```

이 프로젝트는 presence를 **장면 분류기**로 빌려 썼을 때 무엇이 잘못되는지 진단하고, 그걸 고치는 모듈을 만들었어요. 원래 주제는 "presence가 애매할 때만 가까이 다시 보기(look-closer)"였는데, 진단해 보니 문제의 중심이 이미지가 아니라 **말(프롬프트)** 쪽이었어요. 그래서 방향을 바꿨어요.

**평가 설정**
- 데이터: AID. 30개 클래스에서 클래스당 100장씩 seed 0으로 뽑았어요. 클래스당 20장은 val, 80장은 test예요(val 600 / test 2,400).
- 판정: 클래스마다 프롬프트를 넣고, 점수가 가장 높은 클래스를 답으로 봐요.
- 규칙: 모든 선택(표현, T, r, 정규화 강도)은 val에서만 하고, test는 마지막에 한 번만 봤어요.

## 2. 진단

### 2-1. 기준선 재현
| AID 정확도 | 원격탐사 평가 논문 (AID 전체) | 이 저장소 (3,000장) |
| --- | --- | --- |
| presence만 | 37.26% | 37.10% |
| loc만 | 28.25% | 30.27% |
| 둘의 곱 | 38.70% | 39.07% |

강 사진 100장 중 41장만 맞혔고, 31장은 Farmland로 답했어요. 평가 논문에서 보고된 실패가 그대로 재현됐어요.

### 2-2. "presence가 애매할 때 다시 보기"는 좋은 트리거인가? → 아니요
틀린 사진의 대부분은 presence 0.8 이상으로 **확신하면서** 틀렸어요(test에서 1,223장 중 59.5%가 오답).

| 트리거 신호 (test) | AUROC | 사진 30%만 다시 볼 때 잡히는 오류 |
| --- | --- | --- |
| 낮은 확신 (원래 아이디어) | 0.550 | 31.5% |
| 1·2등 점수 차가 작음 | 0.669 | 35.9% |
| presence 1등 ≠ loc 1등 | 0.672 | 39.6% |
| 무작위 | 0.449 | 26.8% |

### 2-3. 오류의 성격 (test 오류 1,516장)
| 유형 | 기준 | 비율 |
| --- | --- | --- |
| B. 정답 개념을 못 봄 | 정답 클래스 presence < 0.1 | 43% |
| C. 애매함 | 0.1 ~ 0.5 | 32% |
| A. 다른 개념도 진짜로 있음 | 정답 클래스 presence ≥ 0.5 | 25% |

### 2-4. 실패한 시도: "장면의 주인공" 확인 담당 (덮는 면적)
presence 상위 후보 중 사진을 가장 넓게 덮는 개념을 고르게 했더니, **모든 규칙에서 정확도가 떨어졌어요.** 예를 들어 presence × 덮는 면적(후보 3개)은 36.8%에서 34.5%가 됐어요(고친 176 / 망친 232). 이유는 두 가지예요.
1. 교회, 저장탱크, 야구장처럼 장면을 정의하는 물체는 작아요.
2. 분할 헤드도 개념이 있을 때만 학습돼서, 없는 개념도 넓게 칠해요.

코드는 `experiments/area_verifier.py`에 있어요.

### 2-5. 프롬프트 문제인가, 이미지 문제인가?
정확도 0%였던 클래스의 이름을 생김새가 드러나는 말로 바꿔봤어요(val).

| 클래스 | 원래 이름 → 정답 사진 평균 presence | 바꾼 표현 → 정답 사진 평균 presence / 클래스 정확도 |
| --- | --- | --- |
| Port | "port" 8.0% | "harbor" 82.6% / 80% |
| Playground | "playground" 10.0% | "athletic field" 90.6% / 100% |
| Viaduct | "viaduct" 0.1% | "highway interchange" 88.8% / 95% |
| Commercial | "commercial" 7.5% | "office building" 80.0% / 70% |
| Church | "church" 1.5% | 어떤 표현도 2% 미만 (이미지 쪽 문제) |
| School | "school" 0.8% | 최고 "running track" 15.8%, 정확도 0% |

→ 대부분은 **기능 이름이 눈에 보이는 것과 연결되지 않는 문제**였어요. 교회와 학교만 위에서 본 모습 자체를 못 알아보는 문제로 남았어요.

## 3. 모듈

| 모듈 | 하는 일 | 라벨 |
| --- | --- | --- |
| **OCT** Overhead Concept Translation | 클래스 이름을 위에서 보이는 물체 명사구로 바꿔요(`prompts/phrases.json`). 표현은 val 정확도로 고르거나(S1), 라벨 없이 "일부 사진에서만 강하게 반응하는가"(대비 점수)로 골라요(S2) | S1: val / S2: 없음 |
| **MPV** Multi-Phrase Voting | 같은 클래스를 표현 r개로 따로 묻고 presence를 평균해요. SAM 3는 짧은 명사구 하나만 받기 때문에 바깥에서 합쳐요 | 순위 매기기에 val |
| **PPR** Presence Prior Rebalancing | 클래스마다 보정값을 더해서, test 묶음의 예측이 클래스별로 고르게 나뉘게 해요. 가정은 "클래스 비율이 비슷하다" 하나예요 | 없음 (T만 val) |
| **PSC** Part-to-Scene Composer | 표현 149개의 presence를 부분으로 보고, 장면마다 가중치를 학습해요(로지스틱 회귀) | val 클래스당 20장 |

**CLEAR** = OCT + MPV + PPR (학습 없음) / **CLEAR-FS** = PSC + PPR (few-shot)

## 4. 결과 (AID test 2,400장)

| 구분 | 방법 | test 정확도 |
| --- | --- | --- |
| 라벨 안 씀 | SAM 3 (원래 클래스 이름) | 36.8% |
| | + PPR | 44.5% |
| | OCT-S2 (라벨 없이 표현 고르기) | 44.8% |
| val 라벨은 고르기에만 | OCT-S1 | 51.4% |
| | OCT-S1 + PPR | 56.3% |
| | **CLEAR (OCT + MPV r=2 + PPR)** | **58.7%** |
| val 라벨로 가중치 학습 (클래스당 20장) | 대조군: 원래 이름 30개로 로지스틱 회귀 | 77.1% |
| | PSC (표현 149개) | 85.4% |
| | **CLEAR-FS (PSC + PPR)** | **86.1%** |

- 쏠림: 원래 이름으로는 Parking이 408번 예측됐고(정답 80장), 상위 5개 클래스가 예측의 51%를 가져갔어요. PPR을 쓰면 26~31%로 줄어요.
- MPV는 혼자 쓰면 44.1%로 OCT-S1보다 낮아요. 클래스마다 점수 크기가 달라서예요. PPR과 같이 써야 올라요.
- PSC 이득의 대부분은 분류기 학습에서 와요(대조군 77.1%). 부분이 더한 건 +8.3%p예요.

## 4-2. Part 2: 다른 데이터셋 · 다른 모델에서도 되나?

AID 하나에서만 확인했다는 한계를 풀려고, 같은 모듈을 **데이터셋 4개 × 모델 3개 = 12개 조합**에 그대로 붙였어요.

| 데이터셋 | 시점 | 클래스 | val / test | 표현 |
| --- | --- | --- | --- | --- |
| AID | 위 | 30 | 600 / 2,400 | `prompts/phrases.json` (149개) |
| UC Merced | 위 | 21 | 420 / 630 | `prompts/phrases_ucm.json` (105개) |
| NWPU-RESISC45 | 위 | 45 | 900 / 900 | `prompts/phrases_resisc45.json` (222개) |
| SUN-AID | **앞** | 26 | 520 / 520 | AID 표현 그대로 (129개) |

- **SUN-AID**는 SUN397에서 AID와 같은 개념의 장면(church, harbor, stadium…)을 골라 만든, 앞에서 찍은 사진 세트예요(`prompts/sun_aid_classes.json`). **질문(표현)은 AID와 똑같이** 두고 시점만 바꿨어요.
- 모델: SAM 3(presence), CLIP ViT-L/14(OpenAI), RemoteCLIP ViT-L/14(원격탐사용 CLIP). CLIP 계열은 "a satellite photo of {}." / "a photo of {}." 템플릿 점수를 그대로 써요.
- 새 비교 기준(RS-TransCLIP을 단순화한 두 가지, 둘 다 test 묶음을 한꺼번에 봐요):
  - **TransLP**: 비슷한 이미지끼리 예측을 나눠 갖게 해요(k-NN 그래프 라벨 전파).
  - **TransCLIP-lite**: 클래스 중심을 이미지 특징으로 다시 잡고, 텍스트 점수를 사전확률로 써요(EM). RS-TransCLIP에 있는 라플라시안 항과 공분산 학습은 없어요.
  - CLIP 계열은 이미지 특징에 **선형 분류기(linear probe)**도 붙였어요. SAM 3는 이미지 임베딩을 쓰지 않아서 이 기준이 없어요.
- 규칙은 Part 1과 같아요. 표현·T·r·α·λ는 val에서만 고르고, test는 한 번만 봤어요.

**학습 없음 (test 정확도 %)**

| 데이터셋 | 모델 | 원래 | +PPR | +TransLP | +TransCLIP-lite | **CLEAR** | 향상 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AID | SAM 3 | 36.8 | 44.5 | 41.4 | 44.7 | **58.7** | +21.8 |
| | CLIP | 63.7 | 71.7 | 66.1 | 70.5 | **85.2** | +21.5 |
| | RemoteCLIP | 81.8 | 90.3 | 86.4 | 88.2 | **91.3** | +9.6 |
| UC Merced | SAM 3 | 42.1 | 64.1 | 44.3 | 59.0 | **79.4** | +37.3 |
| | CLIP | 64.0 | 75.7 | 65.7 | 72.5 | **88.3** | +24.3 |
| | RemoteCLIP | 90.6 | 96.0 | 90.6 | 95.1 | **96.2** | +5.6 |
| NWPU-RESISC45 | SAM 3 | 37.9 | 52.9 | 38.0 | 46.6 | **66.0** | +28.1 |
| | CLIP | 61.1 | 70.1 | 63.4 | 67.9 | **79.9** | +18.8 |
| | RemoteCLIP | 76.7 | 85.0 | 79.0 | 82.9 | **88.4** | +11.8 |
| SUN-AID (앞) | SAM 3 | 55.6 | 68.3 | 56.3 | 61.2 | **78.7** | +23.1 |
| | CLIP | 78.8 | 80.4 | 78.8 | 81.9 | **88.7** | +9.8 |
| | RemoteCLIP | 67.7 | 78.3 | 68.1 | 73.1 | **80.6** | +12.9 |
| **평균** | SAM 3 | 43.1 | 57.4 | 45.0 | 52.9 | **70.7** | +27.6 |
| | CLIP | 66.9 | 74.5 | 68.5 | 73.2 | **85.5** | +18.6 |
| | RemoteCLIP | 79.2 | 87.4 | 81.0 | 84.8 | **89.1** | +9.9 |

**few-shot (val 클래스당 20장)**

| 데이터셋 | 모델 | 대조군(원래 이름 LR) | PSC | **CLEAR-FS** | 선형 분류기(이미지 특징) |
| --- | --- | --- | --- | --- | --- |
| AID | SAM 3 / CLIP / RemoteCLIP | 77.1 / 90.2 / 93.3 | 85.4 / 92.3 / 93.5 | **86.1 / 92.4 / 93.6** | – / 95.5 / 95.8 |
| UC Merced | 〃 | 86.5 / 91.1 / 96.0 | 92.7 / 93.5 / 97.6 | **93.5 / 93.7 / 97.8** | – / 95.2 / 97.8 |
| NWPU-RESISC45 | 〃 | 72.8 / 84.1 / 88.2 | 80.0 / 86.0 / 88.7 | **81.9 / 86.1 / 89.3** | – / 88.3 / 92.1 |
| SUN-AID (앞) | 〃 | 79.0 / 86.9 / 85.2 | 78.8 / 89.6 / 87.9 | **80.4 / 89.6 / 87.9** | – / 89.6 / 86.3 |

전체 ablation과 val에서 고른 설정은 [`results/multi_benchmark.md`](results/multi_benchmark.md)에 있어요.

**읽는 법**
1. **모델을 바꿔도 통해요.** CLEAR는 12개 조합 전부에서 올랐어요(+5.6 ~ +37.3%p). 원래 점수가 낮을수록 많이 올라요(SAM 3 > CLIP > RemoteCLIP). SAM 3 전용 요령이 아니라, "클래스 이름이 모델이 아는 말이 아니다"와 "예측이 일부 클래스로 쏠린다"는 공통 문제를 고치는 거예요.
2. **데이터셋을 바꿔도 통해요.** UC Merced와 RESISC45의 표현은 그 데이터셋 이미지에 모델을 돌려보기 전에 같은 지시문으로 써 둔 거예요.
3. **시점을 바꿔도 통해요.** SUN-AID에서는 위에서 본 모습을 묘사한 AID 표현을 그대로 썼는데도 올랐어요(SAM 3 +23.1, CLIP +9.8). 같은 개념인데도 SAM 3 원래 점수는 앞에서 찍은 사진(55.6)이 위에서 찍은 사진(AID 36.8)보다 훨씬 높아요. **SAM 3의 약점이 "위에서 본 장면"이라는 진단**과 맞아요. 반대로 RemoteCLIP은 앞에서 찍은 사진에서 일반 CLIP보다 낮아요(67.7 vs 78.8). 위성 쪽으로 특화한 대가예요.
4. **PPR이 TransCLIP 계열보다 나아요.** 12개 중 10개에서 PPR > TransCLIP-lite > TransLP였어요(예외: SUN-AID CLIP, AID SAM 3는 TransCLIP-lite가 근소하게 높음). 이웃끼리 예측을 맞추는 TransLP는 거의 효과가 없었어요(+0~4.6%p). 오류의 대부분이 "이웃이 흔들리는 문제"가 아니라 "클래스 쏠림"이라서, 쏠림을 직접 겨냥하는 PPR이 더 잘 먹혀요. CLEAR 위에 TransCLIP-lite를 얹는 것도 대부분 CLEAR보다 낮았어요.
5. **few-shot에서는 SAM 3일 때만 PSC가 의미 있어요.** CLIP 계열은 이미지 특징에 선형 분류기를 붙이는 쪽이 PSC보다 같거나 높아요(SUN-AID RemoteCLIP만 예외). PSC는 SAM 3처럼 쓸 만한 이미지 임베딩이 없는 모델에서 "개념 점수로 장면 만들기"로 쓸 때 가치가 있어요.
6. 솔직한 비교: 학습 없는 설정에서 **SAM 3 + CLEAR(평균 70.7)는 아직 CLIP 원래 점수(66.9)를 겨우 넘는 수준**이고, CLIP + CLEAR(85.5)나 RemoteCLIP(79.2)보다는 낮아요. 장면 분류 자체는 CLIP 계열이 유리하고, SAM 3의 장점은 마스크까지 같이 준다는 데 있어요.

## 5. 한계 (꼭 같이 읽어주세요)
- **표본은 데이터셋마다 1개(seed 0)**이고, test는 클래스당 20~80장이에요. 1~2%p 차이는 표본 오차 안에 있을 수 있어요.
- **OCT 표현은 사람과 LLM이 만들었어요.** AID 8개 클래스는 진단 실험에서 먼저 썼고, 나머지는 고정된 지시문으로 생성했어요(`prompts/phrases*.json`). UC Merced·RESISC45 표현은 모델을 돌려보기 전에 썼지만, 데이터셋 클래스 이름은 보고 썼어요.
- **RemoteCLIP의 학습 데이터에는 UC Merced·RSICD·RSITMD 캡션이 들어 있어요.** 그래서 RemoteCLIP의 UC Merced 점수는 학습 분포와 겹쳐 있어서 높게 나왔을 수 있어요.
- **TransLP · TransCLIP-lite는 RS-TransCLIP을 단순화한 거예요**(라플라시안 항, 공분산 학습 없음). 원 논문의 수치와 직접 비교하면 안 돼요.
- **SUN-AID는 직접 만든 세트예요.** SUN397에서 AID 개념과 맞는 장면 26개를 골랐고(AID 30개 중 4개는 맞는 장면이 없어서 뺐어요), 한 SUN 장면이 AID 개념과 정확히 같지는 않아요.
- **PPR은 test 묶음의 클래스 비율이 비슷하다고 가정해요.** 비율이 크게 다르면 `prior=`로 비율을 넣어야 해요.
- **PSC가 배운 "부분"은 주변 환경인 경우가 있어요.** 예를 들어 Church ← "crowded rooftops", "parked cars"예요. AID 밖에서는 통하지 않을 수 있어요.
- 기법 자체(LLM 프롬프트, 앙상블, 클래스 비율 보정, 개념 점수 위 선형 분류기)는 CLIP 계열에서 알려진 것들이에요. 이 저장소의 기여는 **SAM 3에 맞춘 적용과, 왜 통하는지에 대한 진단**이에요.

## 6. 실행 방법 (Colab, GPU)
1. Hugging Face에서 [`facebook/sam3`](https://huggingface.co/facebook/sam3) 접근을 신청하고, Colab Secrets에 `HF_TOKEN`을 등록해요.
2. `notebooks/colab_demo.ipynb`를 열어 위에서부터 실행해요. 또는 직접 이렇게 실행해요.
```bash
git clone https://github.com/facebookresearch/sam3 && pip install --no-deps -e sam3
pip install -r requirements.txt
python experiments/run_pipeline.py --work_dir /content/drive/MyDrive/SAM3_PGLC --data_dir /content/aid
```
- 1단계(GPU): 표현 149개 × 3,000장의 presence를 계산해 캐시해요. L4에서 약 3시간 걸리고, 끊겨도 이어서 돼요.
- 2단계(CPU, 수 초): 위 표를 전부 다시 계산해서 `results/results.json`에 저장해요.
- CPU 부분 테스트: `pytest -q tests`

**Part 2 (다른 데이터셋 · 모델)** — `notebooks/benchmarks.ipynb`, 또는
```bash
pip install timm ftfy iopath datasets open_clip_torch
python experiments/multi_benchmark.py --work_dir /content/drive/MyDrive/SAM3_PGLC \
    --datasets ucm,sun_aid,resisc45,aid --models openai,remoteclip,sam3
python experiments/make_report.py /content/drive/MyDrive/SAM3_PGLC/bench/multi_benchmark.json
```
- 데이터는 Hugging Face 미러에서 받아요: `blanchon/UC_Merced`, `timm/resisc45`, `tanganke/sun397`(test 7.2GB, 스트리밍), `blanchon/AID`.
- L4 기준 CLIP 두 개는 데이터 준비 포함 약 40분, SAM 3는 UC Merced 40분 · SUN-AID 50분 · RESISC45 2.3시간 걸려요. AID의 SAM 3 점수는 Part 1 캐시를 다시 써요.
- 점수 캐시가 있으면 모델을 불러오지 않고 CPU로 평가만 다시 해요.

이 README의 숫자는 개발 노트북(셀 단위로 캐시하며 실행)에서 나왔어요. `run_pipeline.py`는 같은 절차를 한 번에 돌리지만, bf16 배치 계산의 작은 차이나 S1 선택의 동점 처리 때문에 소수점 아래 숫자는 조금 다를 수 있어요.

## 7. 구조
```
sam3_rs/
  scoring.py              presence·loc 분리 추출 (프롬프트 여러 개를 한 번에 디코딩)
  data.py                 AID 목록·split·병렬 다운로드·이미지 로딩
  concept_translation.py  OCT (S1, S2 선택)
  ensemble.py             MPV
  balance.py              PPR
  composition.py          PSC
  metrics.py              정확도, 고친/망친 수, 오류 유형, 트리거 신호
  benchmarks.py           Part 2 데이터셋 준비 (UC Merced, RESISC45, SUN-AID)
  clip_scoring.py         CLIP / RemoteCLIP 점수·이미지 특징
  transductive.py         비교 기준: TransLP, TransCLIP-lite
prompts/phrases.json      AID 클래스별 표현 + 생성 지시문 (SUN-AID도 이걸 씀)
prompts/phrases_ucm.json, phrases_resisc45.json   Part 2 표현
prompts/sun_aid_classes.json  AID 개념 → SUN397 장면 대응표
experiments/run_pipeline.py   Part 1 전체 재현
experiments/area_verifier.py  실패한 확인 담당 실험 (부정 결과)
experiments/multi_benchmark.py  Part 2: 데이터셋 4 × 모델 3
experiments/make_report.py    Part 2 결과 표 만들기
notebooks/colab_demo.ipynb    Part 1 Colab 실행용
notebooks/benchmarks.ipynb    Part 2 Colab 실행용
results/                  notebook_results.md (Part 1), multi_benchmark.{json,md} (Part 2)
tests/                    합성 데이터로 하는 CPU 테스트
```

## 8. 참고
- SAM 3: Segment Anything with Concepts, [arXiv:2511.16719](https://arxiv.org/abs/2511.16719) · [code](https://github.com/facebookresearch/sam3)
- Promptable Concept Segmentation from Above: Evaluating SAM 3 in Remote Sensing, [arXiv:2607.09583](https://arxiv.org/abs/2607.09583)
- AID: Xia et al., TGRS 2017 · HF 미러 [`blanchon/AID`](https://huggingface.co/datasets/blanchon/AID)
- UC Merced Land Use: Yang & Newsam, 2010 · NWPU-RESISC45: Cheng et al., Proc. IEEE 2017 · SUN397: Xiao et al., CVPR 2010
- CLIP: Radford et al., 2021 · RemoteCLIP: Liu et al., [arXiv:2306.11029](https://arxiv.org/abs/2306.11029) · RS-TransCLIP: El Khoury et al., ICASSP 2025, [arXiv:2409.00698](https://arxiv.org/abs/2409.00698)
- 관련: CuPL, DCLIP, Waffle (LLM 기반 클래스 설명) · Meta-prompting for zero-shot RS scene classification (ISPRS 2026) · SegEarth-OV3 ([arXiv:2512.08730](https://arxiv.org/abs/2512.08730))

## 라이선스
이 저장소의 코드는 MIT 라이선스예요. SAM 3 코드와 체크포인트는 Meta의 SAM License를 따르고, 이 저장소에는 들어 있지 않아요. 연구 결과를 발표할 때는 SAM 사용을 밝혀야 해요. AID 이미지도 포함하지 않았고, 원 데이터셋 라이선스는 명시돼 있지 않아요.
