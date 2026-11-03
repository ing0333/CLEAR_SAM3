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

## 5. 한계 (꼭 같이 읽어주세요)
- **데이터셋 1개, 표본 1개(seed 0), test는 클래스당 80장**이에요. 다른 데이터셋(NWPU-RESISC45 등)에서도 확인이 필요해요.
- **OCT 표현은 사람과 LLM이 만들었어요.** 8개 클래스는 진단 실험에서 먼저 썼고, 나머지 22개는 고정된 지시문으로 생성했어요(`prompts/phrases.json`).
- **PPR은 test 묶음의 클래스 비율이 비슷하다고 가정해요.** 비율이 크게 다르면 `prior=`로 비율을 넣어야 해요.
- **PSC가 배운 "부분"은 주변 환경인 경우가 있어요.** 예를 들어 Church ← "crowded rooftops", "parked cars"예요. AID 밖에서는 통하지 않을 수 있어요.
- **비교 대상이 부족해요.** RemoteCLIP 같은 원격탐사용 CLIP 계열의 zero-shot·few-shot 결과와 같은 조건으로 비교하지 않았어요.
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
prompts/phrases.json      클래스별 표현 + 생성 지시문
experiments/run_pipeline.py   전체 재현
experiments/area_verifier.py  실패한 확인 담당 실험 (부정 결과)
notebooks/colab_demo.ipynb    Colab 실행용
tests/                    합성 데이터로 하는 CPU 테스트
```

## 8. 참고
- SAM 3: Segment Anything with Concepts, [arXiv:2511.16719](https://arxiv.org/abs/2511.16719) · [code](https://github.com/facebookresearch/sam3)
- Promptable Concept Segmentation from Above: Evaluating SAM 3 in Remote Sensing, [arXiv:2607.09583](https://arxiv.org/abs/2607.09583)
- AID: Xia et al., TGRS 2017 · HF 미러 [`blanchon/AID`](https://huggingface.co/datasets/blanchon/AID)
- 관련: CuPL, DCLIP, Waffle (LLM 기반 클래스 설명) · Meta-prompting for zero-shot RS scene classification (ISPRS 2026) · SegEarth-OV3 ([arXiv:2512.08730](https://arxiv.org/abs/2512.08730))

## 라이선스
이 저장소의 코드는 MIT 라이선스예요. SAM 3 코드와 체크포인트는 Meta의 SAM License를 따르고, 이 저장소에는 들어 있지 않아요. 연구 결과를 발표할 때는 SAM 사용을 밝혀야 해요. AID 이미지도 포함하지 않았고, 원 데이터셋 라이선스는 명시돼 있지 않아요.
