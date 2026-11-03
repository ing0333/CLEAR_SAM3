## 장면 분류: 5개 split 평균 ± 표준편차 (test 정확도 %)

| 데이터셋 | 모델 | baseline | +PPR | +TransLP | +TransCLIP-lite | +RS-TransCLIP | CLEAR | CLEAR+RS-TransCLIP | PSC | CLEAR-FS | linear probe (image emb.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AID (위) | SAM 3 | 37.1 ± 0.2 | 45.0 ± 0.3 | 41.2 ± 0.5 | 44.8 ± 0.3 | 44.0 ± 0.4 | **59.9 ± 1.8** | 66.6 ± 1.7 | 86.5 ± 0.6 | 87.0 ± 0.5 | – |
| AID (위) | CLIP ViT-L/14 | 63.8 ± 0.2 | 71.6 ± 0.3 | 67.0 ± 0.5 | 71.0 ± 0.4 | 77.1 ± 0.3 | **85.0 ± 0.7** | 92.3 ± 0.6 | 92.7 ± 0.2 | 92.8 ± 0.2 | 95.5 ± 0.2 |
| AID (위) | RemoteCLIP ViT-L/14 | 81.3 ± 0.3 | 90.4 ± 0.2 | 87.1 ± 0.4 | 88.0 ± 0.3 | 92.5 ± 0.4 | **90.2 ± 1.1** | 93.1 ± 1.1 | 93.6 ± 0.3 | 93.7 ± 0.3 | 95.5 ± 0.3 |
| UC Merced (위) | SAM 3 | 41.7 ± 1.1 | 65.4 ± 1.2 | 45.0 ± 1.1 | 62.2 ± 2.0 | 51.7 ± 0.9 | **80.2 ± 1.2** | 85.5 ± 2.0 | 93.3 ± 0.8 | 94.0 ± 0.7 | – |
| UC Merced (위) | CLIP ViT-L/14 | 63.9 ± 0.9 | 75.4 ± 0.7 | 65.4 ± 0.8 | 72.4 ± 1.2 | 77.3 ± 4.8 | **88.7 ± 1.6** | 90.6 ± 2.0 | 93.1 ± 1.1 | 93.1 ± 1.1 | 95.5 ± 0.6 |
| UC Merced (위) | RemoteCLIP ViT-L/14 | 91.2 ± 0.4 | 96.6 ± 0.3 | 92.1 ± 1.2 | 95.7 ± 0.3 | 96.3 ± 0.3 | **96.9 ± 0.5** | 97.3 ± 0.1 | 97.0 ± 0.5 | 97.0 ± 0.5 | 98.3 ± 0.4 |
| NWPU-RESISC45 (위) | SAM 3 | 37.9 ± 0.3 | 51.8 ± 0.7 | 38.1 ± 0.5 | 47.0 ± 0.6 | 47.6 ± 1.2 | **66.5 ± 1.0** | 68.0 ± 1.6 | 81.0 ± 1.0 | 82.6 ± 1.2 | – |
| NWPU-RESISC45 (위) | CLIP ViT-L/14 | 62.0 ± 1.0 | 70.5 ± 0.9 | 64.3 ± 0.7 | 68.2 ± 1.2 | 71.8 ± 1.3 | **80.0 ± 1.4** | 84.1 ± 1.1 | 85.7 ± 0.8 | 85.8 ± 0.8 | 87.9 ± 0.7 |
| NWPU-RESISC45 (위) | RemoteCLIP ViT-L/14 | 76.2 ± 0.4 | 85.1 ± 0.5 | 79.6 ± 0.8 | 83.1 ± 0.5 | 85.2 ± 0.9 | **87.4 ± 0.9** | 90.4 ± 0.7 | 89.0 ± 0.6 | 89.2 ± 0.5 | 91.9 ± 0.3 |
| SUN-AID (앞) | SAM 3 | 55.0 ± 0.8 | 66.5 ± 1.6 | 56.2 ± 0.6 | 59.7 ± 1.1 | 60.8 ± 1.0 | **77.8 ± 0.8** | 77.8 ± 0.8 | 78.7 ± 0.9 | 79.8 ± 0.6 | – |
| SUN-AID (앞) | CLIP ViT-L/14 | 77.7 ± 1.1 | 79.0 ± 1.2 | 78.6 ± 1.1 | 80.6 ± 1.5 | 80.5 ± 0.9 | **87.7 ± 1.3** | 89.9 ± 1.2 | 88.6 ± 0.6 | 88.8 ± 0.6 | 90.0 ± 0.3 |
| SUN-AID (앞) | RemoteCLIP ViT-L/14 | 68.8 ± 1.3 | 77.5 ± 1.3 | 69.5 ± 1.3 | 73.3 ± 1.5 | 75.8 ± 1.2 | **81.7 ± 0.7** | 85.8 ± 0.5 | 86.0 ± 1.6 | 86.1 ± 1.5 | 87.8 ± 1.4 |

## 모델별 평균 (데이터셋 4개, 5 split)

| 모델 | baseline | +PPR | +TransLP | +TransCLIP-lite | +RS-TransCLIP | CLEAR | CLEAR+RS-TransCLIP | PSC | CLEAR-FS | linear probe (image emb.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM 3 | 42.9 | 57.2 | 45.1 | 53.4 | 51.1 | 71.1 | 74.5 | 84.9 | 85.8 | – |
| CLIP ViT-L/14 | 66.8 | 74.1 | 68.8 | 73.1 | 76.7 | 85.3 | 89.2 | 90.0 | 90.1 | 92.2 |
| RemoteCLIP ViT-L/14 | 79.4 | 87.4 | 82.1 | 85.0 | 87.4 | 89.1 | 91.6 | 91.4 | 91.5 | 93.4 |

## 비교: CLEAR vs 공식 RS-TransCLIP (split별 승수)

| 데이터셋 | 모델 | CLEAR > RS-TransCLIP | CLEAR+RS-TransCLIP > CLEAR |
| --- | --- | --- | --- |
| AID | SAM 3 | 5/5 | 5/5 |
| AID | CLIP ViT-L/14 | 5/5 | 5/5 |
| AID | RemoteCLIP ViT-L/14 | 0/5 | 5/5 |
| UC Merced | SAM 3 | 5/5 | 5/5 |
| UC Merced | CLIP ViT-L/14 | 5/5 | 5/5 |
| UC Merced | RemoteCLIP ViT-L/14 | 3/5 | 4/5 |
| NWPU-RESISC45 | SAM 3 | 5/5 | 5/5 |
| NWPU-RESISC45 | CLIP ViT-L/14 | 5/5 | 5/5 |
| NWPU-RESISC45 | RemoteCLIP ViT-L/14 | 4/5 | 5/5 |
| SUN-AID | SAM 3 | 5/5 | 2/5 |
| SUN-AID | CLIP ViT-L/14 | 5/5 | 5/5 |
| SUN-AID | RemoteCLIP ViT-L/14 | 5/5 | 5/5 |

## 전체 ablation (평균 ± 표준편차)

| method | aid/sam3 | aid/openai | aid/remoteclip | ucm/sam3 | ucm/openai | ucm/remoteclip | resisc45/sam3 | resisc45/openai | resisc45/remoteclip | sun_aid/sam3 | sun_aid/openai | sun_aid/remoteclip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 37.1 ± 0.2 | 63.8 ± 0.2 | 81.3 ± 0.3 | 41.7 ± 1.1 | 63.9 ± 0.9 | 91.2 ± 0.4 | 37.9 ± 0.3 | 62.0 ± 1.0 | 76.2 ± 0.4 | 55.0 ± 0.8 | 77.7 ± 1.1 | 68.8 ± 1.3 |
| +PPR | 45.0 ± 0.3 | 71.6 ± 0.3 | 90.4 ± 0.2 | 65.4 ± 1.2 | 75.4 ± 0.7 | 96.6 ± 0.3 | 51.8 ± 0.7 | 70.5 ± 0.9 | 85.1 ± 0.5 | 66.5 ± 1.6 | 79.0 ± 1.2 | 77.5 ± 1.3 |
| +TransLP | 41.2 ± 0.5 | 67.0 ± 0.5 | 87.1 ± 0.4 | 45.0 ± 1.1 | 65.4 ± 0.8 | 92.1 ± 1.2 | 38.1 ± 0.5 | 64.3 ± 0.7 | 79.6 ± 0.8 | 56.2 ± 0.6 | 78.6 ± 1.1 | 69.5 ± 1.3 |
| +TransCLIP-lite | 44.8 ± 0.3 | 71.0 ± 0.4 | 88.0 ± 0.3 | 62.2 ± 2.0 | 72.4 ± 1.2 | 95.7 ± 0.3 | 47.0 ± 0.6 | 68.2 ± 1.2 | 83.1 ± 0.5 | 59.7 ± 1.1 | 80.6 ± 1.5 | 73.3 ± 1.5 |
| +RS-TransCLIP | 44.0 ± 0.4 | 77.1 ± 0.3 | 92.5 ± 0.4 | 51.7 ± 0.9 | 77.3 ± 4.8 | 96.3 ± 0.3 | 47.6 ± 1.2 | 71.8 ± 1.3 | 85.2 ± 0.9 | 60.8 ± 1.0 | 80.5 ± 0.9 | 75.8 ± 1.2 |
| OCT-S1 | 51.1 ± 0.5 | 78.2 ± 0.9 | 82.6 ± 0.7 | 62.8 ± 0.9 | 80.7 ± 1.8 | 93.6 ± 1.0 | 54.0 ± 0.8 | 74.2 ± 1.4 | 82.0 ± 0.8 | 70.8 ± 1.5 | 85.8 ± 1.6 | 77.5 ± 0.7 |
| OCT-S1+PPR | 56.7 ± 1.6 | 83.6 ± 0.7 | 88.9 ± 0.8 | 71.6 ± 1.8 | 84.8 ± 1.4 | 96.3 ± 0.7 | 59.8 ± 1.7 | 77.1 ± 1.1 | 87.6 ± 0.8 | 75.5 ± 2.0 | 87.5 ± 1.3 | 81.3 ± 1.0 |
| OCT-S2+PPR | 52.5 ± 1.6 | 74.4 ± 1.0 | 81.8 ± 1.1 | 63.3 ± 1.6 | 82.8 ± 1.6 | 92.9 ± 0.6 | 50.5 ± 0.7 | 71.3 ± 1.2 | 79.8 ± 1.1 | 62.0 ± 1.6 | 79.7 ± 1.3 | 75.8 ± 1.0 |
| MPV | 44.6 ± 1.0 | 76.8 ± 0.5 | 80.6 ± 0.8 | 54.4 ± 1.6 | 80.6 ± 1.4 | 91.6 ± 0.4 | 48.8 ± 1.5 | 71.8 ± 0.7 | 80.3 ± 1.0 | 67.0 ± 1.7 | 85.0 ± 1.1 | 72.1 ± 0.9 |
| CLEAR | 59.9 ± 1.8 | 85.0 ± 0.7 | 90.2 ± 1.1 | 80.2 ± 1.2 | 88.7 ± 1.6 | 96.9 ± 0.5 | 66.5 ± 1.0 | 80.0 ± 1.4 | 87.4 ± 0.9 | 77.8 ± 0.8 | 87.7 ± 1.3 | 81.7 ± 0.7 |
| OCT+MPV+TransLP | 45.8 ± 2.1 | 81.9 ± 1.2 | 86.0 ± 0.9 | 57.2 ± 2.1 | 82.9 ± 0.6 | 93.0 ± 0.6 | 49.4 ± 1.8 | 73.5 ± 0.7 | 82.4 ± 0.9 | 66.3 ± 2.1 | 86.6 ± 0.9 | 72.8 ± 1.5 |
| OCT+MPV+TransCLIP-lite | 60.3 ± 2.1 | 83.4 ± 0.7 | 86.9 ± 1.3 | 69.4 ± 2.1 | 87.6 ± 1.4 | 95.3 ± 0.2 | 58.2 ± 1.0 | 77.1 ± 0.9 | 84.9 ± 0.7 | 70.7 ± 1.2 | 87.9 ± 1.1 | 76.2 ± 1.7 |
| CLEAR+RS-TransCLIP | 66.6 ± 1.7 | 92.3 ± 0.6 | 93.1 ± 1.1 | 85.5 ± 2.0 | 90.6 ± 2.0 | 97.3 ± 0.1 | 68.0 ± 1.6 | 84.1 ± 1.1 | 90.4 ± 0.7 | 77.8 ± 0.8 | 89.9 ± 1.2 | 85.8 ± 0.5 |
| LR(class names) | 78.8 ± 0.9 | 90.2 ± 0.3 | 93.1 ± 0.2 | 88.2 ± 1.6 | 90.8 ± 0.7 | 95.9 ± 0.6 | 75.2 ± 1.4 | 84.5 ± 0.5 | 88.2 ± 0.4 | 78.0 ± 1.1 | 86.4 ± 0.7 | 85.2 ± 0.5 |
| PSC | 86.5 ± 0.6 | 92.7 ± 0.2 | 93.6 ± 0.3 | 93.3 ± 0.8 | 93.1 ± 1.1 | 97.0 ± 0.5 | 81.0 ± 1.0 | 85.7 ± 0.8 | 89.0 ± 0.6 | 78.7 ± 0.9 | 88.6 ± 0.6 | 86.0 ± 1.6 |
| CLEAR-FS | 87.0 ± 0.5 | 92.8 ± 0.2 | 93.7 ± 0.3 | 94.0 ± 0.7 | 93.1 ± 1.1 | 97.0 ± 0.5 | 82.6 ± 1.2 | 85.8 ± 0.8 | 89.2 ± 0.5 | 79.8 ± 0.6 | 88.8 ± 0.6 | 86.1 ± 1.5 |
| linear probe (image emb.) | – | 95.5 ± 0.2 | 95.5 ± 0.3 | – | 95.5 ± 0.6 | 98.3 ± 0.4 | – | 87.9 ± 0.7 | 91.9 ± 0.3 | – | 90.0 ± 0.3 | 87.8 ± 1.4 |

## 분할: mIoU % (5개 split, 선택 100장 / test 400장)

| 데이터셋 | names | SegEarth-OV3 | OCT-all | OCT-S1 | OCT+MPV |
| --- | --- | --- | --- | --- | --- |
| openearthmap | 34.0 ± 0.3 | 45.5 ± 0.3 | 44.6 ± 0.3 | 45.5 ± 0.4 | 45.1 ± 0.5 |
| loveda | 38.9 ± 0.2 | 45.2 ± 0.2 | 44.9 ± 0.3 | 47.0 ± 0.3 | 46.3 ± 0.5 |

**openearthmap 클래스별 IoU (5 split 평균)**

| 클래스 | names | SegEarth-OV3 | OCT-all | OCT-S1 | OCT+MPV |
| --- | --- | --- | --- | --- | --- |
| bareland | 0.4 | 8.7 | 8.9 | 18.2 | 11.9 |
| rangeland | 4.2 | 38.3 | 32.9 | 35.2 | 36.8 |
| developed space | 13.3 | 26.5 | 14.3 | 17.2 | 17.3 |
| road | 33.0 | 40.1 | 42.7 | 42.3 | 41.5 |
| tree | 56.7 | 63.0 | 61.6 | 58.0 | 60.3 |
| water | 66.6 | 73.7 | 75.5 | 73.4 | 74.1 |
| agriculture land | 28.8 | 42.8 | 50.8 | 49.6 | 46.6 |
| building | 69.1 | 70.8 | 70.3 | 70.0 | 72.0 |

OCT-S1이 고른 표현 (split별): sand and gravel surface, low green vegetation, paved parking lot, asphalt road, group of trees, water, farmland, building; sand and gravel surface, low green vegetation, paved parking lot, asphalt road, group of trees, river, agriculture land, building; sand and gravel surface, low green vegetation, concrete pavement, asphalt road, group of trees, water, farmland, building; sand and gravel surface, low green vegetation, paved parking lot, asphalt road, group of trees, water, farmland, building; sand and gravel surface, low green vegetation, paved parking lot, asphalt road, group of trees, river, farmland, building

**loveda 클래스별 IoU (5 split 평균)**

| 클래스 | names | SegEarth-OV3 | OCT-all | OCT-S1 | OCT+MPV |
| --- | --- | --- | --- | --- | --- |
| background | 42.8 | 42.6 | 36.4 | 42.5 | 40.1 |
| building | 62.5 | 63.6 | 63.7 | 62.5 | 62.8 |
| road | 50.8 | 50.8 | 49.9 | 50.3 | 51.0 |
| water | 47.7 | 47.7 | 47.6 | 50.5 | 49.6 |
| barren | 0.5 | 32.9 | 31.6 | 36.7 | 34.5 |
| forest | 22.2 | 32.7 | 39.9 | 40.2 | 39.3 |
| agricultural | 46.0 | 45.9 | 45.5 | 46.4 | 46.5 |

OCT-S1이 고른 표현 (split별): open ground, building, road, water, exposed dirt ground, group of trees, farmland; open ground, building, highway, water, exposed dirt ground, group of trees, agricultural; open ground, building, highway, water, exposed dirt ground, group of trees, farmland; open ground, building, road, water, exposed dirt ground, group of trees, agricultural; open ground, building, road, water, exposed dirt ground, group of trees, agricultural
