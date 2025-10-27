# DF-DUNE: Dual-Faithful DUNE for Safe Robot Navigation

> **鍒涙柊鏂规瀹屾暣鍒嗘瀽涓庡疄鏂芥寚鍗?*  
> 鍩轰簬NeuPAN椤圭洰鐨勬繁搴︿唬鐮佸鏌ヤ笌鏈€鏂版枃鐚皟鐮?2023-2025)

---

## 馃搵 鏂囨。瀵艰埅

鏈粨搴撳寘鍚獶F-DUNE鍒涙柊鏂规鐨勫畬鏁存妧鏈枃妗?

| 鏂囨。 | 鍐呭 | 閫傜敤瀵硅薄 |
|------|------|---------|
| **[鍒嗘瀽鎶ュ憡](DF-DUNE鍒涙柊鏂规鍒嗘瀽鎶ュ憡.md)** | 鍒涙柊鎬ц瘎浼般€佸彲琛屾€у垎鏋愩€侀闄╄瘎浼?| 鍐崇瓥鑰呫€佺爺绌惰€?|
| **[瀹炴柦鎸囧崡](DF-DUNE鎶€鏈疄鐜版寚鍗?md)** | 浠ｇ爜瀹炵幇銆侀厤缃柟娉曘€佽皟璇曟妧宸?| 宸ョ▼甯堛€佸紑鍙戣€?|
| **[瀹為獙鏂规](DF-DUNE瀹為獙璁捐鏂规.md)** | 璇勬祴鎸囨爣銆佸満鏅璁°€佺粺璁℃柟娉?| 瀹為獙浜哄憳銆佽鏂囦綔鑰?|

---

## 馃幆 鏍稿績鍒涙柊鐐?

DF-DUNE鍦∟euPAN鐨凞UNE妯″潡鍩虹涓?寮曞叆涓夊眰鍒涙柊:

### 1锔忊儯 鐞嗚灞? 纭害鏉熺缁忓寲
- **A-1**: 闆跺弬鏁扮悆鎶曞奖 (宸插疄鐜扳渽)
- **A-2**: Learned-Prox杞昏繎绔ご (鍙€?
- **A-3**: KKT娈嬪樊姝ｅ垯 猸?**鎺ㄨ崘浼樺厛瀹炵幇**

### 2锔忊儯 绠楁硶灞? PDHG灞曞紑
- **B-1**: 鍘熷-瀵瑰伓娣峰悎姊害灞曞紑(J=1/2/3) 猸愨瓙 **鏍稿績鍗栫偣**
- **B-2**: BPQP鏁欏笀钂搁 (鍙€?

### 3锔忊儯 褰掔撼鍋忕疆灞? SE(2)绛夊彉
- **C**: 鏋佸潗鏍囩紪鐮?+ 鏃嬭浆绛夊彉 猸?**鎻愬崌椴佹鎬?*

---

## 馃搳 棰勬湡鏁堟灉

鍩轰簬鐞嗚鍒嗘瀽涓庢枃鐚鏍?棰勬湡鏀硅繘:

| 鎸囨爣 | Baseline | DF-DUNE | 鏀硅繘骞呭害 |
|------|----------|---------|---------|
| 瀵瑰伓杩濆弽鐜?| ~10% | <1% | **-90%** 鉁?|
| 璺濈浼拌MAE | 鍩哄噯 | -20% | **鎻愬崌20%** 鉁?|
| 闂幆鎴愬姛鐜?| 鍩哄噯 | +5% | **鎻愬崌5%** 鉁?|
| 鏈€灏忓畨鍏ㄩ棿璺?| 鍩哄噯 | +20% | **鎻愬崌20%** 鉁?|
| 鎺ㄧ悊鏃跺欢 | 0.5ms | 1.2ms | +140% 鈿狅笍 (浠嶆弧瓒冲疄鏃? |

---

## 馃殌 蹇€熷紑濮?

### 鏈€灏忓彲琛屽疄鐜?5鍒嗛挓)

1. **鍚敤纭姇褰?* (宸插疄鐜?浠呴渶閰嶇疆)

```yaml
# example/dune_train/dune_train_acker_df.yaml
train:
  projection: 'hard'           # 鍚敤纭姇褰?
  monitor_dual_norm: true      # 鐩戞帶杩濆弽鐜?
```

2. **杩愯璁粌**

```bash
cd NeuPAN-py38
python example/dune_train/dune_train_acker.py --config dune_train_acker_df.yaml
```

3. **妫€鏌ユ棩蹇?*

鏌ョ湅杈撳嚭涓殑 `dual_norm_violation_rate` 鍜?`dual_norm_p95`,搴旀樉钁椾綆浜巄aseline銆?

### 瀹屾暣瀹炵幇(3-4鍛?

鍙傝€?[瀹炴柦鎸囧崡](DF-DUNE鎶€鏈疄鐜版寚鍗?md) 鐨勮矾绾垮浘:

- **Week 1**: 瀹炵幇KKT姝ｅ垯 + PDHG-Unroll + SE(2)绛夊彉
- **Week 2**: 娑堣瀺鐮旂┒涓庤秴鍙傛暟璋冧紭
- **Week 3**: 闂幆璇勬祴(4涓満鏅?
- **Week 4**: 鏁版嵁鍒嗘瀽涓庤鏂囨挵鍐?

---

## 馃搱 瀹為獙璁捐姒傝

### 瀵规瘮鏂规硶(娑堣瀺)

| ID | 閰嶇疆 | 鐩殑 |
|----|------|------|
| M0 | Baseline | 鍘熷NeuPAN |
| M1 | +纭姇褰?| 楠岃瘉鎶曞奖鏁堟灉 |
| M2 | M1+KKT | 楠岃瘉KKT姝ｅ垯 |
| M4 | M2+PDHG(J=3) | 楠岃瘉PDHG灞曞紑 |
| M5 | M4+SE(2) | 瀹屾暣DF-DUNE |

### 璇勬祴灞傛

```
Level 1: 妯″潡绾?(DUNE鍗曠嫭)
  鈹溾攢 鍑犱綍绮惧害: distance_mae, angle_error
  鈹溾攢 绾︽潫婊¤冻: violation_rate, slack_p95
  鈹斺攢 璁＄畻鏁堢巼: inference_time, throughput

Level 2: 绯荤粺绾?(DUNE+NRMP)
  鈹溾攢 浼樺寲璐ㄩ噺: trajectory_cost, convergence_iters
  鈹斺攢 瀹夊叏鎬? min_clearance, constraint_violations

Level 3: 闂幆 (瀹屾暣NeuPAN)
  鈹溾攢 浠诲姟鎴愬姛: success_rate, path_efficiency
  鈹溾攢 瀹炴椂鎬? planning_freq, total_time
  鈹斺攢 椴佹鎬? collision_rate, OOD_performance
```

### 娴嬭瘯鍦烘櫙

1. **鍦烘櫙1**: 闈欐€侀殰纰?鍩虹) - 50 episodes
2. **鍦烘櫙2**: 鐙獎閫氶亾(鎸戞垬) - 30 episodes
3. **鍦烘櫙3**: 鍔ㄦ€侀殰纰?楂樼骇) - 40 episodes
4. **鍦烘櫙4**: 鐪熷疄鍦板浘(楠岃瘉) - 60 episodes

---

## 馃敩 瀛︽湳浠峰€?

### 鍒涙柊鎬х煩闃?

| 缁村害 | 鍒涙柊鐐?| 瀵规爣鏂囩尞 | 宸紓鍖?|
|------|--------|---------|--------|
| 鐞嗚 | 瀵瑰伓绌洪棿纭害鏉?| CNF (NeurIPS'23) | 棣栨鐢ㄤ簬瀵艰埅 |
| 绠楁硶 | PDHG灞曞紑 | Unrolled Opt (2024) | 棣栨鐢ㄤ簬鐐光啋瀵瑰伓 |
| 鍑犱綍 | SE(2)绛夊彉瀵瑰伓 | E(2)-ViT (UAI'23) | 鏂板簲鐢ㄥ満鏅?|
| 绯荤粺 | 绔埌绔彲寰?| NeuPAN (T-RO'25) | 淇濈暀鐗╃悊涓€鑷存€?|

### 鎶曠寤鸿

**鐩爣浼氳**:
- **ICRA 2026** (鎺ㄨ崘): 鎴2025骞?鏈?鏈夊厖瓒虫椂闂村畬鍠?
- **IROS 2025** (婵€杩?: 鎴2025骞?鏈?鏃堕棿绱ц揩
- **NeurIPS 2025** (楂橀闄?: 闇€瑕佸己鐞嗚璇佹槑

**璁烘枃缁撴瀯**:
1. **寮曡█**: 寮鸿皟"鍙鎬р啋瀹夊叏涓婄晫"鐨勯噸瑕佹€?
2. **鏂规硶**: 
   - Sec 3.1: 纭害鏉熺缁忓寲
   - Sec 3.2: PDHG灞曞紑 鈫?**鏍稿績璐＄尞**
   - Sec 3.3: SE(2)绛夊彉
3. **瀹為獙**: 妯″潡绾?+ 闂幆 + 娑堣瀺
4. **鐞嗚**: 闄勫綍璇佹槑PDHG鏀舵暃鎬?

---

## 鈿狅笍 椋庨櫓涓庣紦瑙?

### 鎶€鏈闄?

| 椋庨櫓 | 姒傜巼 | 褰卞搷 | 缂撹В鎺柦 |
|------|------|------|---------|
| PDHG涓嶆敹鏁?| 涓?| 楂?| 娣诲姞safeguard,鍥為€€鍒扮‖鎶曞奖 |
| 鏃跺欢瓒呴绠?| 浣?| 涓?| 鏃╁仠鏈哄埗,鑷€傚簲J |
| 璁粌涓嶇ǔ瀹?| 涓?| 涓?| 璇剧▼瀛︿範,姊害瑁佸壀 |

### 瀛︽湳椋庨櫓

| 椋庨櫓 | 姒傜巼 | 褰卞搷 | 缂撹В鎺柦 |
|------|------|------|---------|
| 瀹＄浜鸿川鐤戝垱鏂版€?| 涓?| 楂?| 寮哄寲涓嶤NF/BPQP鐨勫姣?|
| 缂轰箯鐞嗚璇佹槑 | 楂?| 涓?| 琛ュ厖鏀舵暃鎬у垎鏋?|
| 瀹為獙涓嶅鍏呭垎 | 涓?| 涓?| 澧炲姞鐪熷疄鍦烘櫙娴嬭瘯 |

---

## 馃摎 鍙傝€冩枃鐚?绮鹃€?

### 鏍稿績渚濇嵁

1. **NeuPAN** (T-RO 2025): 鍘熷妗嗘灦
   - [arXiv](https://export.arxiv.org/abs/2403.06828)

2. **Constrained Neural Fields** (NeurIPS 2023): 纭害鏉熺缁忓寲
   - [Paper](https://papers.nips.cc/paper_files/paper/2023/hash/47547ee84e3fbbcbbbbad7c1fd9a973b-Abstract-Conference.html)

3. **BPQP** (NeurIPS 2024): 鍙井鍑稿眰
   - [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8db12f7214d3a1a0c450ba751163e0fd-Abstract-Conference.html)

4. **E(2)-Equivariant ViT** (UAI 2023): 绛夊彉缃戠粶
   - [Paper](https://proceedings.mlr.press/v216/xu23b.html)

5. **NVBlox-ESDF** (2023-2024): 宸ヤ笟绾SDF
   - [Docs](https://nvidia-isaac.github.io/nvblox/pages/torch_examples_esdf.html)

### 鏂规硶璁?

6. **Neural SDF** (NeurIPS 2023): 鍑犱綍涓€鑷存€?
   - [Paper](https://papers.nips.cc/paper_files/paper/2023/hash/c87bd5843849884e9430f1693b018d71-Abstract-Conference.html)

---

## 馃洜锔?浠ｇ爜缁撴瀯

```
NeuPAN-py38/
鈹溾攢鈹€ neupan/
鈹?  鈹溾攢鈹€ blocks/
鈹?  鈹?  鈹溾攢鈹€ dune.py              # 涓绘ā鍧?闇€淇敼)
鈹?  鈹?  鈹溾攢鈹€ dune_train.py        # 璁粌鑴氭湰(闇€淇敼)
鈹?  鈹?  鈹溾攢鈹€ obs_point_net.py     # 缃戠粶缁撴瀯(闇€淇敼)
鈹?  鈹?  鈹斺攢鈹€ pdhg_layer.py        # 鏂板: PDHG灞?
鈹?  鈹斺攢鈹€ evaluation/
鈹?      鈹斺攢鈹€ dune_metrics.py      # 鏂板: 璇勬祴鎸囨爣
鈹溾攢鈹€ example/
鈹?  鈹溾攢鈹€ dune_train/
鈹?  鈹?  鈹斺攢鈹€ dune_train_acker_df.yaml  # 鏂板: DF-DUNE閰嶇疆
鈹?  鈹斺攢鈹€ evaluation/
鈹?      鈹斺攢鈹€ closed_loop_test.py       # 鏂板: 闂幆娴嬭瘯
鈹斺攢鈹€ experiments/
    鈹溾攢鈹€ results/                 # 瀹為獙缁撴灉
    鈹斺攢鈹€ figures/                 # 鍥捐〃
```

---

## 馃摓 鏀寔涓庡弽棣?

### 甯歌闂

**Q1: 纭姇褰卞凡瀹炵幇,涓轰粈涔堣繕闇€瑕並KT姝ｅ垯?**

A: 纭姇褰变粎鍦?*鎺ㄧ悊鏃?*寮哄埗绾︽潫,璁粌鏃剁綉缁滃彲鑳藉鍒拌繚鍙嶇害鏉熺殑瑙ｃ€侹KT姝ｅ垯鍦?*璁粌鏃?*寮曞缃戠粶瀛︿範鍙瑙?涓よ€呬簰琛ャ€?

**Q2: PDHG灞曞紑浼氭樉钁楀鍔犳椂寤跺悧?**

A: J=3鏃?鏃跺欢浠?.5ms澧炶嚦1.2ms(+140%),浣嗕粛婊¤冻20Hz瀹炴椂瑕佹眰(50ms棰勭畻)銆傚彲閫氳繃鏃╁仠鏈哄埗杩涗竴姝ヤ紭鍖栥€?

**Q3: 濡傛灉闂幆鎬ц兘娌℃湁鎻愬崌鎬庝箞鍔?**

A: 閲嶆柊瀹氫綅涓?绾︽潫婊¤冻涓庡畨鍏ㄦ€?鑰岄潪"鎬ц兘鎻愬崌",寮鸿皟杩濆弽鐜囬檷浣庡拰瀹夊叏闂磋窛澧炲姞銆傚弬鑰冨疄楠屾柟妗堢殑"澶囬€夋柟妗?銆?

### 鑱旂郴鏂瑰紡

- **鎶€鏈棶棰?*: 鎻愪氦Issue鍒版湰浠撳簱
- **瀛︽湳璁ㄨ**: 鍙傝€冨師NeuPAN璁烘枃浣滆€呰仈绯绘柟寮?
- **浠ｇ爜璐＄尞**: 娆㈣繋Pull Request

---

## 馃搫 璁稿彲璇?

鏈枃妗ｉ伒寰?**CC BY-NC-SA 4.0** 璁稿彲璇併€?

浠ｇ爜瀹炵幇搴旈伒寰狽euPAN椤圭洰鐨?**GPL-3.0** 璁稿彲璇併€?

---

## 馃檹 鑷磋阿

- **NeuPAN鍥㈤槦**: 鎻愪緵浼樼鐨勫紑婧愭鏋?
- **鏂囩尞浣滆€?*: CNF, BPQP, E(2)-ViT绛夊伐浣滅殑鍚彂
- **瀹℃煡鑰?*: 瀵规湰鏂规鐨勫弽棣堜笌寤鸿

---

## 馃搮 鏇存柊鏃ュ織

- **2025-01-03**: 鍒濆鐗堟湰,瀹屾垚浠ｇ爜瀹℃煡涓庢柟妗堣璁?
- **2025-01-XX**: (寰呮洿鏂?瀹炵幇KKT姝ｅ垯
- **2025-01-XX**: (寰呮洿鏂?瀹炵幇PDHG-Unroll
- **2025-XX-XX**: (寰呮洿鏂?瀹屾垚瀹為獙璇勬祴

---

## 馃帗 寮曠敤

濡傛灉鏈柟妗堝鎮ㄧ殑鐮旂┒鏈夊府鍔?璇峰紩鐢?

```bibtex
@misc{df-dune-2025,
  title={DF-DUNE: Dual-Faithful DUNE for Safe Robot Navigation},
  author={[Your Name]},
  year={2025},
  note={Technical Report based on NeuPAN framework}
}

@article{neupan2025,
  title={NeuPAN: Direct Point Robot Navigation with End-to-End Model-based Learning},
  author={Han, Ruihua and others},
  journal={IEEE Transactions on Robotics},
  year={2025}
}
```

---

**鏈€鍚庢洿鏂?*: 2025-01-03  
**鐗堟湰**: v1.0  
**鐘舵€?*: 鉁?鏂规璁捐瀹屾垚,绛夊緟瀹炴柦


