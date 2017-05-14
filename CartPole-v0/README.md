## CartPole-v0
棒のバランスをとるようにどう台車を動かすか。[公式サイト](https://gym.openai.com/envs/CartPole-v0),[公式Wiki](https://github.com/openai/gym/wiki/CartPole-v0)
- [動作説明コード](https://github.com/shunyooo/openaigym/blob/master/CartPole-v0/cart_pole1.py)(cart_pole1.py)
  - openai gymの環境をどのように動かすかの基本的な説明とコード。
- [離散的な解法-Q学習-](https://github.com/shunyooo/openaigym/blob/master/CartPole-v0/cart_pole3.py)(cart_pole3.py)
  - 連続値の状態(台車位置,台車速度,ポール角度,ポール先端速度)をそれぞれ離散値に分割してMDPとして扱い、Q学習にかけてみるコード。あんまり上手くいかない。
- [線形モデルによる解法1](https://github.com/shunyooo/openaigym/blob/master/CartPole-v0/cart_pole5.py)(cart_pole5.py)
  - 線形モデルを作成し、そのパラメータを推定する手法。パラメータは状態数と同じ4つ用意し、その推定はランダム探索での報酬和を最大に取ったものを採用していく方式。行動は線形結合の結果から決定される。かなりうまくいく。状態が4次元程度で少ないからだと考えられる。

## Environment
公式のを訳したやつ

### 状態(Observation)
4属性ある

Num | Observation | Min | Max
---|---|---|---
0 | 台車の位置(Cart Position) | -2.4 | 2.4
1 | 台車の速度(Cart Velocity) | -Inf | Inf
2 | ポールの角度(Pole Angle) | ~ -41.8&deg; | ~ 41.8&deg;
3 | ポールの先端速度(Pole Velocity At Tip) | -Inf | Inf


### 行動(Actions)
2行動ある

Num | Action
--- | ---
0 | 左へ押す(Push cart to the left)
1 | 右へ押す(Push cart to the right)

<img src = 'https://cloud.githubusercontent.com/assets/17490886/25933753/7ca4cd52-3654-11e7-857d-3b1f097736c9.png' height = 300>

### 報酬(Reward)
最後のステップも含めて、立っていると判断されている間(done=False)は報酬が１。

### 初期状態(Starting State)
±0.05間の一様乱数から与えられる。

### エピソードの終わり(Episode Termination)
1. ポールの角度が ±20.9°　を超えた時。
2. 台車の位置が　±2.4　を超えた時。つまり台車の中心が画面端に到達した時。
3. エピソードの長さが 200　を超えた時。 

### 解決要件(Solved Requirements)
openai gymにコードをあげる時の条件っぽい。
平均報酬が100回の連続した試行にわたって195.0以上であるときに解決されたとみなされる。
