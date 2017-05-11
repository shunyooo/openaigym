## [CartPole-v0](https://gym.openai.com/envs/CartPole-v0)
棒のバランスをとるようにどう台車を動かすか。[公式Wiki](https://github.com/openai/gym/wiki/CartPole-v0)
- [動作説明コード](https://github.com/shunyooo/openaigym/blob/master/CartPole-v0/cart_pole1.py)(cart_pole1.py)
  - openai gymの環境をどのように動かすかの基本的な説明とコード。
- [離散的な解法-Q学習-](https://github.com/shunyooo/openaigym/blob/master/CartPole-v0/cart_pole3.py)(cart_pole3.py)
  - 連続値の状態(台車位置,台車速度,ポール角度,ポール先端速度)をそれぞれ離散値に分割してMDPとして扱い、Q学習にかけてみるコード。あんまり上手くいかない。
