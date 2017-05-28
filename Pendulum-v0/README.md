## Pendulum-v0
![2017-05-28 9 10 27](https://cloud.githubusercontent.com/assets/17490886/26525057/8fdaa462-4385-11e7-8670-6d7144337bdd.png)

棒のバランスをとるようにどう力を与えるか。[公式サイト](https://gym.openai.com/envs/Pendulum-v0),[公式Wiki](https://github.com/openai/gym/wiki/Pendulum-v0)
- 政策勾配法([pendulum.py](https://github.com/shunyooo/openaigym/blob/master/Pendulum-v0/pendulum.py))
  - ガウス分布による政策勾配法の適用。
- 自然勾配法([pendulum2.py](https://github.com/shunyooo/openaigym/blob/master/Pendulum-v0/pendulum2.py))
  - ガウス分布による自然勾配法の適用。
  
## Environment
公式のを訳したやつ


### Observation
状態3次元
Type: Box(3)

Num | Observation  | Min | Max  
----|--------------|-----|----   
0   | cos値   | -1.0| 1.0
1   | sin値   | -1.0| 1.0
2   | 角速度    | -8.0| 8.0


## Actions
行動1次元
Type: Box(1)

Num | Observation  | Min | Max  
----|--------------|-----|----   
0   | ジョイントに与える力 | -2.0| 2.0

## Reward
報酬
以下の式で算出されている。

    -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)

Theta is normalized between -pi and pi. Therefore, the lowest cost is `-(pi^2 + 0.1*8^2 + 0.001*2^2) = -16.2736044`, and the highest cost is `0`. In essence, the goal is to remain at zero angle (vertical), with the least rotational velocity, and the least effort. 
シータは-piからpiに正規化されているので、この報酬は-16から0の範囲をとる。

## Starting State
初期状態。角度は-piからpiから、角速度は-1から1までをランダムにとる。
Random angle from -pi to pi, and random velocity between -1 and 1

## Episode Termination
終了条件。特にない。doneがTrueになることはない。step数に最大数を設けてあげるべき。
There is no specified termination.
Adding a maximum number of steps might be a good idea.

## Solved Requirements
解決要件。特になし。
None yet specified
