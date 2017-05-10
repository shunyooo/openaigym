class QLearning():
    """Q学習を行うクラス"""
    
    def __init__(self,mdp,alpha=0.01,greedy_ratio = 0.2,gamma = 0.95):
        """初期化。全ての状態sと行動aについてQ(s,a)を初期化する。
        Q(s,a)は{(s,a):value,(s,a):value,..}というような形式。
        """
        self.Q = {}
        for s in mdp.states:
            for a in mdp.actions(s):
                self.Q[s,a] = random.uniform(1,10)
        self.alpha = alpha
        self.mdp = mdp
        self.greedy_ratio = greedy_ratio
        self.gamma = gamma
        
    def learn(self,max_epoch = 100):
        """1エピソードを消化して学習を行う。
        ある程度行動したら、失敗としてreturnする。"""
        #  状態を最初の状態s0にセット。
        state = self.mdp.init
        count = 0# ゴールにたどり着くまでにかかった行動回数。
        while True:
            #状態s_tからある方法で行動a_tを選択。ε-greedyを用いる。
            action = self.action_e_greedy(state)
            #選択した行動から、Q(s,a)を更新
            self.update_Qvalue(state,action)
            #次状態へ以降
            state = self.mdp.go(state,action)
            count += 1
            if state in mdp.terminals:# ゴール。エピソードを終了。
                return count,True
            elif count >= max_epoch:
                return count,False
        
        
    def update_Qvalue(self,state,action):
        """sにおいて選択した行動aから、Q(s,a)を更新"""
        # 更新式:
        #       Q(s, a) <- Q(s, a) + alpha * {r(s, a) + gamma max{Q(s`, a`)} -  Q(s,a)}
        #               Q(s, a): 状態sにおける行動aを取った時のQ値      Q_s_a
        #               r(s, a): 状態sにおける報酬      r_s_a
        #               max{Q(s`, a`) 次の状態s`が取りうる行動a`の中で最大のQ値 mQ_s_a)
        Q_s_a = self.get_Qvalue(state,action)#状態sにおける行動aを取った時のQ値  
        n_state = self.mdp.go(state,action)#次状態next_stateを取得
        r_s_a = self.mdp.R(state,action)#状態sで行動aを取った時の報酬R(s,a)
        
        # 次状態n_stateが取りうる行動n_actionの中で最大のQ値を求める
        mQ_ns_a = max([self.get_Qvalue(n_state,n_action) for n_action in self.mdp.actions(n_state)])
        
        # calculate
        q_value = Q_s_a + self.alpha * ( r_s_a +  self.gamma * mQ_ns_a - Q_s_a)
        
        # update
        self.set_Qvalue(state,action,q_value)
        
    def get_Qvalue(self,state,action):
        """Q(s,a)を取得。"""
        try:
            return self.Q[state,action]
        except KeyError:
            print("Q({0},{1})のKeyIndexError".format(state,action))
            raise
            # return 0.0
    
    def set_Qvalue(self,state,action,q_value):
        """Q値に値を代入する"""
        self.Q[state,action] = q_value
            
    def action_e_greedy(self,state):
        """ε-greedy法で行動を決定"""
        if self.greedy_ratio > random.random():
            #ランダムに行動選択
            return random.choice(self.mdp.actions(state))
        else:
            #greedyに行動選択
            return self.action_greedy(state)
    
    def action_greedy(self,state):
        """Q(s,a)を比較してgreedy法で行動を決定。"""
        best_actions = []  #最高の行動が複数存在した場合
        max_q_value = -1 #最大の行動価値を保存
        for a in self.mdp.actions(state):#すごく単純な最大求めるやつ
            q_value = self.get_Qvalue(state,a)
            if q_value > max_q_value:
                best_actions = [a,]
                max_q_value = q_value
            elif q_value == max_q_value:
                best_actions.append(a)
        return random.choice(best_actions)#Q値の最大値が複数存在する場合はその中からランダムに選択。
    
    def printPi(self):
        print("実行結果")
        state =  self.mdp.init
        count = 0
        
        grid = copy.copy(self.mdp.grid)#コピー
        
        while True:
            action = self.action_greedy(state)# greedyに最適なactionを選択
            n_state = self.mdp.go(state,action) # 次状態に以降
            print(state,toArrow(action),n_state)
            #pprint(grid)
            grid[state[0]][state[1]] = toArrow(action)
            
            state = n_state
            count += 1
            if state in mdp.terminals:# ゴール。エピソードを終了。
                break
            if count > 100:#100回移動してゴールに行かなければ収束していないことの表れ
                print("収束していないようです")
                break
                
        return pd.DataFrame(grid).style.highlight_null(null_color="Black")