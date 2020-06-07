#space invades oyunu için q learning algoritması
import numpy as np
import gym  # modelleri sağlayan atari emulatörü
from gym import wrappers

#q-learning algoritması : 
# Q(s,a)=Q(s,a)+Ir*(r(s,a)+Y*max(Q(s',a'))-Q(s,a))
#Q(s,a) (state,action) dediğimiz değer bizim şu anda {bulunduğumuz, gideceğimiz} dizin
#“lr” dediğimiz değer öğrenme katsayısı (learning rate)
#“r(s,a)” dediğimiz değer bizim {bulunduğumuz, gideceğimiz} ödül tablomuzdaki ödül değerimiz
# “Y” değeri gamma
#, “max (Q (s’,a’))” değeri ise gidebileceğimiz {gideceğimiz,gideceğimiz yerden gidebileceğimiz} yerlerin en yüksek Q değeridir

initial_lr = 1.0 # ilk bastaki öğrenme katsayısı
min_lr = 0.003    # minimum öğrenme katsayısı
n_states = 40   # durum sayısı
iter_max = 10000  #uygulayacağımız max iterasyon
gamma = 1.0       
t_max = 10000
eps = 0.02

def run_episode(env, policy=None, render=False):#toplam ödülün hesaplanması
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

def obs_to_state(env, obs):
    # bir durumu bir gözlemle eşitlendir
    env_low = env.observation_space.low #düşük gözlem
    env_high = env.observation_space.high # yüksek gözlem
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b
if __name__ == '__main__':
    #SpaceInvaders-v0
    env_name = 'SpaceInvaders-v0'  # space invanders simülasyonu
    env = gym.make(env_name) #env bizim öğrenme cevremizdir
    env.seed(0)
    np.random.seed(0)
    print ("q -learning kullanımı")
    q_table = np.zeros((n_states, n_states, 3)) # değerleri tutacağımız q tablosu
    for i in range(iter_max):
        obs = env.reset()  #öğrenme cevresinin resetlenmiş hali
        total_reward = 0   #toplam ödül miktarı
        # eta: öğrenme oranı 
         #  öğrenme oranı her adımda azalır
        eta = max(min_lr, initial_lr * (0.85 ** (i//100)))
        for j in range(t_max):
            a, b = obs_to_state(env, obs)
            if np.random.uniform(0, 1) < eps:
                action = np.random.choice(env.action_space.n)
            else:
                logits = q_table[a][b]
                logits_exp = np.exp(logits)
                probs = logits_exp / np.sum(logits_exp)
                action = np.random.choice(env.action_space.n, p=probs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            # q tablosunu güncelle
            a_, b_ = obs_to_state(env, obs)
            q_table[a][b][action] = q_table[a][b][action] + eta * (reward + gamma *  np.max(q_table[a_][b_]) - q_table[a][b][action])
            if done:
                break
        if i % 100 == 0:
            print('Adım(Iteration) #%d -- Toplam Ödül Miktarı(Total reward) = %d.' %(i+1, total_reward))
cozum_politikasi = np.argmax(q_table, axis=2)
cozum_politikasi_skoru = [run_episode(env, cozum_politikasi, False) for _ in range(100)]
print("Ortalama çözüm puanı = ", np.mean(cozum_politikasi_skoru))
