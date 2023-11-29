# 1. Getting Started - Load Python Packages
# 1.1. Import Packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import sys
import time
import datetime
sys.path.append("../FinRL-Library")
os.chdir('FinRL-Meta')
sys.path.insert(1, '/Users/mertcemri/Desktop/my_scripts/cs285/project/FinRL-Meta')
from meta import config
from meta.data_processor import DataProcessor
from meta.env_fx_trading.env_fx import tgym

from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# 2. Download and Preprocess Data
config.USE_TIME_ZONE_SELFDEFINED = 1
config.TIME_ZONE_SELFDEFINED = 'US/Eastern'

dp = DataProcessor(data_source="yahoofinance",
                   start_date = '2017-01-01',
                   end_date = '2021-01-01',
                   time_interval='1D')

symbol="GBPUSD"
dp.run(ticker_list = [f'{symbol}=X'], technical_indicator_list = config.INDICATORS, if_vix=False)
df = dp.dataframe

#print(df.head())

#df['time'] = df['time'] + ' 00:00'
#df['time'] = pd.to_datetime(df['time'], format='%Y.%m.%d %H:%M')
df['dt'] = pd.to_datetime(df['time'], unit='D', origin='2017-01-01')
df['symbol'] = symbol
#df['time'] = df['dt']
df.index = df['dt']
df['minute'] = df['dt'].dt.minute
df['hour'] = df['dt'].dt.hour
df['weekday'] = df['dt'].dt.dayofweek
df['week'] = df['dt'].dt.isocalendar().week
df['month'] = df['dt'].dt.month
df['year'] = df['dt'].dt.year
df['day'] = df['dt'].dt.day
#print(df['dt'])

# 3. Train
def train(env, agent, df, if_vix = True,**kwargs):
    learning_rate = kwargs.get('learning_rate', 2 ** -15)
    batch_size = kwargs.get('batch_size', 2 ** 11 )
    gamma = kwargs.get('gamma', 0.99)
    seed = kwargs.get('seed', 312)
    total_timesteps = kwargs.get('total_timesteps', 1e6)
    net_dimension = kwargs.get('net_dimension', 2**9)
    cwd = kwargs.get('cwd','./'+str(agent))

    
    # env_instance = map(env, [pd.read_csv(f) for f in files])
    if agent is not None:

        env_train = DummyVecEnv([lambda:env(df)])
        
        if agent =='ppo':
            model = PPO("MlpPolicy", env_train, learning_rate=learning_rate,
                        n_steps=2048, batch_size=batch_size, ent_coef=0.0,
                        gamma=gamma, seed=seed)
        elif agent =='a2c':
            model = A2C("MlpPolicy", env_train,learning_rate=learning_rate,
                        n_steps=2048, ent_coef=0.0,
                        gamma=gamma, seed=seed)    
        elif agent =='ddpg':
            model = DDPG("MlpPolicy", env_train, learning_rate=learning_rate,
                        gradient_steps=10,batch_size=batch_size, seed=seed)
            
        elif agent == "multiagent_implementation_1":
            model1 = PPO("MlpPolicy", env_train, learning_rate=learning_rate,
                        n_steps=2048, batch_size=batch_size, ent_coef=0.0,
                        gamma=gamma, seed=seed)
            
            action1_list = []
            reward_list = []
            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, info = t.step(action)
                #print(info["Close"])
                action1_list.append(action)
                reward_list.append(rewards)
            
            ##alter df or observation space
            env_train_2 = DummyVecEnv([lambda:env(df)])
            model2 = PPO("MlpPolicy", env_train_2, learning_rate=learning_rate,
                        n_steps=2048, batch_size=batch_size, ent_coef=0.0,
                        gamma=gamma, seed=seed)


        start_time = time.time()
        s = datetime.datetime.now()
        print(f"Training {agent} agent")
        print(f'Training start: {s}')
        model.learn(total_timesteps=total_timesteps, tb_log_name = agent)
        print('Training finished!')
        model_name = f"./data/models/{symbol}-{agent}-week-" + s.strftime('%Y%m%d%H%M%S')
        model.save(model_name)
        print(f'Trained model saved in {model_name}')
        print(f"trainning time: {(time.time() - start_time)}")

    else:
        raise ValueError('DRL library input is NOT supported. Please check.')
    
df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)

train(env=tgym,agent="ddpg",df=df)

assert False


# # if model: del model # remove to demonstrate saving and loading
# # PPO Models: ./data/models/GBPUSD-ppo-week-20231129120408 ./data/models/GBPUSD-week-20231126131448, ./data/models/GBPUSD-week-20231123194718.zip
# # A2C Models: ./data/models/GBPUSD-a2c-week-20231129121005 ./data/models/GBPUSD-week-20231126185131, ./data/models/GBPUSD-week-20231126205306
# # DDPG Models: ./data/models/GBPUSD-week-20231129114537
print("we are predicting now")

symbol="GBPUSD"


dp = DataProcessor(data_source="yahoofinance",
                   start_date = '2017-01-01',
                   end_date = '2018-01-01',
                   time_interval='1D')

dp.run(ticker_list = [f'{symbol}=X'], technical_indicator_list = config.INDICATORS, if_vix=False)
df = dp.dataframe
df['dt'] = pd.to_datetime(df['time'], unit='D', origin='2017-01-01')

df['symbol'] = symbol
#df['time'] = df['dt']
df.index = df['dt']
df['minute'] = df['dt'].dt.minute
df['hour'] = df['dt'].dt.hour
df['weekday'] = df['dt'].dt.dayofweek
df['week'] = df['dt'].dt.isocalendar().week
df['month'] = df['dt'].dt.month
df['year'] = df['dt'].dt.year
df['day'] = df['dt'].dt.day


df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
t = tgym(df)

model_name=f'./data/models/GBPUSD-ppo-week-20231129120408.zip'
model = PPO.load(model_name)
print("model loaded")

start_time = time.time()
obs = t.reset()
t.current_step=0
done = False

reward_list = []
balance_list = []
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = t.step(action)
    #print(info["Close"])
    balance_list.append(t.balance)
    reward_list.append(rewards)

reward_list_ppo = reward_list
balance_list_ppo = balance_list

model_name=f'./data/models/GBPUSD-week-20231129114537.zip'
model = DDPG.load(model_name)
print("model loaded")

start_time = time.time()
obs = t.reset()
t.current_step=0
done = False

reward_list = []
balance_list = []
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = t.step(action)
    #print(info["Close"])
    balance_list.append(t.balance)
    reward_list.append(rewards)
    #t.render(mode='graph')

plt.figure()
print(os.getcwd())
#plt.plot(reward_list_ppo)
plt.plot(balance_list)
#plt.legend(["ppo", "a2c"])
plt.savefig(f'trial_figures/forex_{symbol}_balance_ddpg.jpg')
print(f"--- running time: {(time.time() - start_time)}---")

