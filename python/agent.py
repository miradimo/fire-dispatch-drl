import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

LR = 0.001
BATCH_SIZE = 32
EPSILON = 1.0 
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
current_loss = 0.0
current_reward = 0.0

memory = deque(maxlen=5000)

active_transitions = {} 

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = None
optimizer = None

def flatten_state(severity, distances, station_status):
    state = [severity]
    state.extend(distances)
    for status in station_status:
        state.extend(status)
    return np.array(state, dtype=np.float32)

def get_action(fire_id, severity, distances, station_status):
    global model, optimizer, active_transitions
    
    state = flatten_state(severity, distances, station_status)
    input_dim = len(state)
    output_dim = len(distances)
    
    if model is None:
        model = DQN(input_dim, output_dim)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        print(f"\n[RL] Model created. Inputs: {input_dim}, Outputs: {output_dim}")

    if random.random() < EPSILON:
        action = random.randint(0, output_dim - 1)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = torch.argmax(q_values).item()
        
    active_transitions[fire_id] = {'state': state, 'action': action}
    
    return action

def update_reward(fire_id, travel_time):
    global active_transitions, EPSILON, current_loss
    reward = -travel_time * 10.0

    with open("dqn_high.csv", "a", encoding="utf-8") as log_file:
        log_file.write(f"{fire_id},{travel_time}\n")
        
    if fire_id in active_transitions:
        transition = active_transitions.pop(fire_id)
        memory.append((transition['state'], transition['action'], reward))
        print(f"    [PYTHON] Memory updated for fire {fire_id}. Buffer: {len(memory)}/{memory.maxlen}")
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY 
        train_model()

def train_model():
    global current_loss, current_reward

    if len(memory) < BATCH_SIZE:
        return
        
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)

    q_values = model(states).gather(1, actions)

    loss = nn.MSELoss()(q_values, rewards)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    current_loss = loss.item()
    current_reward = rewards.mean().item()
    
    print(f"    [RL] Train Step! Loss: {current_loss:.4f} | Avg Reward: {current_reward:.2f} | Epsilon: {EPSILON:.2f}")



def save_weights():
    global model
    if model is not None:
        torch.save(model.state_dict(), 'dqn_weights.pth')
        return "Успешно: Веса нейросети сохранены (dqn_weights.pth)!"
    else:
        return "Ошибка: Модель еще не создана. Сгенерируйте хотя бы один пожар!"

def load_weights():
    global model, EPSILON
    if model is not None:
        if os.path.exists('dqn_weights.pth'):
            try:
                model.load_state_dict(torch.load('dqn_weights.pth'))
                EPSILON = EPSILON_MIN 
                return "Успешно: Обученные веса загружены! Epsilon снижен."
            except Exception as e:
                return f"Ошибка при загрузке: {e}"
        else:
            return "Ошибка: Файл dqn_weights.pth не найден. Сначала сохраните веса."
    else:
        return "Ошибка: Модель еще не инициализирована. Сгенерируйте хотя бы один пожар перед загрузкой!"
    

def get_metrics():
    global current_loss, EPSILON, current_reward
    return f"{current_loss},{EPSILON},{current_reward}"