import numpy as np
import random


class CSMA_CA_Agent: 
    """
    CSMA-CA Agent for the CSMA/CA environment.
    This agent implements the CSMA-CA protocol with two strategies: BEB (Binary Exponential Backoff) and Random Backoff.
    """
    
    def __init__(self, node_id, cw_min, cw_max, strategy='beb'):
        self.node_id = node_id  #노드 id
        self.cw_min = cw_min    #최소 대기 시간
        self.cw_max = cw_max    #최대 대기 시간
        self.current_cw = cw_min #현재 대기 시간
        self.strategy = strategy #전송 전략 beb, random
        self.backoff_timer = 0  
        self.action = [0]       # Initialize action = [wait, transmit]
        self.set_new_backoff()  # Set backoff timer

    def set_new_backoff(self):
        if self.strategy.lower() == 'random':
            self.backoff_timer = random.randint(1, self.cw_max) 
        elif self.strategy.lower() == 'beb':
            self.backoff_timer = random.randint(1, self.current_cw)

    def reset_backoff(self, collision_occured):     #충돌 발생 시 함수 
        if self.strategy.lower() == 'beb':          #beb일때만 충돌 영향있음 
            if collision_occured:   
                self.current_cw = min(self.current_cw * 2, self.cw_max)   # 2배, 최대값 중에 최소값 사용
            else:
                self.current_cw = self.cw_min
        self.set_new_backoff()

    def decrement_backoff(self):  # 대기시간 감소
        if self.backoff_timer > 0:  # 대기시간 >0 일때만 감소
            self.backoff_timer -= 1

    def ready(self): 
        return self.backoff_timer == 0

    def act(self, state):
        channel_state, collision_occured = state  # global state로부터 채널 상태, 충돌 발생 여부를 관찰  
        
        # Update backoff based on previous action and channel state
        if self.action[0] == 1:  # 전송 시도했을 때 충돌 발생 여부 부 확인 
            self.reset_backoff(collision_occured=collision_occured)
        else:   # 이전 액션이 대기라면
            self.decrement_backoff() # 대기시간 감소

        # Set new action
        self.action = [1] if self.ready() else [0] # 노드 전송 준비 상태일경우 전송 -> backoff timer=0
        return self.action

    def __repr__(self): #"node id: READY or WAITING (backoff timer / current cw)"
        return f"{self.node_id}: {'READY' if self.ready() else 'WAITING'} ({self.backoff_timer}/{self.current_cw})"