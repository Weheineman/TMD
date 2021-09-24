from abc import ABC, abstractmethod

class GreedyWrapper(ABC):
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    @abstractmethod
    def step(self):
        pass
    

# class ForwardWrapper(GreedyWrapper):
    # def __init__(self, model, data):
    #     self.state = [False] * data.dimension
    #     super().__init__(model, data)

    # def next_states(self) -> List[List[bool]]:
    # state_list = []
    # unused_vars = [ index for index, used in enumerate(self.current_state) if not used ]
    # for unused_index in unused_vars:
    #     next_state = self.current_state.copy()
    #     next_state[unused_index] = True
    #     state_list.append(next_state)
    
    # return state_list