import numpy as np
import sys
sys.path.append("/home/youtao/CT_toy")
import QCT as qct
from metric_func import half_system_entanglement_entropy
from tqdm import tqdm
import json

def read_circuit():
    with open('circuit.json', 'r') as f:
        circuit = json.load(f)
    return circuit

def read_initial_state():
    with open('initial_state.json', 'r') as f:
        initial_state = np.array(json.load(f))
    return initial_state


def _compute_single_chunk(L, p_ctrl, p_proj, num_time_steps):
    """Compute singular values using only Tao's implementation"""

    # Run Tao's implementation
    qct_tao = qct.QCT(L, p_ctrl, p_proj)
    for _ in tqdm(range(num_time_steps)):
        qct_tao.step_evolution()
    _, singular_values = half_system_entanglement_entropy(
        qct_tao.state, L, selfaverage=False, n=0, threshold=1e-15, return_singular_values=True)
    print(singular_values)
    
    # Return a dictionary with the results
    return {
        'L': L,
        'p_ctrl': p_ctrl,
        'p_proj': p_proj,
        'singular_values': singular_values,
        'p_proj': p_proj
    }



if __name__ == "__main__":
    L = 8
    p_ctrl = 0.4
    p_proj = 0.7
    num_time_steps = 2**L*2
    circuit = read_circuit()
    print(circuit[0])