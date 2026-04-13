#!/usr/bin/env python3
"""
Test script to verify that RAQL training is now reproducible with seeding.
Run this to verify the reproducibility fix works.
"""


def inference(run_name: str, n_steps: int = 5):
    import os
    import numpy as np

    from multi_agent_power_allocation.utils.seed import set_seed, create_generator
    from multi_agent_power_allocation.utils.train_config import TrainConfig
    from multi_agent_power_allocation.wireless_environment.env import WirelessEnvironment
    from multi_agent_power_allocation.algorithms.low_level import RAQL as LLRAQL
    from multi_agent_power_allocation.algorithms.high_level import RAQL as HLRAQL

    from multi_agent_power_allocation import BASE_DIR

    # Setup
    config_path = os.path.join(BASE_DIR, "run", "default_config.yaml")
    
    # SET SEED FIRST - before creating TrainConfig
    set_seed(1)
    rng = create_generator(1)
    
    # Create config
    config = TrainConfig(config_path, rng=rng)
    env = WirelessEnvironment(**config.env_config)
    
    # Get RAQL algorithms
    algorithm_mapping = config.env_config["algorithm_mapping"]
    for agent_id, policy in algorithm_mapping.items():
        # Get RAQL low-level algorithm
        assert isinstance(policy, HLRAQL)
    algorithm_mapping: dict[str, HLRAQL]

    # Test RAQL inference reproducibility
    print(f"TEST: {run_name} - RAQL Inference Reproducibility")

    # Create deterministic observation by seeding the space first
    obs_space = algorithm_mapping["0"].observation_space(
        env.wc_clusters["0"].num_devices,
        env.wc_clusters["0"].L_max
    )
    # CRITICAL: Seed the observation space since it has its own RNG
    if hasattr(obs_space, 'seed'):
        obs_space.seed(1)
    obs = np.array([obs_space.sample()])
    
    run_data = []
    for agent_id, policy in sorted(algorithm_mapping.items()):
        raql = policy.low_level_algorithm
        print(f"Epsilon of agent: {agent_id}, Run: {run_name}", raql.epsilon)
        
        # Get inference output 5 times
        agent_outputs = []
        for _ in range(n_steps):
            action = raql.inference(obs).detach().cpu().numpy()
            agent_outputs.append(action)
        
        run_data.append(agent_outputs)
    
    del obs
    del config
    del env
    run_data = np.array(run_data)
    del np

    return run_data
    
def test_raql_reproducibility_with_inference():
    """Test RAQL reproducibility by comparing inference outputs."""

    n_runs = 3
    n_steps = 200
    run_output = []
    for i in range(n_runs):
        import numpy as np

        run_output.append(inference(f"Run {i+1}", n_steps=n_steps))

        del np
    
    import numpy as np
    run_output = np.array(run_output)
    n_runs, n_agents, _ = run_output.shape[:3]
    
    print(f"\n{'='*70}")
    print(f"Reproducibility Analysis")
    print(f"Shape: {n_runs} runs, {n_agents} agents, {n_steps} steps per agent")
    print(f"{'='*70}")

    # 1. Check Consistency: Does Run 1 == Run 2 == Run 3 for each agent?
    for a in range(n_agents):
        # Compare all runs for this specific agent
        first_run = run_output[0, a]
        for r in range(1, n_runs):
            other_run = run_output[r, a]
            is_reproducible = np.allclose(first_run, other_run)
            
            status = "✓ REPRODUCIBLE" if is_reproducible else "✗ NOT REPRODUCIBLE"
            print(f"Agent {a} Run 1 vs Run {r+1}: {status}")
            
            if not is_reproducible:
                print(f"  Run 1 first action: {first_run[0]}")
                print(f"  Run {r+1} first action: {other_run[0]}")
                print(f"  Max difference: {np.abs(first_run - other_run).max()}")
            
            assert is_reproducible, f"Agent {a} failed reproducibility between Run 1 and Run {r+1}"

    # 2. Check Diversity: Are different agents actually different?
    # Only relevant if agents have different weights or IDs.
    if n_agents > 1:
        for r in range(n_runs):
            agent_0_actions = run_output[r, 0]
            agent_1_actions = run_output[r, 1]
            
            # We WANT them to be different
            are_different = not np.allclose(agent_0_actions, agent_1_actions)
            status = "✓ UNIQUE" if are_different else "⚠ IDENTICAL"
            print(f"Run {r+1} Agent 0 vs Agent 1: {status}")
            
            assert are_different, "Agents produced identical actions!"

def sample_action(run_name: str, n_steps: int = 5):
    import os
    import numpy as np

    from multi_agent_power_allocation.utils.seed import set_seed, create_generator
    from multi_agent_power_allocation.utils.train_config import TrainConfig
    from multi_agent_power_allocation.wireless_environment.env import WirelessEnvironment
    from multi_agent_power_allocation.algorithms.low_level import RAQL as LLRAQL
    from multi_agent_power_allocation.algorithms.high_level import RAQL as HLRAQL

    from multi_agent_power_allocation import BASE_DIR

    # Setup
    config_path = os.path.join(BASE_DIR, "run", "default_config.yaml")
    
    # SET SEED FIRST - before creating TrainConfig
    set_seed(1)
    rng = create_generator(1)
    
    # Create config
    config = TrainConfig(config_path, rng=rng)
    env = WirelessEnvironment(**config.env_config)
    
    # Get RAQL algorithms
    algorithm_mapping = config.env_config["algorithm_mapping"]
    for agent_id, policy in algorithm_mapping.items():
        # Get RAQL low-level algorithm
        assert isinstance(policy, HLRAQL)
    algorithm_mapping: dict[str, HLRAQL]

    # Test RAQL inference reproducibility
    print(f"TEST: {run_name} - RAQL Inference Reproducibility")

    # Create deterministic observation by seeding the space first
    obs_space = algorithm_mapping["0"].observation_space(
        env.wc_clusters["0"].num_devices,
        env.wc_clusters["0"].L_max
    )
    # CRITICAL: Seed the observation space since it has its own RNG
    if hasattr(obs_space, 'seed'):
        obs_space.seed(1)
    obs = np.array([obs_space.sample()])
    
    run_data = []
    for agent_id, policy in sorted(algorithm_mapping.items()):
        raql= policy.low_level_algorithm
        
        # Get inference output 5 times
        print(f"\nAgent {agent_id} RAQL inference outputs:")
        agent_outputs = []
        for _ in range(n_steps):
            action = raql.action_space.sample()
            agent_outputs.append(action)
        
        run_data.append(agent_outputs)
    
    del obs
    del config
    del env
    run_data = np.array(run_data)
    del np

    return run_data

def test_raql_reproducibility_with_action_sample():
    """Test RAQL reproducibility by comparing action sample outputs."""

    n_runs = 3
    n_steps = 200
    run_output = []
    for i in range(n_runs):
        import numpy as np

        run_output.append(sample_action(f"Run {i+1}", n_steps=n_steps))

        del np
    
    import numpy as np
    run_output = np.array(run_output)
    n_runs, n_agents, n_steps = run_output.shape[:3]

    # 1. Check Consistency: Does Run 1 == Run 2 == Run 3 for each agent?
    for a in range(n_agents):
        # Compare all runs for this specific agent
        first_run = run_output[0, a]
        for r in range(1, n_runs):
            is_reproducible = np.allclose(first_run, run_output[r, a])
            
            status = "✓ REPRODUCIBLE" if is_reproducible else "✗ NOT REPRODUCIBLE"
            print(f"Agent {a} Run 1 vs Run {r+1}: {status}")
            
            assert is_reproducible, f"Agent {a} failed reproducibility between Run 1 and Run {r+1}"

    # 2. Check Diversity: Are different agents actually different?
    # Only relevant if agents have different weights or IDs.
    if n_agents > 1:
        for r in range(n_runs):
            agent_0_actions = run_output[r, 0]
            agent_1_actions = run_output[r, 1]
            
            # We WANT them to be different
            are_different = not np.allclose(agent_0_actions, agent_1_actions)
            status = "✓ UNIQUE" if are_different else "⚠ IDENTICAL"
            print(f"Run {r+1} Agent 0 vs Agent 1: {status}")
            
            assert are_different, "Agents produced identical actions!"