# RL_EPIDEMIC :syringe:

## Project Overview

This project aims to develop and implement a vaccination strategy using reinforcement learning to halt the spread of a disease. Our environment is a grid-based representation of individuals who can be in one of four states: susceptible, infected, vaccinated, or recovered (SIRV). We experimented with several RL algorithms, including PPO, DQN, and REINFORCE, to observe the emergence of effective strategies. While PPO successfully solved the problem, DQN and REINFORCE posed more challenges.

This project is part of the Reinforcement Learning course (EE-568) at EPFL, Spring 2024, worth 6 ECTS credits.

## Environment

The environment is implemented in the GridWorldEnv, located in `EPIDEMIC/envs/GridWorldEnv.py`.

### Grid Representation

The environment is a three-dimensional grid with dimensions representing channels (C), height (H), and width (W). Each cell in this grid can be in one of four states, encoded across four channels:

- **Susceptible (S)**: (1, 0, 0, 0) - Susceptible individuals can contract the disease and need vaccination.
- **Infected (I)**: (0, 1, 0, 0) - Infected individuals can spread the disease to neighbors; it's too late for vaccination.
- **Recovered (R)**: (0, 0, 1, 0) - Recovered individuals cannot pass on the disease.
- **Vaccinated (V)**: (0, 0, 0, 1) - Vaccinated individuals are immune and cannot spread the disease.

### Action Space

The agent can perform five discrete actions:

- Move right
- Move left
- Move up
- Move down
- Vaccinate

## Run History

You can find all the runs and detailed logs of this project on [Weights & Biases](https://wandb.ai/andreamiele/EPIDEMIC_RL?nw=nwuserandreamiele). A report summarizing the key plots and findings is available [here](https://wandb.ai/andreamiele/EPIDEMIC_RL/reports/Report--Vmlldzo4MTU1OTcy).

## Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.6+
- Pip

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/RL_EPIDEMIC.git
   cd RL_EPIDEMIC
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run a specific algorithm, use the corresponding script and add the necessary arguments. For example, to run the PPO algorithm:

```bash
python ppo.py
```

Replace `ppo.py` with `dqn.py` or `reinforce.py` to run the respective algorithms.

## Project Structure

```
RL_EPIDEMIC/
│
├── envs/
│   └── GridWorldEnv.py     # Environment implementation
│
├── ppo.py              # PPO algorithm implementation
│── dqn.py              # DQN algorithm implementation
│── reinforce.py        # REINFORCE algorithm implementation
│
├── plot/
│   └── ... .csv                 # Data for plotting for the final report
│   └── plotting.ipynb           # Notebook for plotting for the final report
│
├── requirements.txt        # Required dependencies
├── README.md               # Project overview
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more detailed information and documentation, please refer to the project's main repository and associated documentation files.

```

```
