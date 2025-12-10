# UAV-IRS Optimization Project

This project focuses on optimizing the energy consumption of Unmanned Aerial Vehicles (UAVs) assisting Intelligent Reflecting Surfaces (IRS) in a communication network. It employs meta-heuristic algorithms like **Genetic Algorithm (GA)**, **Hill Climbing (HC)**, and **Random Search** to find optimal configurations for UAV placement, flight paths, and transmission parameters to minimize energy usage while satisfying communication constraints.

## Project Overview

The core objective is to minimize the total energy consumption of the system, which includes:
-   **UAV Flight Energy:** Power required for hovering, vertical flight, and horizontal flight.
-   **Communication Energy:** Power used for data transmission (upload/download).
-   **Hardware Energy:** Power consumed by the IRS elements and other hardware.

The project analyzes the impact of various parameters on energy consumption, including:
-   Bandwidth
-   Data Size
-   Number of Generations (Convergence)
-   Transmission Power
-   UAV Weight
-   Mutation Probability

## Key Features

-   **Optimization Algorithms:** Implementation of Genetic Algorithm and Hill Climbing for finding optimal UAV-IRS configurations.
-   **Energy Analysis:** Detailed breakdown of energy consumption components (Hovering, Flying, Transmission).
-   **Parameter Sensitivity:** Scripts to analyze how different system parameters (Bandwidth, Data Size, etc.) affect total energy.
-   **Visualization:** Generates plots (saved as PDFs or displayed) to visualize the relationship between energy and various parameters.
-   **Data-Driven:** Uses CSV datasets for Base Station (BS), UAV, and User equipment (UE) parameters.

## Project Structure

```text
UAV-IRS Optimization/
├── Dataset/                  # CSV files containing simulation data (BS, UAV, User data, Channel gains)
├── Energy_vs_*.py            # Python scripts for specific parameter analysis (e.g., Energy vs Bandwidth)
├── Genetic_Algorithm.ipynb   # Jupyter optimization using Genetic Algorithm
├── Hill_Climb.ipynb          # Jupyter optimization using Hill Climbing
├── Random_search.ipynb       # Jupyter optimization using Random Search
├── fitness_summary *.csv     # CSV files storing results of fitness evaluations
├── requirements.txt          # Python dependencies
└── *.pdf                     # Generated plots (e.g., Energy vs Bandwidth.pdf)
```

## Prerequisites

-   Python 3.8+
-   Recommended: Virtual Environment (venv or conda)

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Key dependencies include:* `numpy`, `pandas`, `matplotlib`.

## Usage

### Running Analysis Scripts
You can run the individual Python scripts to perform specific analyses. For example, to analyze Energy vs. Generation count:

```bash
python Energy_vs_Generaion.py
```
*Note: The script name has a typo "Generaion" instead of "Generation".*

Other available analysis scripts:
-   `Energy_vs_Bandwidth.py`
-   `Energy_vs_DataSize.py`
-   `Energy_vs_Power.py`
-   `Energy_vs_weight.py`

### Running Notebooks
To explore the algorithms interactively or visualize the optimization process step-by-step, launch Jupyter Notebook:

```bash
jupyter notebook
```
Then open `Genetic_Algorithm.ipynb`, `Hill_Climb.ipynb`, or `Random_search.ipynb`.

## Algorithms Detail

### Genetic Algorithm (GA)
-   **Population:** Randomly initialized solutions (UAV positions, power allocation).
-   **Crossover & Mutation:** Standard genetic operators are applied to numerical values (Power, Bandwidth, Velocity) to explore the solution space.
-   **Fitness Function:** Minimizes the total energy consumption equation.
-   **Selection:** Auction-based assignment or standard selection to pair UAVs with Base Stations..

### Hill Climbing (HC)
-   **Iterative Improvement:** Starts with a random solution and iteratively makes small perturbations.
-   **Acceptance:** Accepts neighbors only if they lower the energy consumption (Fitness).
-   **Local Search:** Good for fine-tuning but may get stuck in local optima.

## Results
The project generates PDF plots showing the trade-offs and trends, such as:
-   Increasing data size typically increases energy consumption.
-   Optimal bandwidth selection can minimize transmission time and energy.
-   Optimization algorithms show convergence (decreasing energy) over generations.
