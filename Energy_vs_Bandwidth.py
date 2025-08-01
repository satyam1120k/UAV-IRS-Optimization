import multiprocessing
import pandas as pd
import numpy as np
import random
import math
import cmath
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

# Load datasets related to Base Stations, UAVs, and Clients
BS_data = pd.read_csv(r'Dataset\Base_station_data.csv')
UAV_data = pd.read_csv(r'Dataset\UAV_data.csv')
Individual_data = pd.read_csv(r'Dataset\Individual_data.csv')

V_down = pd.read_csv(r'Dataset\V_lm_down.csv')
h_l_km = pd.read_csv(r'Dataset\h_l_km.csv')
h_l_m = pd.read_csv(r'Dataset\h_l_m.csv')
V_up = pd.read_csv(r'Dataset\V_lm_up.csv')
g_l_km = pd.read_csv(r'Dataset\g_l_km.csv')
g_l_m = pd.read_csv(r'Dataset\g_l_m.csv')
V_har = pd.read_csv(r'Dataset\V_lm_har.csv')
f_l_km = pd.read_csv(r'Dataset\f_l_km.csv')
f_l_m = pd.read_csv(r'Dataset\f_l_m.csv')

# ===================== Global Variables =====================
P_m_har = BS_data['P_m_har']
T_m_har = BS_data['T_m_har']
P_m_down = BS_data['P_m_down']
f_km = Individual_data['f_km']
P_km_up = Individual_data['P_km_up']
V_lm_vfly = UAV_data['V_lm_vfly']
V_lm_hfly = UAV_data['V_lm_hfly']

# Constants
D_l_hfly_value = 100
Wl_value = 35.28
H_value = 20
p_km_max = 10
T_m = 10
delta = 0.012
Ar = 0.1256
s = 0.05
Nr = 4
V_tip = 102
Cd = 0.022
Af = 0.2113
D_km = 0.5
Dm = 0.49
p_max = 10
sigma_km = 10 ** (-13)
eta = 10
kappa = 0.5
num_population = 50
Bh = (1 - 2.2558 * pow(10, -5) * H_value) ** 4.2577
p_l_b = (delta / 8) * Bh * Ar * s * pow(V_tip, 3)

# Fitness function to calculate total energy consumption
def Fitness(E_ml_har, E_ml_down, E_ml_UAV): #equation no 30
    return E_ml_har + E_ml_down + E_ml_UAV

# Energy consumption of the UAV-IRS
def E_ml_UAV(P_l_vfly, T_l_vfly, P_lm_hfly, T_l_hfly, P_l_hov, T_lm_hov): #equation no 20
    return P_l_vfly * T_l_vfly + P_lm_hfly * T_l_hfly + P_l_hov * T_lm_hov

# Power calculations for different flight modes
def P_l_vfly(Wl, V_l_vfly, P_l_b, Nr, Ar, Bh): #equation no 11
    temp2 = Nr * Bh * Ar
    temp3 = np.sqrt(V_l_vfly**2 + (2 * Wl) / temp2)
    return ((Wl / 2) * (V_l_vfly + temp3)) + Nr * P_l_b

def P_lm_hfly(P_lm_blade, P_lm_fuselage, P_lm_induced): #equation no 13
    return P_lm_blade + P_lm_fuselage + P_lm_induced

def P_lm_blade(Nr, P_l_b, V_tip, V_lm_hfly): #equation no 14
    return Nr * P_l_b * (1 + ((3 * (V_lm_hfly**2)) / pow(V_tip, 2)))

def P_lm_fuselage(Cd, Af, Bh, V_lm_hfly): #equation no 15
    return (1 / 2) * Cd * Af * Bh * (V_lm_hfly**3)

def P_lm_induced(Nr, Bh, Ar, Wl, V_lm_hfly): #equation no 16
    return Wl * ((np.sqrt((Wl**2) / (4 * (Nr**2) * (Bh**2) * (Ar**2)) + ((V_lm_hfly**4) / 4)) - ((V_lm_hfly**2) / 2))**(1 / 2))

def P_l_hov(Wl, P_l_b, Nr, Ar, Bh): #equation no 18
    temp1 = Nr * P_l_b
    temp3 = np.sqrt(2 * (Nr * Bh * Ar))
    temp4 = ((Wl)**3 / 2) / temp3
    return temp1 + temp4

def T_lm_hov(T_km_com, T_kml_up, T_ml_down): #equation no 19
    return T_km_com + T_kml_up + T_ml_down

def R_ml_down(B,P_m_down,h_ml_worst): #eqation number 7
    temp1=(h_ml_worst*P_m_down) 
    if (1+temp1) <= 0:
        return 0  # Return 0 if log argument is non-positive to avoid error
    return B*math.log2(1+temp1)

def h_ml_worst(h_kml_down,sigma_km): #eqation number 8
    return h_kml_down/(sigma_km) # it will return the sigal value which is minimum of all
            # the value for each itaration

def calculate_exp_i_theta(theta): # part of equation 8
  return cmath.exp(1j * theta) # 1j represents the imaginary unit in Python

def h_kml_down(Angle,h_l_m,h_l_km): # part of equation 8
    result=[]
    if isinstance(Angle, float): # Check if Angle is float, if so, return 0 or handle appropriately
        return 0 # Or raise an exception or return a default value as needed

    if not isinstance(Angle, pd.Series): # added check to handle non-series input
        raise TypeError(f"Expected Angle to be pd.Series, got {type(Angle)}")

    for i in range(len(Angle)):
        theta_radians = math.radians(Angle.iloc[i]) # Use iloc for position-based indexing
        results= calculate_exp_i_theta(theta_radians)
        result.append(results)

    diagonal=np.diag(result)
    # Ensure h_l_m and h_l_km are correctly formatted as numpy arrays
    h_l_m_np = h_l_m.to_numpy() # Convert Series to numpy array
    h_l_km_np = h_l_km.to_numpy() # Convert Series to numpy array
    if h_l_m_np.ndim == 1:
        h_l_m_np = h_l_m_np.reshape(-1, 1) # Reshape to 2D if necessary
    if h_l_km_np.ndim == 1:
        h_l_km_np = h_l_km_np.reshape(1, -1) # Reshape to 2D if necessary

    a=np.dot(h_l_km_np,diagonal) # Use numpy arrays for dot product
    b=np.dot(a,h_l_m_np)      # Use numpy arrays for dot product
    final=abs(b[0][0]) # Take absolute value and ensure it's a scalar
    return (final**2)

def R_kml_up(B,P_km_up,h_kml_up,Sub,sigma_m): #eqation number 4
    temp1=(P_km_up*h_kml_up)/ (Sub+(sigma_m))
    return B*math.log2(1+temp1)
#this is inside the equation 4 have to take summation of h_i_up and P_i_up
def sub(P_i_up,h_il_up):
    return P_i_up*h_il_up

def E_km_com(f_km,T_km_com): #eqation number 3
    return eta*(10**(-28))*(f_km**3)*T_km_com

def E_kml_up(P_km_up,T_km_up): #eqation number 6
    return P_km_up*T_km_up

def E_kml_har(P_m_har,T_m_har,h_km_har): #equation no 2
    return kappa*P_m_har*T_m_har*h_km_har

Total_individual=500
num_bs = 5
num_irs_ele=50
num_generation =1
num_uav_irs = 8
population_size =50
S=50

# Define keys that should be subjected to perturbation (numerical parameters)
numerical_keys_for_crossover = [
    'P_m_down_value', 'P_m_har_value', 'T_m_har_value',
    'f_km_value', 'V_lm_vfly_value', 'V_lm_hfly_value',
    'P_km_up_value','V_down_value','V_up_value','V_har_value',
]

def GA(B): # Define function to process 
    print(f"calculaiton for B for GA",B)
    all_best_combinations = []
    sum_fitness_current = 0 # Initialize sum of fitness 
    # Main Genetic Algorithm Loop
    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]

        unique_row_indices = range(0,50)
        h_l_km_df_bs = h_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        g_l_km_df_bs = g_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        f_l_km_df_bs = f_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        f_km_bs = f_km[unique_row_indices[0:population_size]].reset_index(drop=True)
        P_km_up_bs = P_km_up[unique_row_indices[0:population_size]].reset_index(drop=True)
        
        valid_indices = [i for i in unique_row_indices[0:population_size] if i < len(P_km_up)]
        P_km_up_bs = P_km_up.iloc[valid_indices].reset_index(drop=True)

        for k in range(num_uav_irs):
            population = []
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]

            Sub_value=0

            for i in range(len(valid_indices)): # Using length of valid_indices
                h_il_up_value=h_kml_down(V_up.iloc[i, :],g_l_m.iloc[k, :],g_l_km_df_bs.iloc[i, :]) # Pass Series, corrected index to i
                Sub_value+=sub(P_km_up_bs[i],h_il_up_value)
            # compute power
            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)
            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
            T_l_vfly_value = H_value / V_lm_vfly_value
            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
            T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value
            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)

            #intialize population
            for i in range(S): 
                f_km_value = f_km_bs[random.randint(0,population_size)] 
                P_km_up_value = P_km_up_bs[random.randint(0,population_size)] 

                V_down_value = V_down.iloc[random.randint(0,num_irs_ele), :] 
                h_l_m_row = h_l_m.iloc[k, :]
                h_l_km_row = h_l_km_df_bs.iloc[random.randint(0,num_irs_ele), :] 
                V_up_value = V_up.iloc[random.randint(0,num_irs_ele), :] 
                g_l_m_row = g_l_m.iloc[k, :] 
                g_l_km_row = g_l_km_df_bs.iloc[random.randint(0,num_irs_ele), :] 
                V_har_value = V_har.iloc[random.randint(0,num_irs_ele), :] 
                f_l_m_row = f_l_m.iloc[k, :]
                f_l_km_row = f_l_km_df_bs.iloc[random.randint(0,num_irs_ele), :] 

                # Calculate Fitness
                E_ml_har_value = P_m_har_value * T_m_har_value
                h_kml_down_value=h_kml_down(V_down_value,h_l_m_row,h_l_km_row) # Pass Series
                h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
                R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                T_ml_down_value=Dm/R_ml_down_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value = D_km / f_km_value
                h_kml_up_value=h_kml_down(V_up_value,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

                R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                T_km_up_value=Dm/R_kml_up_value # equation number 5
                T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

                # Calculate fitness
                result_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

                # Store initial population data
                population.append({
                    'fitness': result_fitness,
                    'data':  {
                        'P_m_down_value': P_m_down_value,
                        'P_m_har_value': P_m_har_value,
                        'T_m_har_value': T_m_har_value,
                        'f_km_value': f_km_value,
                        'V_lm_vfly_value': V_lm_vfly_value,
                        'V_lm_hfly_value': V_lm_hfly_value,
                        'P_km_up_value':P_km_up_value,
                        'V_down_value':V_down_value,
                        'V_up_value':V_up_value,
                        'V_har_value': V_har_value,
                        }
                    })
            generations_data = []
            for j in range(num_generation):
                child_population = []
                for x in range(0, S, 2): # Loop through population with step of 2
                    if x + 1 >= len(valid_indices): # Check if i+1 is within bounds, if not break to avoid error in accessing population[x+1]
                        break
                    # Crossover
                    ranodmpopulation=[]
                    for i in range(10):
                        ranodmpopulation.append(random.choice(population))
                    ranodmpopulation = sorted(ranodmpopulation, key=lambda x: x['fitness'])
                    parent1 = ranodmpopulation[0]
                    parent2 = ranodmpopulation[1]
                    child_data = {}
                    for key in parent1['data']:
                        if key in numerical_keys_for_crossover:
                            if key in ['V_down_value','V_up_value','V_har_value']: # Handle Angle Series
                                child_data[key] = pd.Series(index=V_down.columns, dtype='float64') # Initialize empty Series for child
                                for col in V_down.columns: # Iterate through each column (angle direction)
                                    child_data[key][col] = float(parent1['data'][key][col]) * 0.6 + float(parent2['data'][key][col]) * (1 - 0.6)
                            else: # Handle scalar values
                                child_data[key] = float(parent1['data'][key]) * 0.6 + float(parent2['data'][key]) * (1 - 0.6)
                        else:
                            child_data[key] = float(parent1['data'][key]) * 0.6 + float(parent2['data'][key]) * (1 - 0.6)

                    u = np.random.uniform(0, 1, 1)[0]
                    P_mutation = 0.5
                    if u < P_mutation:
                        for key in numerical_keys_for_crossover: # Apply mutation only to numerical keys
                            if key in ['V_down_value','V_up_value','V_har_value']: # Handle Angle Series
                                for col in V_down.columns: # Iterate through each column (angle direction)
                                    child_data[key][col] += random.normal(loc=0, scale=1, size=(1))[0]
                            else:
                                child_data[key] += random.normal(loc=0, scale=1, size=(1))[0] 

                    # Compute child fitness
                    def compute_fitness(data):
                        P_m_down_value = data['P_m_down_value']
                        P_m_har_value = data['P_m_har_value']
                        T_m_har_value = data['T_m_har_value']
                        f_km_value = data['f_km_value']
                        V_lm_vfly_value = data['V_lm_vfly_value']
                        V_lm_hfly_value = data['V_lm_hfly_value']
                        P_km_up_value=data['P_km_up_value']
                        V_down_value = data['V_down_value'] 
                        V_up_value = data['V_up_value']
                        V_har_value = data['V_har_value'] 

                        # Calculate power values
                        P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                        P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                        P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

                        # Calculate time and energy values
                        T_l_vfly_value = H_value / V_lm_vfly_value
                        T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value 
                        E_ml_har_value = P_m_har_value * T_m_har_value 

                        h_kml_down_value_compute=h_kml_down(V_down_value,h_l_m_row,h_l_km_row) # Using original Angle_row, h_l_m_row, h_l_km_row for child as well - might need to be based on child data if angles are also part of optimization
                        h_ml_worst_value=h_ml_worst(h_kml_down_value_compute,sigma_km)
                        R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                        if R_ml_down_value <= 0: # check if R_ml_down_value is zero or negative
                            R_ml_down_value = 1e-9 # Assign a small positive value to avoid division by zero
                        T_ml_down_value=Dm/R_ml_down_value
                        E_ml_down_value = P_m_down_value * T_ml_down_value
                        T_km_com_value = D_km / f_km_value
                        h_kml_up_value=h_kml_down(V_up_value,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

                        R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                        T_km_up_value=Dm/R_kml_up_value # equation number 5
                        T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                        P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                        P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                        P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                        E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)
                        h_kml_har_value_compute=h_kml_down(V_har_value,f_l_m_row,f_l_km_row) # Corrected index to Angle2_row
                        E_kml_har_value=E_kml_har(P_m_har_value,T_m_har_value,h_kml_har_value_compute) # Corrected function call for E_kml_har
                        E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                        E_kml_up_value=E_kml_up(P_km_up_value,T_km_up_value)

                        # Calculate fitness
                        fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
                        current_data = {
                            'P_m_down_value': P_m_down_value,
                            'P_m_har_value': P_m_har_value,
                            'T_m_har_value': T_m_har_value,
                            'f_km_value': f_km_value,
                            'V_lm_vfly_value': V_lm_vfly_value,
                            'V_lm_hfly_value': V_lm_hfly_value,
                            'P_km_up_value':P_km_up_value,
                            'V_down_value':V_down_value,
                            'V_up_value':V_up_value,
                            'V_har_value': V_har_value,# Carry forward original index
                        }
                        if V_lm_hfly_value>0 and T_m_har_value>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value<=p_max and P_m_down_value<=p_max and P_km_up_value<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and f_km_value>0  and E_kml_har_value>=(E_kml_up_value+E_kml_com_value) and V_lm_vfly_value>0:
                            return fitness_value, current_data
                        else:
                            return  float('inf'),{} # Return empty dict instead of float('inf') for data

                    child_fitness, child_data1 = compute_fitness(child_data)
                    child_population.append({'fitness': child_fitness, 'data': child_data1})

                # Choose the best population among all population
                new_population = population + child_population
                new_population = sorted(new_population, key=lambda x: x['fitness'])
                population = new_population[:S]
                generations_data.append(population[0].copy())

            best_individual_pair = population[0].copy()
            best_individual_pair['generation'] = j + 1 # Use last j from loop
            best_individual_pair['type'] = 'GA'
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            all_best_combinations.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': population[0]['fitness'],
                'best_individual': best_individual_pair,
                'generation_fitness': [gen['fitness'] for gen in generations_data],
                'unique_row_indices': unique_row_indices # Store unique_row_indices in best_combinations
            })
    # Select the best unique Base station and UAV-IRS pair using Auction based method
    combination_lookup = {}
    for combination in all_best_combinations:
        if combination['bs_index'] not in combination_lookup:
            combination_lookup[combination['bs_index']] = {}
        combination_lookup[combination['bs_index']][combination['uav_index']] = combination

    # Auction-based assignment
    best_assignments = []
    unassigned_bs = list(range(num_bs))
    unassigned_uavs = list(range(num_uav_irs))

    while unassigned_bs and unassigned_uavs:
        best_combination_overall = None

        for l in unassigned_bs:
            best_fitness_for_bs = float('inf')
            best_combination_for_bs = None
            for k in unassigned_uavs:
                if l in combination_lookup and k in combination_lookup[l]:
                    combination = combination_lookup[l][k]
                    if combination['best_fitness'] < best_fitness_for_bs: # Use 'best_fitness' instead of 'best_individual']['fitness']
                        best_fitness_for_bs = combination['best_fitness']
                        best_combination_for_bs = combination

            if best_combination_for_bs:
                if best_combination_overall is None or best_combination_for_bs['best_fitness'] < best_combination_overall['best_fitness']: # Compare with current best overall
                    best_combination_overall = best_combination_for_bs

        if best_combination_overall:
            best_assignments.append(best_combination_overall)
            unassigned_bs.remove(best_combination_overall['bs_index'])
            unassigned_uavs.remove(best_combination_overall['uav_index'])

    # # Print and Plotting
    print(f"\n--- Best Unique UAV Assignments (Auction Based Method) ---")

    for assignment in best_assignments:
        print(f"\nBest Assignment for BS {assignment['bs_index']}:")
        print(f" UAV Index: {assignment['uav_index']}")
        best_ind = assignment['best_individual']
        print(f" Best Individual:")
        print(f"  Generation: {best_ind['generation']}, Type: {best_ind['type']}")
        print(f"  Fitness: {best_ind['fitness']:.4f}") # Print current best fitness only
        unique_indices_to_print = assignment['unique_row_indices'] # Retrieve unique_row_indices
        for key, value in best_ind['data'].items():
            if isinstance(value, pd.Series):
                print(f"  {key}: Series: \n{value.iloc[unique_indices_to_print]}") # Print sliced Series
            elif isinstance(value, list): # Handle list type values explicitly
                print(f"  {key}: {value}") # print list directly without formatting
            else:
                print(f"  {key}: {value:.4f}") # Format scalar values
        print("-" * 20)

        last_generation_fitness = assignment['generation_fitness'][-1] 
        sum_fitness_current += last_generation_fitness

    return sum_fitness_current


def GA_RA(B): # Define function to process 
    print(f"calculaiton for B for GA_RA",B)
    all_best_combinations = []
    sum_fitness_current = 0 # Initialize sum of fitness

    # Main Genetic Algorithm Loop
    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]

        unique_row_indices = range(0,50)
        h_l_km_df_bs = h_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        g_l_km_df_bs = g_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        f_l_km_df_bs = f_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        f_km_bs = f_km[unique_row_indices[0:population_size]].reset_index(drop=True)
        # Corrected line: Ensure indices are within bounds of P_km_up
        valid_indices = [i for i in unique_row_indices[0:population_size] if i < len(P_km_up)]
        P_km_up_bs = P_km_up.iloc[valid_indices].reset_index(drop=True)


        for k in range(num_uav_irs):
            population = []
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]

            Sub_value=0

            for i in range(len(valid_indices)): # Using length of valid_indices
                h_il_up_value=h_kml_down(V_up.iloc[i, :],g_l_m.iloc[k, :],g_l_km_df_bs.iloc[i, :]) # Pass Series, corrected index to i
                Sub_value+=sub(P_km_up_bs[i],h_il_up_value)

            # Initialize population
            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)
            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
            T_l_vfly_value = H_value / V_lm_vfly_value
            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
            T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value
            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)


            for i in range(S): # Using length of valid_indices
                f_km_value = f_km_bs[random.randint(0,population_size)] 
                P_km_up_value = P_km_up_bs[random.randint(0,population_size)] 

                V_down_value = V_down.iloc[random.randint(0,num_irs_ele), :] 
                h_l_m_row = h_l_m.iloc[k, :]
                h_l_km_row = h_l_km_df_bs.iloc[random.randint(0,num_irs_ele), :] 
                V_up_value = V_up.iloc[random.randint(0,num_irs_ele), :] 
                g_l_m_row = g_l_m.iloc[k, :] 
                g_l_km_row = g_l_km_df_bs.iloc[random.randint(0,num_irs_ele), :] 
                V_har_value = V_har.iloc[random.randint(0,num_irs_ele), :] 
                f_l_m_row = f_l_m.iloc[k, :]
                f_l_km_row = f_l_km_df_bs.iloc[random.randint(0,num_irs_ele), :]

                # compute the fitness
                E_ml_har_value = P_m_har_value * T_m_har_value
                h_kml_down_value=h_kml_down(V_down_value,h_l_m_row,h_l_km_row) # Pass Series
                h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
                R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                T_ml_down_value=Dm/R_ml_down_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value = D_km / f_km_value
                h_kml_up_value=h_kml_down(V_up_value,g_l_m_row,g_l_km_row) # Pass Series, using same function
                R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                T_km_up_value=Dm/R_kml_up_value # equation number 5
                T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

                # Calculate fitness
                result_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

                # Store initial population data
                population.append({
                    'fitness': result_fitness,
                    'data':  {
                        'P_m_down_value': P_m_down_value,
                        'P_m_har_value': P_m_har_value,
                        'T_m_har_value': T_m_har_value,
                        'f_km_value': f_km_value,
                        'V_lm_vfly_value': V_lm_vfly_value,
                        'V_lm_hfly_value': V_lm_hfly_value,
                        'P_km_up_value':P_km_up_value,
                        'V_down_value':V_down_value,
                        'V_up_value':V_up_value,
                        'V_har_value': V_har_value,
                        }
                    })

            generations_data = []
            for j in range(num_generation):
                child_population = []
                for x in range(0, S, 2): # Loop through population with step of 2
                    if x + 1 >= len(valid_indices): # Check if i+1 is within bounds, if not break to avoid error in accessing population[i+1]
                        break
                    # Crossover
                    ranodmpopulation=[]
                    for i in range(10):
                        ranodmpopulation.append(random.choice(population))
                    ranodmpopulation = sorted(ranodmpopulation, key=lambda x: x['fitness'])
                    parent1 = ranodmpopulation[0]
                    parent2 = ranodmpopulation[1]
                    child_data = {}
                    for key in parent1['data']:
                        if key in numerical_keys_for_crossover:
                            child_data[key] = parent1['data'][key] # Keep parent1's numerical value
                            if key in ['V_down_value','V_up_value','V_har_value']: # Ensure angles are also Series after crossover
                                child_data[key] = pd.Series([random.uniform(1, 180) for _ in range(len(V_down.columns))], index=V_down.columns)
                            else:
                                child_data[key] = float(parent1['data'][key]) * 0.6 + float(parent2['data'][key]) * (1 - 0.6)
                        else:
                            child_data[key] = float(parent1['data'][key]) * 0.6 + float(parent2['data'][key]) * (1 - 0.6)

                    u = np.random.uniform(0, 1, 1)[0]
                    P_mutation = 0.5
                    if u < P_mutation:
                        for key in numerical_keys_for_crossover: # Apply mutation only to numerical keys
                            if key in ['V_down_value','V_up_value','V_har_value']: # Handle Angle Series
                                for col in V_down.columns: # Iterate through each column (angle direction)
                                    child_data[key][col] += random.normal(loc=0, scale=1, size=(1))[0]
                            else:
                                child_data[key] += random.normal(loc=0, scale=1, size=(1))[0]

                    # Compute child fitness
                    def compute_fitness(data):
                        P_m_down_value = data['P_m_down_value']
                        P_m_har_value = data['P_m_har_value']
                        T_m_har_value = data['T_m_har_value']
                        f_km_value = data['f_km_value']
                        V_lm_vfly_value = data['V_lm_vfly_value']
                        V_lm_hfly_value = data['V_lm_hfly_value']
                        P_km_up_value=data['P_km_up_value']
                        V_down_value = data['V_down_value'] 
                        V_up_value = data['V_up_value']
                        V_har_value = data['V_har_value'] 

                        # Calculate power values
                        P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                        P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                        P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

                        # Calculate time and energy values
                        T_l_vfly_value = H_value / V_lm_vfly_value
                        T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value 
                        E_ml_har_value = P_m_har_value * T_m_har_value 

                        h_kml_down_value_compute=h_kml_down(V_down_value,h_l_m_row,h_l_km_row) # Using original Angle_row, h_l_m_row, h_l_km_row for child as well - might need to be based on child data if angles are also part of optimization
                        h_ml_worst_value=h_ml_worst(h_kml_down_value_compute,sigma_km)
                        R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                        if R_ml_down_value <= 0: # check if R_ml_down_value is zero or negative
                            R_ml_down_value = 1e-9 # Assign a small positive value to avoid division by zero
                        T_ml_down_value=Dm/R_ml_down_value
                        E_ml_down_value = P_m_down_value * T_ml_down_value
                        T_km_com_value = D_km / f_km_value
                        h_kml_up_value=h_kml_down(V_up_value,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

                        R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                        T_km_up_value=Dm/R_kml_up_value # equation number 5
                        T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                        P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                        P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                        P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                        E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)
                        h_kml_har_value_compute=h_kml_down(V_har_value,f_l_m_row,f_l_km_row) 
                        E_kml_har_value=E_kml_har(P_m_har_value,T_m_har_value,h_kml_har_value_compute) 
                        E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                        E_kml_up_value=E_kml_up(P_km_up_value,T_km_up_value)

                        # Calculate fitness
                        fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
                        current_data = {
                            'P_m_down_value': P_m_down_value,
                            'P_m_har_value': P_m_har_value,
                            'T_m_har_value': T_m_har_value,
                            'f_km_value': f_km_value,
                            'V_lm_vfly_value': V_lm_vfly_value,
                            'V_lm_hfly_value': V_lm_hfly_value,
                            'P_km_up_value':P_km_up_value,
                            'V_down_value':V_down_value,
                            'V_up_value':V_up_value,
                            'V_har_value': V_har_value,# Carry forward original index
                        }
                        if V_lm_hfly_value>0 and T_m_har_value>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value<=p_max and P_m_down_value<=p_max and P_km_up_value<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and f_km_value>0  and E_kml_har_value>=(E_kml_up_value+E_kml_com_value) and V_lm_vfly_value>0:
                            return fitness_value, current_data
                        else:
                            return  float('inf'),{} # Return empty dict instead of float('inf') for data

                    child_fitness, child_data1 = compute_fitness(child_data)
                    child_population.append({'fitness': child_fitness, 'data': child_data1})

                # Create new population
                new_population = population + child_population
                new_population = sorted(new_population, key=lambda x: x['fitness'])
                population = new_population[:S]
                generations_data.append(population[0].copy())


            best_individual_pair = population[0].copy()
            best_individual_pair['generation'] = j + 1 # Use last j from loop
            best_individual_pair['type'] = 'GA'
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            all_best_combinations.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': population[0]['fitness'],
                'best_individual': best_individual_pair,
                'generation_fitness': [gen['fitness'] for gen in generations_data],
                'unique_row_indices': unique_row_indices # Store unique_row_indices in best_combinations
            })

    # Select the best unique Base station and UAV-IRS pair using Auction based method
    combination_lookup = {}
    for combination in all_best_combinations:
        if combination['bs_index'] not in combination_lookup:
            combination_lookup[combination['bs_index']] = {}
        combination_lookup[combination['bs_index']][combination['uav_index']] = combination

    # Auction-based assignment
    best_assignments = []
    unassigned_bs = list(range(num_bs))
    unassigned_uavs = list(range(num_uav_irs))

    while unassigned_bs and unassigned_uavs:
        best_combination_overall = None

        for l in unassigned_bs:
            best_fitness_for_bs = float('inf')
            best_combination_for_bs = None
            for k in unassigned_uavs:
                if l in combination_lookup and k in combination_lookup[l]:
                    combination = combination_lookup[l][k]
                    if combination['best_fitness'] < best_fitness_for_bs: # Use 'best_fitness' instead of 'best_individual']['fitness']
                        best_fitness_for_bs = combination['best_fitness']
                        best_combination_for_bs = combination

            if best_combination_for_bs:
                if best_combination_overall is None or best_combination_for_bs['best_fitness'] < best_combination_overall['best_fitness']: # Compare with current best overall
                    best_combination_overall = best_combination_for_bs

        if best_combination_overall:
            best_assignments.append(best_combination_overall)
            unassigned_bs.remove(best_combination_overall['bs_index'])
            unassigned_uavs.remove(best_combination_overall['uav_index'])

    # Print and Plotting
    print(f"\n--- Best Unique UAV Assignments (Auction Based Method) ---")

    for assignment in best_assignments:
        print(f"\nBest Assignment for BS {assignment['bs_index']}:")
        print(f" UAV Index: {assignment['uav_index']}")
        best_ind = assignment['best_individual']
        print(f" Best Individual:")
        print(f"  Generation: {best_ind['generation']}, Type: {best_ind['type']}")
        print(f"  Fitness: {best_ind['fitness']:.4f}") # Print current best fitness only
        unique_indices_to_print = assignment['unique_row_indices'] # Retrieve unique_row_indices
        for key, value in best_ind['data'].items():
            if isinstance(value, pd.Series):
                print(f"  {key}: Series: \n{value.iloc[unique_indices_to_print]}") # Print sliced Series
            elif isinstance(value, list): # Handle list type values explicitly
                print(f"  {key}: {value}") # print list directly without formatting
            else:
                print(f"  {key}: {value:.4f}") # Format scalar values
        print("-" * 20)

        last_generation_fitness = assignment['generation_fitness'][-1] # Access the last element
        sum_fitness_current += last_generation_fitness # Sum last generation fitness values

    return sum_fitness_current

# Define a function to process 
def HC(B):
    print(f"calculaiton for B for HC_RA", B)
    all_best_combinations = []
    sum_fitness_current = 0  # Initialize sum of fitness

    # Main Hill Climbing Algorithm Loop
    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]

        unique_row_indices = range(0,50)
        h_l_km_df_bs = h_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        g_l_km_df_bs = g_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        f_l_km_df_bs = f_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        f_km_bs = f_km[unique_row_indices[0:population_size]].reset_index(drop=True)
        valid_indices = [i for i in unique_row_indices[0:population_size] if i < len(P_km_up)]
        P_km_up_bs = P_km_up.iloc[valid_indices].reset_index(drop=True)

        for k in range(num_uav_irs):
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]

            Sub_value = 0
            # Corrected loop range to use valid_indices length
            for i in range(len(valid_indices)):  # Using length of valid_indices
                h_il_up_value = h_kml_down(V_up.iloc[i, :], g_l_m.iloc[k, :], g_l_km_df_bs.iloc[i, :])  # Pass Series, corrected index to i
                Sub_value += sub(P_km_up_bs[i], h_il_up_value)

            i = 0  # Using first index for initialization
            f_km_value = f_km_bs[i]
            P_km_up_value = P_km_up_bs[i]

            V_down_value = V_down.iloc[i, :]
            h_l_m_row = h_l_m.iloc[k, :]
            h_l_km_row = h_l_km_df_bs.iloc[i, :]
            V_up_value = V_up.iloc[i, :]
            g_l_m_row = g_l_m.iloc[k, :]
            g_l_km_row = g_l_km_df_bs.iloc[i, :]
            V_har_value = V_har.iloc[i, :]
            f_l_m_row = f_l_m.iloc[k, :]
            f_l_km_row = f_l_km_df_bs.iloc[i, :]

            # Calculate power values
            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

            # Calculate time and energy values
            T_l_vfly_value = H_value / V_lm_vfly_value
            T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value
            E_ml_har_value = P_m_har_value * T_m_har_value
            h_kml_down_value = h_kml_down(V_down_value, h_l_m_row, h_l_km_row)  # Pass Series
            h_ml_worst_value = h_ml_worst(h_kml_down_value, sigma_km)
            R_ml_down_value = R_ml_down(B, P_m_down_value, h_ml_worst_value)
            T_ml_down_value = Dm / R_ml_down_value
            E_ml_down_value = P_m_down_value * T_ml_down_value
            T_km_com_value = D_km / f_km_value
            h_kml_up_value = h_kml_down(V_up_value, g_l_m_row, g_l_km_row)  # Pass Series, using same function, might need different one if logic is different

            R_kml_up_value = R_kml_up(B, P_km_up_value, h_kml_up_value, Sub_value, sigma_km)
            T_km_up_value = Dm / R_kml_up_value  # equation number 5
            T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
            E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

            # Calculate fitness for initial solution
            initial_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

            current_solution = {
                'fitness': initial_fitness,
                'data': {
                    'P_m_down_value': P_m_down_value,
                    'P_m_har_value': P_m_har_value,
                    'T_m_har_value': T_m_har_value,
                    'f_km_value': f_km_value,
                    'V_lm_vfly_value': V_lm_vfly_value,
                    'V_lm_hfly_value': V_lm_hfly_value,
                    'P_km_up_value': P_km_up_value,
                    'V_down_value': V_down_value,
                    'V_up_value': V_up_value,
                    'V_har_value': V_har_value,
                }
            }
            generations_data = []
            for j in range(num_generation): # Simulated Annealing iterations (Temperature steps)
                neighbor_solution_data = current_solution['data'].copy()
                for i in range(4): # You can keep this loop if it's intended for repeated perturbations per temperature step
                    for key in numerical_keys_for_crossover:
                        # Apply mutation only to numerical keys
                        if key in ['V_down_value','V_up_value','V_har_value']:
                            for col in V_down.columns:  # Iterate through each column (angle direction)
                                neighbor_solution_data[key][col] += random.normal(loc=0, scale=1, size=(1))[0]  # Perturb EXISTING value
                                if neighbor_solution_data[key][col] < 0:  # Check if the RESULTING angle is negative
                                    neighbor_solution_data[key][col] = abs(
                                        neighbor_solution_data[key][col])  # Take abs value of the RESULT
                        elif key in ['P_m_down_value', 'P_m_har_value', 'T_m_har_value','P_km_up_value']:
                            neighbor_solution_data[key] += random.normal(loc=0, scale=0.1, size=(1))[0]
                        elif key in ['V_lm_vfly_value', 'V_lm_hfly_value']:
                            neighbor_solution_data[key] += random.normal(loc=0, scale=1, size=(1))[0]
                        elif key in ['f_km_value']:
                            neighbor_solution_data[key] += random.normal(loc=0, scale=0.1, size=(1))[0] 
                        else:
                            neighbor_solution_data[key] = neighbor_solution_data[key] + np.random.normal(loc=0, scale=1, size=(1))[0]
           
                    # Compute neighbor fitness
                    def compute_fitness(data):  # Define compute_fitness WITHIN the generation loop - scope is fine here
                        P_m_down_value = data['P_m_down_value']
                        P_m_har_value = data['P_m_har_value']
                        T_m_har_value = data['T_m_har_value']
                        f_km_value = data['f_km_value']
                        V_lm_vfly_value = data['V_lm_vfly_value']
                        V_lm_hfly_value = data['V_lm_hfly_value']
                        P_km_up_value = data['P_km_up_value']
                        V_down_value = data['V_down_value']
                        V_up_value = data['V_up_value']
                        V_har_value = data['V_har_value']

                        # Calculate power values
                        P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                        P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                        P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

                        # Calculate time and energy values
                        T_l_vfly_value = H_value / V_lm_vfly_value
                        T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value
                        E_ml_har_value = P_m_har_value * T_m_har_value

                        h_kml_down_value_compute = h_kml_down(V_down_value, h_l_m_row,
                                                                h_l_km_row)  # Using original h_l_m_row, h_l_km_row for neighbor as well - might need to be based on neighbor data if angles are also part of optimization
                        h_ml_worst_value = h_ml_worst(h_kml_down_value_compute, sigma_km)
                        R_ml_down_value = R_ml_down(B, P_m_down_value, h_ml_worst_value)
                        if R_ml_down_value <= 0:  # check if R_ml_down_value is zero or negative
                            R_ml_down_value = 1e-9  # Assign a small positive value to avoid division by zero
                        T_ml_down_value = Dm / R_ml_down_value
                        E_ml_down_value = P_m_down_value * T_ml_down_value
                        T_km_com_value = D_km / f_km_value

                        h_kml_up_value = h_kml_down(V_up_value, g_l_m_row,
                                                      g_l_km_row)  # Pass Series, using same function, might need different one if logic is different
                        R_kml_up_value = R_kml_up(B, P_km_up_value, h_kml_up_value, Sub_value, sigma_km)
                        T_km_up_value = Dm / R_kml_up_value  # equation number 5

                        T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                        P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                        P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                        P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                        E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value,
                                                   P_l_hov_value, T_lm_hov_value)
                        h_kml_har_value_compute = h_kml_down(V_har_value, f_l_m_row,
                                                                f_l_km_row)
                        E_kml_har_value = E_kml_har(P_m_har_value, T_m_har_value,
                                                      h_kml_har_value_compute)
                        E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                        E_kml_up_value = E_kml_up(P_km_up_value, T_km_up_value)

                        # Calculate fitness
                        fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
                        current_data = {
                            'P_m_down_value': P_m_down_value,
                            'P_m_har_value': P_m_har_value,
                            'T_m_har_value': T_m_har_value,
                            'f_km_value': f_km_value,
                            'V_lm_vfly_value': V_lm_vfly_value,
                            'V_lm_hfly_value': V_lm_hfly_value,
                            'P_km_up_value': P_km_up_value,
                            'V_down_value': V_down_value,
                            'V_up_value': V_up_value,
                            'V_har_value': V_har_value,  # Carry forward original index
                        }
                        if V_lm_hfly_value > 0 and T_m_har_value > 0 and T_ml_down_value > 0 and T_km_up_value > 0 and P_m_har_value <= p_max and P_m_down_value <= p_max and P_km_up_value <= p_km_max and (
                                T_km_com_value + T_km_up_value + T_ml_down_value) <= T_m and f_km_value > 0 and E_kml_har_value >= (
                                E_kml_up_value + E_kml_com_value) and V_lm_vfly_value > 0:
                            return fitness_value, current_data
                        else:
                            return float('inf'), {}  # Return empty dict instead of float('inf') for data

                    neighbor_fitness, neighbor_data1 = compute_fitness(neighbor_solution_data)

                    # Decide whether to accept the neighbor
                    if neighbor_fitness < current_solution['fitness']:  # Assuming minimization
                        current_solution = {'fitness': neighbor_fitness, 'data': neighbor_data1}

                generations_data.append(current_solution.copy())  # Append the *updated* current_solution

            best_individual_pair = current_solution.copy()
            best_individual_pair['generation'] = j + 1  # Use last j from loop
            best_individual_pair['type'] = 'HC'
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            all_best_combinations.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': current_solution['fitness'],
                'best_individual': best_individual_pair,
                'generation_fitness': [gen['fitness'] for gen in generations_data],
                'unique_row_indices': unique_row_indices  # Store unique_row_indices in best_combinations
            })
    # Select the best unique Base station and UAV-IRS pair using Auction based method
    combination_lookup = {}
    for combination in all_best_combinations:
        if combination['bs_index'] not in combination_lookup:
            combination_lookup[combination['bs_index']] = {}
        combination_lookup[combination['bs_index']][combination['uav_index']] = combination

    # Auction-based assignment
    best_assignments = []
    unassigned_bs = list(range(num_bs))
    unassigned_uavs = list(range(num_uav_irs))

    while unassigned_bs and unassigned_uavs:
        best_combination_overall = None

        for l in unassigned_bs:
            best_fitness_for_bs = float('inf')
            best_combination_for_bs = None
            for k in unassigned_uavs:
                if l in combination_lookup and k in combination_lookup[l]:
                    combination = combination_lookup[l][k]
                    if combination['best_fitness'] < best_fitness_for_bs:  # Use 'best_fitness' instead of 'best_individual']['fitness']
                        best_fitness_for_bs = combination['best_fitness']
                        best_combination_for_bs = combination

            if best_combination_for_bs:
                if best_combination_overall is None or best_combination_for_bs['best_fitness'] < best_combination_overall[
                    'best_fitness']:  # Compare with current best overall
                    best_combination_overall = best_combination_for_bs

        if best_combination_overall:
            best_assignments.append(best_combination_overall)
            unassigned_bs.remove(best_combination_overall['bs_index'])
            unassigned_uavs.remove(best_combination_overall['uav_index'])

    # Print and Plotting
    print(f"\n--- Best Unique UAV Assignments (Auction Based Method) ---")

    for assignment in best_assignments:
        print(f"\nBest Assignment for BS {assignment['bs_index']}:")
        print(f" UAV Index: {assignment['uav_index']}")
        best_ind = assignment['best_individual']
        print(f" Best Individual:")
        print(f"  Generation: {best_ind['generation']}, Type: {best_ind['type']}")
        print(f"  Fitness: {best_ind['fitness']:.4f}")  # Print current best fitness only
        unique_indices_to_print = assignment['unique_row_indices']  # Retrieve unique_row_indices
        for key, value in best_ind['data'].items():
            if isinstance(value, pd.Series):
                print(f"  {key}: Series: \n{value.iloc[unique_indices_to_print]}")  # Print sliced Series
            elif isinstance(value, list):  # Handle list type values explicitly
                print(f"  {key}: {value}")  # print list directly without formatting
            else:
                print(f"  {key}: {value:.4f}")  # Format scalar values
        print("-" * 20)

        last_generation_fitness = assignment['generation_fitness'][-1]  # Access the last element
        sum_fitness_current += last_generation_fitness  # Sum last generation fitness values

    return sum_fitness_current

# Define a function to process 
def HC_RA(B):
    print(f"calculaiton for B for HC_RA",B)
    all_best_combinations = []
    sum_fitness_current = 0 # Initialize sum of fitness 

    # Main Hill Climbing Algorithm Loop
    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]

        unique_row_indices = range(0,50)
        h_l_km_df_bs = h_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        g_l_km_df_bs = g_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        f_l_km_df_bs = f_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        f_km_bs = f_km[unique_row_indices[0:population_size]].reset_index(drop=True)
        valid_indices = [i for i in unique_row_indices[0:population_size] if i < len(P_km_up)]
        P_km_up_bs = P_km_up.iloc[valid_indices].reset_index(drop=True)

        for k in range(num_uav_irs):
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]

            Sub_value=0
            # Corrected loop range to use valid_indices length
            for i in range(len(valid_indices)): # Using length of valid_indices
                h_il_up_value=h_kml_down(V_up.iloc[i, :],g_l_m.iloc[k, :],g_l_km_df_bs.iloc[i, :]) # Pass Series, corrected index to i
                Sub_value+=sub(P_km_up_bs[i],h_il_up_value)

            i=0 # Using first index for initialization
            f_km_value = f_km_bs[i] 
            P_km_up_value = P_km_up_bs[i] 

            V_down_value = V_down.iloc[i, :] 
            h_l_m_row = h_l_m.iloc[k, :] 
            h_l_km_row = h_l_km_df_bs.iloc[i, :] 
            V_up_value = V_up.iloc[i, :] 
            g_l_m_row = g_l_m.iloc[k, :] 
            g_l_km_row = g_l_km_df_bs.iloc[i, :] 
            V_har_value = V_har.iloc[i, :] 
            f_l_m_row = f_l_m.iloc[k, :] 
            f_l_km_row = f_l_km_df_bs.iloc[i, :] 

            # Calculate power values
            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

            # Calculate time and energy values
            T_l_vfly_value = H_value / V_lm_vfly_value
            T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value
            E_ml_har_value = P_m_har_value * T_m_har_value
            h_kml_down_value=h_kml_down(V_down_value,h_l_m_row,h_l_km_row) # Pass Series
            h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
            R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
            T_ml_down_value=Dm/R_ml_down_value
            E_ml_down_value = P_m_down_value * T_ml_down_value
            T_km_com_value = D_km / f_km_value
            h_kml_up_value=h_kml_down(V_up_value,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

            R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
            T_km_up_value=Dm/R_kml_up_value # equation number 5
            T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
            E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

            # Calculate fitness for initial solution
            initial_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

            current_solution = {
                'fitness': initial_fitness,
                'data': {
                    'P_m_down_value': P_m_down_value,
                    'P_m_har_value': P_m_har_value,
                    'T_m_har_value': T_m_har_value,
                    'f_km_value': f_km_value,
                    'V_lm_vfly_value': V_lm_vfly_value,
                    'V_lm_hfly_value': V_lm_hfly_value,
                    'P_km_up_value':P_km_up_value,
                    'V_down_value':V_down_value,
                    'V_up_value':V_up_value,
                    'V_har_value': V_har_value,
                }
            }
            generations_data = []
            for j in range(num_generation): # Hill Climbing iterations
                neighbor_solution_data = current_solution['data'].copy() 

                for i in range(10):
                    for key in numerical_keys_for_crossover:
                        if key in ['V_down_value','V_up_value','V_har_value']: # Ensure angles are also Series
                            neighbor_solution_data[key] = pd.Series([random.uniform(1, 180) for _ in range(len(V_down.columns))], index=V_down.columns)
                        elif key in ['P_m_down_value', 'P_m_har_value', 'T_m_har_value','P_km_up_value']:
                            neighbor_solution_data[key] += random.normal(loc=0, scale=0.1, size=(1))[0]
                        elif key in ['V_lm_vfly_value', 'V_lm_hfly_value']:
                            neighbor_solution_data[key] += random.normal(loc=0, scale=1, size=(1))[0]
                        elif key in ['f_km_value']:
                            neighbor_solution_data[key] += random.normal(loc=0, scale=0.1, size=(1))[0] 
                        else:
                            neighbor_solution_data[key] = neighbor_solution_data[key] + np.random.normal(loc=0, scale=1, size=(1))[0]
                        # Compute neighbor fitness 
                        def compute_fitness(data): # Define compute_fitness WITHIN the generation loop - scope is fine here
                            P_m_down_value = data['P_m_down_value']
                            P_m_har_value = data['P_m_har_value']
                            T_m_har_value = data['T_m_har_value']
                            f_km_value = data['f_km_value']
                            V_lm_vfly_value = data['V_lm_vfly_value']
                            V_lm_hfly_value = data['V_lm_hfly_value']
                            P_km_up_value=data['P_km_up_value']
                            V_down_value = data['V_down_value'] 
                            V_up_value = data['V_up_value']
                            V_har_value = data['V_har_value'] 

                            # Calculate power values
                            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

                            # Calculate time and energy values
                            T_l_vfly_value = H_value / V_lm_vfly_value
                            T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value 
                            E_ml_har_value = P_m_har_value * T_m_har_value 

                            h_kml_down_value_compute=h_kml_down(V_down_value,h_l_m_row,h_l_km_row) # Using original h_l_m_row, h_l_km_row for neighbor as well - might need to be based on neighbor data if angles are also part of optimization
                            h_ml_worst_value=h_ml_worst(h_kml_down_value_compute,sigma_km)
                            R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                            if R_ml_down_value <= 0: # check if R_ml_down_value is zero or negative
                                R_ml_down_value = 1e-9 # Assign a small positive value to avoid division by zero
                            T_ml_down_value=Dm/R_ml_down_value
                            E_ml_down_value = P_m_down_value * T_ml_down_value
                            T_km_com_value = D_km / f_km_value

                            h_kml_up_value=h_kml_down(V_up_value,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different
                            R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                            T_km_up_value=Dm/R_kml_up_value # equation number 5

                            T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                            E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)
                            h_kml_har_value_compute=h_kml_down(V_har_value,f_l_m_row,f_l_km_row) 
                            E_kml_har_value=E_kml_har(P_m_har_value,T_m_har_value,h_kml_har_value_compute) 
                            E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                            E_kml_up_value=E_kml_up(P_km_up_value,T_km_up_value)

                            # Calculate fitness
                            fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
                            current_data = {
                                'P_m_down_value': P_m_down_value,
                                'P_m_har_value': P_m_har_value,
                                'T_m_har_value': T_m_har_value,
                                'f_km_value': f_km_value,
                                'V_lm_vfly_value': V_lm_vfly_value,
                                'V_lm_hfly_value': V_lm_hfly_value,
                                'P_km_up_value':P_km_up_value,
                                'V_down_value':V_down_value,
                                'V_up_value':V_up_value,
                                'V_har_value': V_har_value, # Carry forward original index
                            }
                            if V_lm_hfly_value>0 and T_m_har_value>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value<=p_max and P_m_down_value<=p_max and P_km_up_value<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and f_km_value>0  and E_kml_har_value>=(E_kml_up_value+E_kml_com_value) and V_lm_vfly_value>0:
                                return fitness_value, current_data
                            else:
                                return  float('inf'),{} # Return empty dict instead of float('inf') for data

                        neighbor_fitness, neighbor_data1 = compute_fitness(neighbor_solution_data)

                        # Decide whether to accept the neighbor
                        if neighbor_fitness < current_solution['fitness']: # Assuming minimization
                            current_solution = {'fitness': neighbor_fitness, 'data': neighbor_data1}

                generations_data.append(current_solution.copy()) # Append the *updated* current_solution

            best_individual_pair = current_solution.copy()
            best_individual_pair['generation'] = j + 1 # Use last j from loop
            best_individual_pair['type'] = 'HC'
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            all_best_combinations.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': current_solution['fitness'],
                'best_individual': best_individual_pair,
                'generation_fitness': [gen['fitness'] for gen in generations_data],
                'unique_row_indices': unique_row_indices # Store unique_row_indices in best_combinations
            })
    # Select the best unique Base station and UAV-IRS pair using Auction based method
    combination_lookup = {}
    for combination in all_best_combinations:
        if combination['bs_index'] not in combination_lookup:
            combination_lookup[combination['bs_index']] = {}
        combination_lookup[combination['bs_index']][combination['uav_index']] = combination

    # Auction-based assignment
    best_assignments = []
    unassigned_bs = list(range(num_bs))
    unassigned_uavs = list(range(num_uav_irs))

    while unassigned_bs and unassigned_uavs:
        best_combination_overall = None

        for l in unassigned_bs:
            best_fitness_for_bs = float('inf')
            best_combination_for_bs = None
            for k in unassigned_uavs:
                if l in combination_lookup and k in combination_lookup[l]:
                    combination = combination_lookup[l][k]
                    if combination['best_fitness'] < best_fitness_for_bs: # Use 'best_fitness' instead of 'best_individual']['fitness']
                        best_fitness_for_bs = combination['best_fitness']
                        best_combination_for_bs = combination

            if best_combination_for_bs:
                if best_combination_overall is None or best_combination_for_bs['best_fitness'] < best_combination_overall['best_fitness']: # Compare with current best overall
                    best_combination_overall = best_combination_for_bs

        if best_combination_overall:
            best_assignments.append(best_combination_overall)
            unassigned_bs.remove(best_combination_overall['bs_index'])
            unassigned_uavs.remove(best_combination_overall['uav_index'])

    # Print and Plotting
    print(f"\n--- Best Unique UAV Assignments (Auction Based Method) ---")

    for assignment in best_assignments:
        print(f"\nBest Assignment for BS {assignment['bs_index']}:")
        print(f" UAV Index: {assignment['uav_index']}")
        best_ind = assignment['best_individual']
        print(f" Best Individual:")
        print(f"  Generation: {best_ind['generation']}, Type: {best_ind['type']}")
        print(f"  Fitness: {best_ind['fitness']:.4f}") # Print current best fitness only
        unique_indices_to_print = assignment['unique_row_indices'] # Retrieve unique_row_indices
        for key, value in best_ind['data'].items():
            if isinstance(value, pd.Series):
                print(f"  {key}: Series: \n{value.iloc[unique_indices_to_print]}") # Print sliced Series
            elif isinstance(value, list): # Handle list type values explicitly
                print(f"  {key}: {value}") # print list directly without formatting
            else:
                print(f"  {key}: {value:.4f}") # Format scalar values
        print("-" * 20)

        last_generation_fitness = assignment['generation_fitness'][-1] # Access the last element
        sum_fitness_current += last_generation_fitness # Sum last generation fitness values

    if best_assignments:
        avg_fitness = sum_fitness_current / len(best_assignments)
    else:
        avg_fitness = 0  # Handle the case where there are no best assignments

    return sum_fitness_current

# Define a function to process
def RS(B):
    print(f"calculaiton for B for RS",B) 
    all_best_combinations = []
    sum_fitness_current = 0 # Initialize sum of fitness

    # Main Hill Climbing Algorithm Loop
    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]

        unique_row_indices = range(0,50) # Use random.sample to select unique indices
        h_l_km_df_bs = h_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        g_l_km_df_bs = g_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        f_l_km_df_bs = f_l_km.iloc[unique_row_indices, :].reset_index(drop=True)
        f_km_bs = f_km[unique_row_indices[0:population_size]].reset_index(drop=True)
        valid_indices = [i for i in unique_row_indices[0:population_size] if i < len(P_km_up)] 
        P_km_up_bs = P_km_up.iloc[valid_indices].reset_index(drop=True)

        for k in range(num_uav_irs):
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]

            Sub_value=0
            for i in range(len(valid_indices)): # Using length of valid_indices
                h_il_up_value=h_kml_down(V_up.iloc[i, :],g_l_m.iloc[k, :],g_l_km_df_bs.iloc[i, :]) # Pass Series, corrected index to i
                Sub_value+=sub(P_km_up_bs[i],h_il_up_value)

            i=0 # Using first index for initialization
            f_km_value = f_km_bs[i] 
            P_km_up_value = P_km_up_bs[i] 

            V_down_value = V_down.iloc[i, :] 
            h_l_m_row = h_l_m.iloc[k, :] 
            h_l_km_row = h_l_km_df_bs.iloc[i, :] 
            V_up_value = V_up.iloc[i, :] 
            g_l_m_row = g_l_m.iloc[k, :] 
            g_l_km_row = g_l_km_df_bs.iloc[i, :] 
            V_har_value = V_har.iloc[i, :] 
            f_l_m_row = f_l_m.iloc[k, :] 
            f_l_km_row = f_l_km_df_bs.iloc[i, :] 

            # Calculate power values
            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

            # Calculate time and energy values
            T_l_vfly_value = H_value / V_lm_vfly_value
            T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value 
            E_ml_har_value = P_m_har_value * T_m_har_value
            h_kml_down_value=h_kml_down(V_down_value,h_l_m_row,h_l_km_row) # Pass Series
            h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
            R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
            T_ml_down_value=Dm/R_ml_down_value
            E_ml_down_value = P_m_down_value * T_ml_down_value
            T_km_com_value = D_km / f_km_value
            h_kml_up_value=h_kml_down(V_up_value,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

            R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
            T_km_up_value=Dm/R_kml_up_value # equation number 5
            T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
            E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

            # Calculate fitness for initial solution
            initial_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

            current_solution = {
                'fitness': initial_fitness,
                'data': {
                    'P_m_down_value': P_m_down_value,
                    'P_m_har_value': P_m_har_value,
                    'T_m_har_value': T_m_har_value,
                    'f_km_value': f_km_value,
                    'V_lm_vfly_value': V_lm_vfly_value,
                    'V_lm_hfly_value': V_lm_hfly_value,
                    'P_km_up_value':P_km_up_value,
                    'V_down_value':V_down_value,
                    'V_up_value':V_up_value,
                    'V_har_value': V_har_value,# Carry forward original index
                }
            }
            generations_data = []
            # Compute neighbor fitness function 
            def compute_fitness(data):
                P_m_down_value = data['P_m_down_value']
                P_m_har_value = data['P_m_har_value']
                T_m_har_value = data['T_m_har_value']
                f_km_value = data['f_km_value']
                V_lm_vfly_value = data['V_lm_vfly_value']
                V_lm_hfly_value = data['V_lm_hfly_value']
                P_km_up_value = data['P_km_up_value']
                V_down_value = data['V_down_value'] 
                V_up_value = data['V_up_value']
                V_har_value = data['V_har_value'] 

                # Calculate power values
                P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

                # Calculate time and energy values
                T_l_vfly_value = H_value / V_lm_vfly_value
                T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value 
                E_ml_har_value = P_m_har_value * T_m_har_value 

                h_kml_down_value_compute=h_kml_down(V_down_value,h_l_m_row,h_l_km_row) # Using original , h_l_m_row, h_l_km_row for neighbor as well - might need to be based on neighbor data if angles are also part of optimization
                h_ml_worst_value=h_ml_worst(h_kml_down_value_compute,sigma_km)
                R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                if R_ml_down_value <= 0: # check if R_ml_down_value is zero or negative
                    R_ml_down_value = 1e-9 # Assign a small positive value to avoid division by zero
                T_ml_down_value=Dm/R_ml_down_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value = D_km / f_km_value
                h_kml_up_value=h_kml_down(V_up_value,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different
                R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                T_km_up_value=Dm/R_kml_up_value # equation number 5
                T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)
                h_kml_har_value_compute=h_kml_down(V_har_value,f_l_m_row,f_l_km_row) 

                E_kml_har_value=E_kml_har(P_m_har_value,T_m_har_value,h_kml_har_value_compute) 
                E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                E_kml_up_value=E_kml_up(P_km_up_value,T_km_up_value)

                # Calculate fitness
                fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
                current_data = {
                    'P_m_down_value': P_m_down_value,
                    'P_m_har_value': P_m_har_value,
                    'T_m_har_value': T_m_har_value,
                    'f_km_value': f_km_value,
                    'V_lm_vfly_value': V_lm_vfly_value,
                    'V_lm_hfly_value': V_lm_hfly_value,
                    'P_km_up_value':P_km_up_value,
                    'V_down_value':V_down_value,
                    'V_up_value':V_up_value,
                    'V_har_value': V_har_value,# Carry forward original index
                    }
                if V_lm_hfly_value>0 and T_m_har_value>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value<=p_max and P_m_down_value<=p_max and P_km_up_value<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and f_km_value>0  and E_kml_har_value>=(E_kml_up_value+E_kml_com_value) and V_lm_vfly_value>0:
                    return fitness_value, current_data
                else:
                    return  float('inf'),{} # Return empty dict instead of float('inf') for data

            for j in range(num_generation*10): # Hill Climbing iterations
                neighbor_solution_data = current_solution['data'].copy() # Start with current solution for each neighbor
                for key in numerical_keys_for_crossover: # Perturb each key for the current neighbor
                    if key in ['V_down_value','V_up_value','V_har_value']:
                        neighbor_solution_data[key] = pd.Series([random.uniform(1, 180) for _ in range(len(V_down.columns))], index=V_down.columns)
                    elif key in ['P_m_down_value', 'P_m_har_value', 'T_m_har_value','P_km_up_value']:
                        neighbor_solution_data[key] = random.uniform(0, 10) 
                    elif key in ['V_lm_vfly_value', 'V_lm_hfly_value']:
                        neighbor_solution_data[key] = random.uniform(0, 100) 
                    elif key in ['f_km_value']:
                        neighbor_solution_data[key] = random.uniform(0,1) 
                    else:
                        neighbor_solution_data[key] = neighbor_solution_data[key] + np.random.normal(loc=0, scale=1, size=(1))[0]

                neighbor_fitness, neighbor_data1 = compute_fitness(neighbor_solution_data)

                # Decide whether to accept the neighbor
                if neighbor_fitness < current_solution['fitness']:
                    current_solution = {'fitness': neighbor_fitness, 'data': neighbor_data1}

                generations_data.append(current_solution.copy())

            best_individual_pair = current_solution.copy()
            best_individual_pair['generation'] = j + 1 # Use last j from loop
            best_individual_pair['type'] = 'RS' 
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            all_best_combinations.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': current_solution['fitness'],
                'best_individual': best_individual_pair,
                'generation_fitness': [gen['fitness'] for gen in generations_data],
                'unique_row_indices': unique_row_indices # Store unique_row_indices in best_combinations
            })

    # Select the best unique Base station and UAV-IRS pair using Auction based method
    combination_lookup = {}
    for combination in all_best_combinations:
        if combination['bs_index'] not in combination_lookup:
            combination_lookup[combination['bs_index']] = {}
        combination_lookup[combination['bs_index']][combination['uav_index']] = combination

    # Auction-based assignment
    best_assignments = []
    unassigned_bs = list(range(num_bs))
    unassigned_uavs = list(range(num_uav_irs))

    while unassigned_bs and unassigned_uavs:
        best_combination_overall = None

        for l in unassigned_bs:
            best_fitness_for_bs = float('inf')
            best_combination_for_bs = None
            for k in unassigned_uavs:
                if l in combination_lookup and k in combination_lookup[l]:
                    combination = combination_lookup[l][k]
                    if combination['best_fitness'] < best_fitness_for_bs: # Use 'best_fitness' instead of 'best_individual']['fitness']
                        best_fitness_for_bs = combination['best_fitness']
                        best_combination_for_bs = combination

            if best_combination_for_bs:
                if best_combination_overall is None or best_combination_for_bs['best_fitness'] < best_combination_overall['best_fitness']: # Compare with current best overall
                    best_combination_overall = best_combination_for_bs

        if best_combination_overall:
            best_assignments.append(best_combination_overall)
            unassigned_bs.remove(best_combination_overall['bs_index'])
            unassigned_uavs.remove(best_combination_overall['uav_index'])

    # Print and Plotting
    print(f"\n--- Best Unique UAV Assignments (Auction Based Method) ---")

    for assignment in best_assignments:
        print(f"\nBest Assignment for BS {assignment['bs_index']}:")
        print(f" UAV Index: {assignment['uav_index']}")
        best_ind = assignment['best_individual']
        print(f" Best Individual:")
        print(f"  Generation: {best_ind['generation']}, Type: {best_ind['type']}")
        print(f"  Fitness: {best_ind['fitness']:.4f}") # Print current best fitness only
        unique_indices_to_print = assignment['unique_row_indices'] # Retrieve unique_row_indices
        for key, value in best_ind['data'].items():
            if isinstance(value, pd.Series):
                print(f"  {key}: Series: \n{value.iloc[unique_indices_to_print]}") # Print sliced Series
            elif isinstance(value, list): # Handle list type values explicitly
                print(f"  {key}: {value}") # print list directly without formatting
            else:
                print(f"  {key}: {value:.4f}") # Format scalar values
        print("-" * 20)

        last_generation_fitness = assignment['generation_fitness'][-1] # Access the last element
        sum_fitness_current += last_generation_fitness # Sum last generation fitness values

    return sum_fitness_current

if __name__ == '__main__': # Add this to prevent issues in multiprocessing on Windows
    Bandwidth = np.arange(1, 11,1) # Power values from 1 to 10
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        fitness_sums_GA = pool.map(GA, Bandwidth)
        fitness_sums_GA_RA = pool.map(GA_RA, Bandwidth)
        fitness_sums_HC = pool.map(HC, Bandwidth)
        fitness_sums_HC_RA = pool.map(HC_RA, Bandwidth)
        fitness_sums_RS = pool.map(RS, Bandwidth)

    data_dict = {
        "Generation": Bandwidth, 
        "Fitness_Sum_HC_RA": fitness_sums_HC_RA,
        "Fitness_Sum_HC": fitness_sums_HC,
        "Fitness_Sum_GA_RA": fitness_sums_GA_RA,
        "Fitness_Sum_GA": fitness_sums_GA,
        "Fitness_Sum_RS": fitness_sums_RS,
    }

    csv_file_path_pandas = "fitness_summary Bandwidth.csv"

    # Create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(data_dict)

    plt.figure(figsize=(12, 7)) 
    Bandwidth = np.arange(1, 11,1)
    plt.rcParams["font.size"] = "20"
    plt.plot(Bandwidth, fitness_sums_GA, marker='*', linestyle='dotted',label = "C2GA")
    plt.plot(Bandwidth, fitness_sums_GA_RA, marker='s', linestyle='dotted',label = "C2GA_RA")
    plt.plot(Bandwidth, fitness_sums_HC, marker='s', linestyle='-',label = "HC")
    plt.plot(Bandwidth, fitness_sums_HC_RA, marker='*', linestyle='-',label = "HC_RA")
    plt.plot(Bandwidth, fitness_sums_RS, marker='o', linestyle='dashdot',label = "RS")
    plt.xlabel("Bandwidth",size=22)
    plt.ylabel('Energy',size=22)
    plt.legend()
    plt.savefig("Energy vs Bandwidth.pdf", format="pdf", bbox_inches="tight", dpi=800) # saved with different name
    plt.show()

    sum_energy_GA = sum(fitness_sums_GA)
    sum_energy_GA_RA = sum(fitness_sums_GA_RA)
    sum_energy_HC = sum(fitness_sums_HC)
    sum_energy_HC_RA = sum(fitness_sums_HC_RA)
    sum_energy_RS = sum(fitness_sums_RS)

    # Calculate percentage improvement over Genetic Algorithm
    improvement_GA_vs_GA_RA = ((sum_energy_GA_RA - sum_energy_GA) / sum_energy_GA_RA) * 100
    improvement_GA_vs_HC  = ((sum_energy_HC - sum_energy_GA) /sum_energy_HC) * 100
    improvement_GA_vs_HC_RA  = ((sum_energy_HC_RA - sum_energy_GA) / sum_energy_HC_RA) * 100
    improvement_GA_vs_RS  = ((sum_energy_RS - sum_energy_GA) / sum_energy_RS) * 100
    
    
    new_row = {"Generation":"Fitness Improvement(%)", "Fitness_Sum_HC_RA":improvement_GA_vs_HC_RA, "Fitness_Sum_HC":improvement_GA_vs_HC, "Fitness_Sum_GA_RA":improvement_GA_vs_GA_RA, "Fitness_Sum_GA":0, "Fitness_Sum_RS":improvement_GA_vs_RS}
    new_row_df = pd.DataFrame([new_row])
    df2 = pd.concat([df, new_row_df], ignore_index=True) # Save the DataFrame to a CSV file
    df2.to_csv(csv_file_path_pandas, index=False) # index=False to prevent writing row indices to CSV

    #Print the calculated improvement values with descriptive labels
    print(f"Improvement of GA vs GA_RA: {improvement_GA_vs_GA_RA:.2f}%")
    print(f"Improvement of GA vs HC: {improvement_GA_vs_HC:.2f}%")
    print(f"Improvement of GA vs HC_RA: {improvement_GA_vs_HC_RA:.2f}%")
    print(f"Improvement of GA vs RS: {improvement_GA_vs_RS:.2f}%")