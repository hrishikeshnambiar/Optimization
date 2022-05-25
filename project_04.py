# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:39:21 2022

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

#Import Data from the data file

data = pd.read_excel ('I:\Sem 3\operation\Project\Data_input.xlsx')
data['Final Demand']= data['AC(kW)']+data['DC(kW)']

input_data =pd.read_excel ('I:\Sem 3\operation\Project\Data_input.xlsx', sheet_name='Data Input')

p = []
j = 0
i=0
input_data['Demand']=0
while j in range(0,len(input_data)-1):
    
    for i in range(i,i+6):
        p.append(float(data['Final Demand'][i]))
        i=i+1
    input_data['Demand'][j]= float(sum(p)/6)
    p[:]=[]
    j=j+1
N=0 # no. of EV
c=21 #battery capacity, kWh
f=3 #maximum electricity charge per unit time of EV, kW
chargingEfficiency = 0.95
g=12.5 #maximum electricity discharge per unit time of EV, kW
dischargingEfficiency = 0.95
c_w = 0 #(CO2e kg/kWh) wind
c_pv = 0 #(CO2e kg/kWh) PV
c_ch = 0 #(CO2e kg/kWh) Charging
c_dc = 0 #(CO2e kg/kWh) Discharging
c_grid = 1.05 #(CO2e kg/kWh) Grid
# %%  Optimization

def CostOpt(N):
    
    #Initialization of variables
    
    model = pyo.ConcreteModel()
    model.i = pyo.RangeSet(0, 23) #i
    model.x = pyo.Var(model.i,domain=pyo.Binary) #x-binary
    model.y = pyo.Var(model.i,domain=pyo.Binary) #y-binary
     
    model.P_grid = pyo.Var(model.i, domain=pyo.NonNegativeReals) #Grid
    model.P_w = pyo.Var(model.i, domain=pyo.NonNegativeReals) #wind
    model.P_pv = pyo.Var(model.i, domain=pyo.NonNegativeReals) #PV
    model.P_c = pyo.Var(model.i, domain=pyo.NonNegativeReals) #Charged
    model.P_d = pyo.Var(model.i, domain=pyo.NonNegativeReals) #Discharged
    model.EG = pyo.Var(model.i, domain=pyo.NonNegativeReals) #Excess Generated
    model.co2 = pyo.Var(model.i, domain=pyo.NonNegativeReals) #co2
    model.storage = pyo.Var(model.i, domain=pyo.NonNegativeReals) #Storage
    
    
    #Constraints
    
    #1. kirchoff's law
    def constraint01(model,i):
        return input_data['Demand'][i] + model.P_c[i] + model.EG[i] == model.P_grid[i] + model.P_w[i] + model.P_pv[i] + model.P_d[i]  #for i in model.i
    model.constraint01 = pyo.Constraint(model.i, rule=constraint01)
    
    #2. P_wind 
    def constraint02(model,i):
        return model.P_w[i] <= input_data['Pw'][i]
    model.constraint02 = pyo.Constraint(model.i, rule=constraint02)
    
    #3. P_pv 
    def constraint03(model,i):
        return model.P_pv[i] == input_data['Pv'][i]
    model.constraint03 = pyo.Constraint(model.i, rule=constraint03)
       
    #4. P_storage 
    def constraint04(model,i):
         return model.storage[i] <= c*N #kW
    model.constraint04 = pyo.Constraint(model.i, rule=constraint04)
     
    #5. P_discharge 
    def constraint05(model,i):
         return model.P_d[i] <= g*N*model.x[i] #kW
    model.constraint05 = pyo.Constraint(model.i, rule=constraint05)
    
    #6. P_charge      
    def constraint06(model,i):  
         return model.P_c[i] <= f*N*model.y[i] #kW
    model.constraint06 = pyo.Constraint(model.i, rule=constraint06)
    
    #7. Binary
    def constraint07(model,i):
         return model.x[i] + model.y[i] <= 1 #kW
    model.constraint07 = pyo.Constraint(model.i, rule=constraint07)
    
    #8. P_charge[t] <= P_battery[t-1] 
    def constraint08(model,i):
        if i==0:
             return model.P_d[i] <= c*N*0.5   
        else:
             return model.P_d[i] <= model.storage[i-1] 
    model.constraint08 = pyo.Constraint(model.i, rule=constraint08)
    
    #9. P_charge[t] + P_battery[t-1] <= 200W
    def constraint09(model,i):
        if i==0:
             return model.P_c[i] + c*N*0.5  <= c*N
        else:
             return model.P_c[i] + model.storage[i-1] <= c*N
    model.constraint09 = pyo.Constraint(model.i, rule=constraint09)
    
    #10. P_battery[t] = P_battery[t-1] - P_discharge[t] + P_charge[t]
    def constraint10(model,i):
        if i==0:
             return model.storage[i] == c*N*0.5 - model.P_d[i]/dischargingEfficiency + model.P_c[i]*chargingEfficiency
        else:
             return model.storage[i] == model.storage[i-1] - model.P_d[i]/dischargingEfficiency + model.P_c[i]*chargingEfficiency
             
    model.constraint10 = pyo.Constraint(model.i, rule=constraint10)
    

    #Objective Function
    
    def co2_rule(model, i):
        return model.co2[i] ==  model.P_grid[i]*c_grid + model.P_w[i]*c_w + model.P_pv[i]*c_pv - model.P_c[i]*c_ch + model.P_d[i]*c_dc 
    model.co2_rule = pyo.Constraint(model.i, rule=co2_rule)
     
    def ObjRule(model):
       return pyo.summation(model.co2)
    model.obj = pyo.Objective(rule=ObjRule, sense=pyo.minimize) 
     
    opt = SolverFactory("gurobi", solver_io="python")
    opt.solve(model)
    return model
 
def get_values(model):
    P_w = []
    P_pv = []
    P_grid = []
    P_c = []
    P_d = []
    SOC = []
    EG = []
    co2 = []
    x = []
    y = []
    for i in range(0,24):
        P_w.append(model.P_w[i].value)
        P_pv.append(model.P_pv[i].value)
        P_grid.append(model.P_grid[i].value)
        P_c.append(model.P_c[i].value)
        P_d.append(model.P_d[i].value)
        SOC.append(model.storage[i].value)
        EG.append(model.EG[i].value)
        co2.append(model.co2[i].value)
        x.append(model.x[i].value)  
        y.append(model.y[i].value)

    return P_w,P_pv,P_grid, P_c, P_d, EG, co2,x,y,SOC
#results_0 for no battery
model = CostOpt(0)

#Getting the values from the model:
P_w,P_pv,P_grid, P_c, P_d, EG, co2,x,y,SOC = get_values(model)
results_0 = pd.DataFrame({'Wind':P_w,'PV':P_pv,'Grid':P_grid, 'CO2':co2, 'Charging':P_c, 'Discharging':P_d,
                        'Excess':EG,'X':x,'Y':y,'SOC':SOC})

#results_5 for no battery
model = CostOpt(5)

#Getting the values from the model:
P_w,P_pv,P_grid, P_c, P_d, EG, co2,x,y,SOC = get_values(model)
results_5 = pd.DataFrame({'Wind':P_w,'PV':P_pv,'Grid':P_grid, 'CO2':co2, 'Charging':P_c, 'Discharging':P_d,
                        'Excess':EG,'X':x,'Y':y,'SOC':SOC})
#results_10 for no battery
model = CostOpt(10)

#Getting the values from the model:
P_w,P_pv,P_grid, P_c, P_d, EG, co2,x,y,SOC = get_values(model)
results_10 = pd.DataFrame({'Wind':P_w,'PV':P_pv,'Grid':P_grid, 'CO2':co2, 'Charging':P_c, 'Discharging':P_d,
                        'Excess':EG,'X':x,'Y':y,'SOC':SOC})

# %% Plot 1
fig, ax = plt.subplots()   

x= input_data['i']
y1= input_data['Pw']
y2= input_data['Pv']
ax.plot
fig, ax = plt.subplots(figsize=(8,6))
plt.plot(x,y1)
plt.plot(x,y2)

plt.ylabel('kW') 
plt.xlabel('Time (hrs)')
plt.legend(['Wind Limit','Pv Limit'])
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('Inputlimits.png', dpi=1000)
plt.show()  
# %% Plot 2
fig, ax = plt.subplots()   

x_out= input_data['i']
y1_out= input_data['Demand']*1.05
y2_out= results_0['CO2']
y3_out= results_5['CO2']
y4_out= results_10['CO2']
ax.plot
fig, ax = plt.subplots(figsize=(8,6))
plt.plot(x_out,y1_out)
plt.plot(x_out,y2_out)
plt.plot(x_out,y3_out)
plt.plot(x_out,y4_out)
plt.ylabel('Kg of Co2') 
plt.xlabel('Time (hrs)')
plt.legend(['Only Grid','No EVs','5 EVs','10 EVs'])
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('Output_1.png', dpi=1000)
plt.show()  

# %% Total
Net_emission_1 = sum(y1_out)
Net_emission_2 = sum(y2_out)
Net_emission_3 = sum(y3_out)
Net_emission_4 = sum(y4_out)

writer = pd.ExcelWriter('I:\Sem 3\operation\Project\Data_output.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
results_0.to_excel(writer, sheet_name='No Evs')
results_5.to_excel(writer, sheet_name='5 Evs')
results_10.to_excel(writer, sheet_name='10 Evs')

# Close the Pandas Excel writer and output the Excel file.
writer.save()