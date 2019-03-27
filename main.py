import time
import os
import re
import numpy as np 
import pandas as pd
import copy
from pulp import *
from math import sqrt
import matplotlib.pyplot as plt
from itertools import product
try:
    import pickle
except:
    import cPickle as pickle
    
import fl_model_resources as fl

#########################################################################
#########################################################################
#######################   SETTING  PARAMETERS   #########################
#########################################################################
#########################################################################
'''
In this part of the file the setup for the model, please scroll until the
"Solving model" section and make sure that all parameters are correctly
configured for your setup.
Directories can be relative paths to the location of this script.
'''

# Choose where to store the results of the program. 
# The folder must exist beforehand
current_area = 'Southampton'
results_folder = '.\\' + current_area + '_results\\'

# In this directory, a folder with the name below will be created, with a resutls table:
# Additionally, files named "sc_1.txt", "sc_2.txt", etc. will be created with the description
# of each scenario
output_filename = 'all_scenarios.txt'

## Input data:
# We expect three subdirectories in the main Data folder: "Census", "GPs", "Pharmacies"
# each of them containing one of the following files, whose names names start by
# the value of the variable "current_area" and trailing the following:
#   · '_gps.csv' - In the "GPs" folder, file with information of GPs and surgeries locations
#   · '_pharmacies.csv' - In the "Pharmacies" folder, file with information of pharmacy locations
#   · '_census.csv' - In the "Census" folder, a file with the census information of the area
# More details of the files below.

# Indicate the main directory the data is stored
data_directory = r'..\Data\\'

# GP locations file
# Should be a CSV file refering to the GP surgeries in the desired region.
# It must contain, at least, the following headers:
# "X"  - Longitude
# "Y"  - Latitude
# "Name"   - Name or unique identifier for the GP surgery
# Please see the online appendix for details on how to obtain this data
gp_locations_file = os.path.join(data_directory + 'GPs', current_area + '_gps.csv')

# Pharmacy locations
# Should be a CSV file refering to the pharmacies in the desired region.
# It must contain, at least, the following headers:
# "X"  - Longitude
# "Y"  - Latitude
# "NamePharm"   - Name or unique identifier for the pharmacy
# Please see the online appendix for details on how to obtain this data
pharmacy_locations_file = os.path.join(data_directory + 'Pharmacies', current_area + '_pharmacies.csv')

# Census file
# CSV file containing the Census output areas or the region, containing their
# population weighted centroids and their population count
#   · "X"   - Longitude of the centroid
#   · "Y"   - Latitude of the centroid
#   · "All Ages"    - Population count for all ages
#   · "oa11cd"  - Output Area code
censusFile = os.path.join(data_directory + 'Census', current_area + '_census.csv')


# Loading instances from disk:
# The script will look for a file called current_area + '.p'
# If false, distances are calculated in first iteration and saved as current_area + '.p',
# loadFromDisk is set to true for the remaining iterations
loadFromDisk = False 

# OSRM configuration
# This script depends on OSRM and openstreetmap data. You need access to a listening
# OSRM server. In our study, this was a local server configured to provide
# walking distances. The OSRM version used was v5.18.0 and CH algorithm.
# The script makes use of this tool to access OSRM from Python: https://github.com/ustroetz/python-osrm
# More info on OSRM servers: https://github.com/Project-OSRM/osrm-backend 
osrm_server = 'http://localhost:5001/'

# Scenarios to solve:
# The script is configured to replicate the results in the article, i.e. all configurations
# for a number of parameters for the model:
#   · testing_demands_list = [0.000230091, 0.0048]
#   · max_travel_values_list = [1, 500, 1000, 2000]
#   · machine_capacity_list = [35*5]
#   · allowed_facilites_list = ['BOTH', 'GP', 'PHARMACY']
# If you wish to run a scenario only, provide only one element in each list.

testing_demands_list = [0.000230091, 0.0048]
max_travel_values_list = [1, 500, 1000, 2000]
machine_capacity_list = [35*5]
allowed_facilites_list = ['BOTH', 'GP', 'PHARMACY']

# Solver configuration
# The script uses pulp, so should be able to use a number of different solvers 
# (including free solvers) however, it has only been tested with Gurobi.
# Please, rerfer to Pulp documentation if you wish to use a different solver
# and how to set time limits and gaps for it
gurobi_time_limit = 3600
gurobi_gap = 1e-18



#########################################################################
#########################################################################
####################### START SOLVING THE MODEL #########################
#########################################################################
#########################################################################
# Move the working directory to find the files in relative paths
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Read facility files with pandas:
gp_branches_csv = pd.read_csv(gp_locations_file ,index_col=None, header=0)
pharmacies_csv = pd.read_csv(pharmacy_locations_file ,index_col=None, header=0)

# Read the census:
cl , maxHeadCount, minHeadCount = fl.read_census_csv(censusFile)

# From this file, we want to set all these variables:
GPLocations = []
GPNames = []
avLat = gp_branches_csv['Y'].mean()
avLon = gp_branches_csv['X'].mean()
validGPs = 0
markerList = []
potentialLocations = []
facPoints = []
facTypes = []
nameIdx = 0
maxLength = 70 # Maximum number of characters for names
for idx,row in gp_branches_csv.iterrows():
    GPLocations.append([float(row['Y']), float(row['X'])])
    cleanName = str(nameIdx) + re.sub('[^A-Za-z0-9]+', '_', row['Name'])
    if len(cleanName) > maxLength:
        cleanName = cleanName[:maxLength]
    GPNames.append(cleanName)
    potentialLocations.append(cleanName)
    facTypes.append('GP')
    nameIdx += 1



totalGPs = len(GPNames)
print('There are ' + str(totalGPs) + ' GP practices on the file ' + gp_locations_file)


# Same for pharmacies
pharmacyLocations = []
pharmacyNames = []
for idx,row in pharmacies_csv.iterrows():
    pharmacyLocations.append([float(row['Y']), float(row['X'])])
    cleanName = str(nameIdx) + re.sub('[^A-Za-z0-9]+', '_', row['NamePharm'])
    pharmacyNames.append(cleanName)
    potentialLocations.append(cleanName)
    facTypes.append('PHARMACY')
    nameIdx += 1

totalPharmacies = len(pharmacyNames)
print('There are ' + str(totalPharmacies) + ' pharmacies on the file ' + pharmacy_locations_file)

facPoints = GPLocations + pharmacyLocations

f = open(os.path.join(results_folder,  output_filename), 'w')
f.write('\n-----\n')
currentTimeStamp = str(time.ctime(time.time()))
f.write('Started run at: ' + currentTimeStamp + '\n')
f.write('Census data: ' + '"' + censusFile + '"\n')
f.write('GP data: ' + '"' + gp_locations_file + '"\n')
f.write('Pharmacy data: ' + '"' + pharmacy_locations_file + '"\n')

f.write('Scenario\t"WCap"\tFtype\tIncidence\t\tMaxT\tAvgT\tWorstT\tPatientsInRange\tObjvalue\tObjStatus"\tMachines\tGP\tPhar\tAMUt\tMMUt\tTime')
f.close()

# Prepare scenarios:
scenarios = product(testing_demands_list, machine_capacity_list, allowed_facilites_list, max_travel_values_list)
totalSC = len(machine_capacity_list)*len(allowed_facilites_list)*len(testing_demands_list)*len(max_travel_values_list)
print('There are ' + str(totalSC) + ' scenarios.')

print('Solver configuration:\n\tTime limit: ' + str(gurobi_time_limit) + ' s \n\tAllowed gap: ' + str(gurobi_gap*100) + '%')
print('Start scenario solving...')
for sccc,currentScenario in enumerate(scenarios):
    start = time.time()
    instanceName = 'sc_' + str(sccc)
    dist_matrix_file = os.path.join(results_folder, current_area + '.p')
    
    print('Scenario "' + instanceName + '" / ' + str(sccc) + ' of ' + str(totalSC))
    machineCapacity = currentScenario[1]
    usedType = currentScenario[2]
    MAX_TRAVEL_VALUE = currentScenario[3]
    incidenceValue = currentScenario[0]
    if MAX_TRAVEL_VALUE < 2 and usedType in ['PHARMACY', 'BOTH']:
        print('Skipped, trivial.')
        continue
    # Update the demand for tests with this value:
    for	cpoint in cl:
        cpoint.CRPDemand = incidenceValue*cpoint.totalPopulation


    scDescription = str(currentScenario)
    print('Solving for: \n' + scDescription + '\n')


    # print('machineCapacity = ' + str(machineCapacity))
    # print('usedType = ' + str(usedType))
    # print('incidenceValue = ' + str(incidenceValue))
    # print('MAX_TRAVEL_VALUE = ' + str(MAX_TRAVEL_VALUE))

    if loadFromDisk:
        print('Loading distances from disk...')
        clb = pickle.load(open(dist_matrix_file, 'rb'))
        for ii,c in enumerate(cl):
            c.asBurden = clb[ii].asBurden 
            c.asName = clb[ii].asName
            c.walkingTimesToGPs = clb[ii].walkingTimesToGPs
            c.walkingTimesToPharmacies = clb[ii].walkingTimesToPharmacies
            c.closestGP = clb[ii].closestGP
            c.closestGPName = clb[ii].closestGPName
            c.walkingTimesToPharmacies = clb[ii].walkingTimesToPharmacies
    else:
        print('Calculating distances from server: "{0}"'.format(osrm_server))
        fl.travel_estimations_osrm(cl, GPLocations, GPNames, pharmacyLocations, pharmacyNames, server=osrm_server)
        pickle.dump(cl, open(dist_matrix_file, "wb" ))
        print('Done. File saved to: "{0}"'.format(dist_matrix_file))
        loadFromDisk = True

    print('Done.')
    # This part cannot be pickled, so do it here:
    totalDemand = 0
    maxMinBurden = -1
    comesFrom = -1
    goesTo = -1
    listOfBurdenWords = []
    debugPRINT = False
    demandMinBurden = []
    countViolated = 0
    for jj,c in enumerate(cl):
        # Calculate base travel:
        baseTravel = 2*c.walkingTimesToGPs[c.closestGP]

        if debugPRINT:
            print('Processing: ' + str(jj) + ' - OA: ' + str(c.OACode))
            print('Base travel: ' + str(baseTravel) )
        minBurden = 10000000000000000
        demandMinBurden.append(10000000000000000)
        GT = ''
        # countLoc = -1
        for i,xxx in enumerate(c.asBurden):
            # countLoc += 1
            c.asBurden[i] = xxx - baseTravel

            # Correct numerical deviations:
            if c.asBurden[i] < 0 and c.asBurden[i] > -1:
                c.asBurden[i] = 0

            if (c.asBurden[i]< 0):
                print('ERROR negative burden')
                print(c.asBurden[i])
                print('Extended:')
                print(xxx)
                print('Base')
                print(baseTravel)
                print('Location')
                print(str(c.coords.lat) + ', ' + str(c.coords.long))
                print('Closest GP:')
                print(GPLocations[c.closestGP])
                print('Assigned place')
                print(facPoints[i])
            validLoc = True
            trailBit = ''

            if usedType.lower() != "both" and (usedType.lower() != facTypes[i].lower()):
                validLoc = False
            if c.asBurden[i] < minBurden and validLoc:
                minBurden = c.asBurden[i]
                GT = i
            if c.asBurden[i] < demandMinBurden[jj] and validLoc:
                demandMinBurden[jj] = c.asBurden[i]

        if demandMinBurden[jj] < MAX_TRAVEL_VALUE:
            demandMinBurden[jj] = MAX_TRAVEL_VALUE
        else:
            countViolated += 1

        if minBurden > maxMinBurden:
            maxMinBurden = minBurden
            comesFrom = jj
            goesTo = GT

        c.travelDistances = makeDict([c.asName], c.asBurden, 0)
        for i,xx in enumerate(c.asBurden):
            if xx < 0:
                print('ERROR IN THIS DISTANCE.' + str(i))
        totalDemand += c.CRPDemand
    minBurden = maxMinBurden

    print('There are ' + str(countViolated) + ' points over ' + str(MAX_TRAVEL_VALUE) + ' m the worst is ' + str(max(demandMinBurden)))
    demandPoints = []
    for i,c in enumerate(cl):
        demandPoints.append('L_' + str(i))


    ####################### ##### START #### #########################
    ####################### #####  ILP  #### #########################
    ####################### ##### MODEL #### #########################
    nServicePoints = len(potentialLocations)
    nDemandPoints = len(demandPoints)

    # Preparation:
    prob = LpProblem("Facility location binary model",LpMinimize)

    # Variables: 
    y = LpVariable.dicts("Y",(potentialLocations),0,None,LpInteger) # Number of potentialLocations to open
    x = LpVariable.dicts("X",(demandPoints, potentialLocations),0,None, LpBinary) 

    totalTravel = LpVariable("TT",0)
    fOpen = LpVariable("SUMOFY",0)

    machWeeklyCap = machineCapacity
    maxTravelEver = 0
    for ii,c in enumerate(cl):
        maxTravelEver += demandMinBurden[ii]*c.CRPDemand

    print('maxTravelEver = ' + str(maxTravelEver))


    # Objective function:
    scaleFactor = 1
    prob += maxTravelEver*fOpen + totalTravel 

    if usedType.lower() != "both":
        if usedType.lower() == "gp":
            for lll in pharmacyNames:
                prob += y[lll] == 0, "Forbidden facility " + str(lll)
        elif usedType.lower() == "pharmacy":
            for lll in GPNames:
                prob += y[lll] == 0, "Forbidden facility " + str(lll)

    ### Constraints ###

    # Count the facilites opened:
    prob += fOpen == lpSum([y[f]] for f in potentialLocations), "Facilities opened"

    # Ensure travel limit is respected
    for i,c in enumerate(cl):
        constraintName = "MBFac_" + str(i) + "_is_" + str(round(demandMinBurden[i],1)).replace('.','_')
        prob += 0 >= lpSum([x[demandPoints[i]][f]*c.travelDistances[f]] for f in potentialLocations) - demandMinBurden[i], constraintName

    # Capacity of a facility
    for f in potentialLocations:
        prob += lpSum([x[p][f]*cl[i].CRPDemand] for (i,p) in enumerate(demandPoints)) <= y[f]*machWeeklyCap, f + " capacity"

    # Assign all demand points
    for i,c in enumerate(cl):
        prob += lpSum([x[demandPoints[i]][f]] for f in potentialLocations) == 1, "Cover demand at L_" + str(i) + ", OA " + c.OACode

    # Total travel:
    prob += totalTravel == lpSum([lpSum([x[demandPoints[i]][f]*c.CRPDemand*c.travelDistances[f]] for f in potentialLocations)] for i,c in enumerate(cl))



    # The problem data is written to an .lp file
    # prob.writeLP(os.path.join(results_folder, instanceName + "_test_model.lp"))

    print('-----------------------------------')
    print('           START  SOLVER           ')
    print('-----------------------------------')

    prob.solve(GUROBI(epgap = gurobi_gap, timeLimit=gurobi_time_limit))

    ####################### ##### FINISHED #### #########################
    ####################### #####    ILP   #### #########################
    ####################### #####   MODEL  #### #########################


    if LpStatus[prob.status] == "Infeasible":
        print('MODEL WAS INFEASIBLE!')
        f = open(os.path.join(results_folder, output_filename), 'a+')
        f.write('\n"' + str(sccc) + '"\t')
        # Scenario info
        f.write(str(machineCapacity) + '\t\t')
        f.write(str(usedType) + '\t')
        f.write(str(incidenceValue) + '\t')
        f.write(str(MAX_TRAVEL_VALUE) + '\t')
        # Solution info
        f.write('---\t---\t---\t---\t"' + str(LpStatus[prob.status]) + '"\t---\t---\t---\t')
        end = time.time()
        elpstime = end - start
        f.write(str(elpstime) + '\t')
        f.close()
        continue


    # The optimised objective function value is printed to the screen    
    obValue = float(value(prob.objective))
    print("Total Costs = ", value(prob.objective))
    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    fCap = []
    fAssi = []
    fUs= []
    totalMachines = 0
    totalF = 0
    totalGP = 0
    avUt = 0
    maxUt = 0
    avutden = 0
    allUtilisationsStr = ''
    for i,loc in enumerate(potentialLocations):
        nmachines = round(y[loc].varValue, 0)
        totalMachines += nmachines
        if loc in GPNames:
            totalGP += nmachines
        else:
            totalF += nmachines
        capacity = 0
        assigned = 0
        utilisation = 0
        if nmachines > 0:
            capacity = nmachines*machWeeklyCap
            utilisation = 0
            # print('\tNOT USED')
            allUtilisationsStr += '\t' + str(loc)
            allUtilisationsStr += '\n\t\tCapacity: ' + str(capacity)
            astotal = 0
            for j,dem in enumerate(demandPoints):
                astotal += x[dem][loc].varValue*cl[j].CRPDemand
            assigned = astotal
            utilisation = astotal/(nmachines*machWeeklyCap)*100
            allUtilisationsStr += '\n\t\tAssigned: ' + str(astotal)
            allUtilisationsStr += '\n\t\tUsage: ' + str(round(utilisation,2)) + '\n'
            avUt += utilisation
            if utilisation > maxUt:
                maxUt = utilisation
            avutden += 1
        fCap.append(capacity)
        fAssi.append(assigned)
        fUs.append(utilisation)


    avUt = avUt/avutden
    worstT = 0
    numerador = 0
    denominador = 0
    percNotInRange = 0
    for j,dem in enumerate(demandPoints):
        jtravel = 0

        for i,loc in enumerate(potentialLocations):
            if round(x[dem][loc].varValue,0) > 0.1:
                jtravel = cl[j].travelDistances[loc]
                numerador += jtravel*cl[j].CRPDemand
                denominador += cl[j].CRPDemand
                if worstT < jtravel:
                    worstT = jtravel
        if jtravel > MAX_TRAVEL_VALUE:
            percNotInRange += cl[j].CRPDemand

        # print('\tOA ' + str(j) + ' travels ' + str(jtravel))
        if	jtravel < 0:
            print("ERROR: NEGATIVE TRAVEL.")
    percNotInRange = 1 - percNotInRange/totalDemand
    print('\n --------------- \n')

    print('Average travel: ' + str(numerador/denominador))
    print('Worst travel: ' + str(worstT))

    # exit(-1)

    # File
    f = open(results_folder + instanceName + '_text.txt', 'w')
    f.write('Scenario description: ' + scDescription + '\n')
    f.write('Objective function: ' + str(obValue) + '\n')
    f.write('maxTravelEver: ' + str(maxTravelEver) + '\n')
    f.write('scaleFactor: ' + str(scaleFactor) + '\n')
    f.write('Average travel: ' + str(numerador/denominador) + '\n')
    f.write('Worst travel: ' + str(worstT) + '\n')
    f.write("Total Costs: " + str(value(prob.objective)) + '\n')
    f.write("Status: " + str(LpStatus[prob.status]) + '\n')
    f.write("totalMachines: " + str(totalMachines) + '\n')
    f.write('avUt: ' + str(avUt) + '\t')
    f.write('maxUt: ' + str(maxUt) + '\t')
    end = time.time()
    elpstime = end - start
    f.write('Time: ' + str(elpstime) + '\n')
    # Each of the variables is printed with it's resolved optimum value
    varValues = ''
    for v in prob.variables():
        if v.varValue > 0 or v.name[0] == 'Y':
            varValues += '\t' + str(v.name) + " = " + str(round(v.varValue, 0)) + '\n'

    f.write('Variable values: (All Y variables and non-zero X variables):\n' + varValues)
    f.write('Machine utilisations: \n' +  allUtilisationsStr)
    f.close()

    f = open(os.path.join(results_folder, output_filename), 'a+')
    f.write('\n"' + str(sccc) + '"\t')
    # Scenario info
    f.write(str(machineCapacity) + '\t\t')
    f.write(str(usedType) + '\t')
    f.write(str(incidenceValue) + '\t')
    f.write(str(MAX_TRAVEL_VALUE) + '\t')
    # Solution info
    f.write(str(numerador/denominador) + '\t')
    f.write(str(worstT) + '\t')
    f.write(str(percNotInRange) + '\t')
    f.write(str(value(prob.objective)) + '\t')
    f.write('"' + str(LpStatus[prob.status]) + '"\t')
    f.write(str(int(totalMachines)) + '\t')
    f.write(str(int(totalGP)) + '\t')
    f.write(str(int(totalF)) + '\t')
    f.write(str(avUt) + '\t')
    f.write(str(maxUt) + '\t')
    f.write(str(elpstime) + '\t')
    f.close()


f = open(os.path.join(results_folder, output_filename), 'a+')
currentTimeStamp = str(time.ctime(time.time()))
f.write('\nFinished run at: ' + currentTimeStamp + '\n---\n')
f.close()


print('Program finished.')