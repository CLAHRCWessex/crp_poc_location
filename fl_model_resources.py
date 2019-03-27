import copy
import operator
import time
import numpy as np
import pandas as pd
from pulp import *
import osrm

class GEOPOINT(object):
    def __init__(self, latitude, longitude):
        self.lat = latitude
        self.long = longitude
    def latlong(self):
        return([self.lat, self.long])
    def longlat(self):
        return([self.long, self.lat])

class DEMAND_POINT(object):
    """docstring for DEMAND_POINT"""
    def __init__(self):
        self.coords = GEOPOINT(0,0)
        self.populationProfile = [] # From census
        self.CRPDemand = 0
        self.totalPopulation = 0
        self.closestGP = -1
        self.closestPharmacy = -1
        self.closestGPName = ''
        self.closestPharmacyName = ''
        self.OACode = ''
        self.travelDistances = [] # Depending on what facility the test is undertaken
        self.asName = []
        self.asBurden = []
        
        self.walkingTimesToGPs = []
        self.walkingTimesToPharmacies = []
        # After solution
        self.assignedTo = ''


def excel_sheet_to_df(excelFile, sheetName='Sheet1'):
    # name = excelFile.split('.')[0]

    # Open file and check quality:
    xl = pd.ExcelFile(excelFile)

    # Check the sheet names
    if sheetName not in xl.sheet_names:
        print('ERROR: The program expects one sheet from ' + excelFile +
         ' to be called "' + sheetName + '", instead we got:' + str(xl.sheet_names) + '.\nProgram terminated.')
        exit(-1)

    pdDataFrame = xl.parse(sheetName)
    return(pdDataFrame)


def read_census_csv(censusFile):
    cdf = pd.read_csv(censusFile ,index_col=None, header=0)
    demandPointList = []
    maxHeadCount = 0
    minHeadCount = 100000000000



    # Data on demand points
    totalDemandPoints = len(cdf[["oa11cd"]])
    print('There are ' + str(totalDemandPoints) + ' demand points on the file ' + censusFile)

    consideredDemands = np.ceil(totalDemandPoints)
    if consideredDemands < totalDemandPoints:
        print('*** WARNING: Considering only ' + str(consideredDemands) + ' demand points out of ' + str(totalDemandPoints) + ' ***')


    # STUFF FROM EXCEL:
    for lineno,row in cdf.iterrows():
        if lineno >= consideredDemands:
            print('WARNING: Early break on reading Demand Points data, read only ' + str(consideredDemands) + ' demand points!')
            break
        newDemandPoint = DEMAND_POINT()
        newDemandPoint.coords.lat = float(row['Y'])
        newDemandPoint.coords.long = float(row['X'])
        newDemandPoint.totalPopulation = float(str(row['All Ages']).replace(',', ''))
        if	newDemandPoint.totalPopulation > maxHeadCount:
            maxHeadCount = newDemandPoint.totalPopulation
        if newDemandPoint.totalPopulation < minHeadCount:
            minHeadCount = newDemandPoint.totalPopulation
        newDemandPoint.OACode = str(row['oa11cd'])

        # Calculated values:
        newDemandPoint.CRPDemand = -1
            
        demandPointList.append(newDemandPoint)

    print('The OA with more population has ' + str(maxHeadCount) + ', the one with less has ' + str(minHeadCount))
    return [demandPointList, maxHeadCount, minHeadCount]


def read_census(censusFile, demand_estimate=0.000230091):

    demandPointList = []
    ### READING INPUT DATA ###
    cdf = excel_sheet_to_df(censusFile, 'Census')
    maxHeadCount = cdf[["TOTAL"]].max().max()
    minHeadCount = cdf[["TOTAL"]].min().min()



    # Data on demand points
    totalDemandPoints = len(cdf[["OACode"]])
    print('There are ' + str(totalDemandPoints) + ' demand points on the file ' + censusFile)
    consideredDemands = np.ceil(totalDemandPoints)
    if consideredDemands < totalDemandPoints:
        print('*** WARNING: Considering only ' + str(consideredDemands) + ' demand points out of ' + str(totalDemandPoints) + ' ***')


    # STUFF FROM EXCEL:
    for lineno,row in cdf.iterrows():

        if lineno >= consideredDemands:
            print('WARNING: Early break on reading Demand Points data, read only ' + str(consideredDemands) + ' demand points!')
            break
        newDemandPoint = DEMAND_POINT()
        newDemandPoint.coords.lat = float(row['lat'])
        newDemandPoint.coords.long = float(row['long'])
        newDemandPoint.totalPopulation = float(row['TOTAL'])
        newDemandPoint.OACode = str(row['OACode'])
        popProfileHeaders = ['0_4', '10_14', '15_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85_89', '90plus']
        newDemandPoint.populationProfile = [] # From census
        for pp in popProfileHeaders:
            newDemandPoint.populationProfile.append(float(row[pp]))

        # Calculated values:
        newDemandPoint.CRPDemand = demand_estimate*newDemandPoint.totalPopulation
            
        demandPointList.append(newDemandPoint)
    
    print('The OA with more population has ' + str(maxHeadCount) + ', the one with less has ' + str(minHeadCount))
    return [demandPointList, maxHeadCount, minHeadCount]


def travel_estimations_osrm(censusList, GPLocations, GPNames, pharmacyLocations, pharmacyNames, server):
    import osrm
    osrm.RequestConfig.host = server
    verbose = 1
    # Prepare the data
    if verbose > 0:
        print('\tCensus points distances:')
    latLonListCensus = []
    for c in censusList:
        latLonListCensus.append(c.coords.latlong())

    print('Matching census to GPs:')
    goToGPs,dummy = list_matching(latLonListCensus, GPLocations, 'distance',maxMat=25000)
    print('Done.')
    print('Matching census to Pharmacies:')
    goToPharmacies,dummy = list_matching(latLonListCensus, pharmacyLocations, 'distance')
    print('Done.')
    for i,c in enumerate(censusList):
        c.walkingTimesToGPs = goToGPs[i]
        c.walkingTimesToPharmacies = goToPharmacies[i]
        min_index, min_value = min(enumerate(c.walkingTimesToGPs), key=operator.itemgetter(1))
        c.closestGP = min_index
        c.closestGPName = GPNames[min_index]

    print('Matching GPs to Pharmacies...')
    GpsToPharmacies,dummy = list_matching(GPLocations, pharmacyLocations, 'distance')
    gp_ph_dist = np.array(GpsToPharmacies)
    print('Done.')
    print('Travel time between GPs...')
    gp_to_gp  = od_matrix_latlong(GPLocations, 'distance')
    print('Done.')

    GP_MODEL = 0 # 0 or 1, see below

    # Calculate burden of each assignment
    for cidx,c in enumerate(censusList):
        asName = []
        asBurden = []
        # if cidx == 23:
        # 	print('Check this')
        if GP_MODEL == 0:
            # Located on a GP, but seen on usual GP
            for i, gpLoc in enumerate(GPLocations):
                tDist = c.walkingTimesToGPs[c.closestGP]

                if i != c.closestGP: # Otherwise distance is 0
                    tDist += gp_to_gp[c.closestGP, i]

                tDist += c.walkingTimesToGPs[i]

                asName.append(GPNames[i])
                asBurden.append(tDist)
                if tDist < 0:
                    print('ERROR: Getting negative distance...')
        else:
            # Diverted to a GP with PoC test available
            for i, gpLoc in enumerate(GPLocations):
                tDist = c.walkingTimesToGPs[i]
                lastBit = 100000000000000000000
                for j, pharLoc in enumerate(pharmacyLocations):
                    proposedLast = gp_ph_dist[i][j]
                    proposedLast += c.walkingTimesToPharmacies[j]
                    if proposedLast <= lastBit:
                        lastBit = proposedLast
                tDist += lastBit
                asName.append(GPNames[i])
                asBurden.append(tDist)
        c.asName = asName
        c.asBurden = asBurden
        if min(asBurden) < 0:
            print('ERROR: Getting negative distance...')

        # Located on a pharmacy
        for j, pharLoc in enumerate(pharmacyLocations):
            tDist = c.walkingTimesToGPs[c.closestGP]
            tDist += gp_ph_dist[c.closestGP][j]
            tDist += c.walkingTimesToPharmacies[j]
            asName.append(pharmacyNames[j])
            asBurden.append(tDist)
            if tDist < 0:
                print('ERROR: Getting negative distance...')
        for i,xx in enumerate(asBurden):
            if xx < 0:
                print('ERROR IN THIS DISTANCE.' + str(i))

def reverse_list_coords(wrongList):
    rightList = []
    for ii in wrongList:
        rightList.append([ii[1], ii[0]])
    return rightList

def od_matrix_latlong(pointList, metric='distance'):
    odmatrix, dummy, dummy = osrm.table(reverse_list_coords(pointList), output='np', annotations=metric)
    return odmatrix

def list_matching(list1, list2, metric, maxMat=50000):
    numberOfSources = len(list1)
    numberOfTargets = len(list2)
    if numberOfSources + numberOfTargets > maxMat:
        print('The list call needs to be split to fit the maximum allowed of ' + str(maxMat))
        print('(Calling with ' + str(numberOfSources) + ' sources and ' + str(numberOfTargets) + ' targets)')
        if numberOfTargets < maxMat - 1:
            nel1 = maxMat - numberOfTargets - 1
            print('Making calls with ' + str(nel1) + ' elements from list1 and ' + str(numberOfTargets) + ' from list2')
            nSplits = float(numberOfSources)/float(nel1)
            nSplits = int(np.floor(nSplits))
            print('Expecting ' + str(nSplits) + ' recursive calls.')
            matchListGo = []
            matchListReturn = []
            stored = 0
            nCalls = 0
            while stored < numberOfSources:
                nCalls += 1
                print('\tProcessing call number ' + str(nCalls) + '...')
                miniList1 = list1[stored:int(min(stored+nel1,numberOfSources))]
                print('\t(call with minlist: ' + str(len(miniList1)) + ' targets: ' + str(len(list2)) + ')')
                miniGo, miniReturn = list_matching(miniList1, list2, metric,maxMat=maxMat)
                matchListGo += copy.deepcopy(miniGo)
                matchListReturn += copy.deepcopy(miniReturn)
                stored += nel1
                print('\tDone.')
            print('Performed ' + str(nCalls) + ' calls.')
            print('> :split exit')
            return [matchListGo, matchListReturn]
        else:
            print('ERROR: Nedd to split second / both lists, not yet implemented!')
            exit(-1)

    superList = reverse_list_coords(list1 + list2)
    l2idx = len(list1)
    [theMatrix, dummy, dummy]  = osrm.table(superList, output='np', annotations=metric)
    matchListGo = []
    matchListReturn = []
    for i in range(len(list1)):
        matchListGo.append(theMatrix[i,l2idx:])
        matchListReturn.append(theMatrix[l2idx:,i])
    
    return [matchListGo, matchListReturn]