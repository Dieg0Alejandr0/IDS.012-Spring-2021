import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import os
import json

"""This module provides functions to extract data from the JHU Daily US Cases"""

"""
A mapping of states to their capitals
"""
capitals={
    'Alabama': 'Montgomery',
    'Alaska': 'Juneau',
    'Arizona':'Phoenix',
    'Arkansas':'Little Rock',
    'California': 'Sacramento',
    'Colorado':'Denver',
    'Connecticut':'Hartford',
    'Delaware':'Dover',
    'Florida': 'Tallahassee',
    'Georgia': 'Atlanta',
    'Hawaii': 'Honolulu',
    'Idaho': 'Boise',
    'Illinois': 'Springfield',
    'Indiana': 'Indianapolis',
    'Iowa': 'Des Monies',
    'Kansas': 'Topeka',
    'Kentucky': 'Frankfort',
    'Louisiana': 'Baton Rouge',
    'Maine': 'Augusta',
    'Maryland': 'Annapolis',
    'Massachusetts': 'Boston',
    'Michigan': 'Lansing',
    'Minnesota': 'St. Paul',
    'Mississippi': 'Jackson',
    'Missouri': 'Jefferson City',
    'Montana': 'Helena',
    'Nebraska': 'Lincoln',
    'Nevada': 'Carson City',
    'New Hampshire': 'Concord',
    'New Jersey': 'Trenton',
    'New Mexico': 'Santa Fe',
    'New York': 'Albany',
    'North Carolina': 'Raleigh',
    'North Dakota': 'Bismarck',
    'Ohio': 'Columbus',
    'Oklahoma': 'Oklahoma City',
    'Oregon': 'Salem',
    'Pennsylvania': 'Harrisburg',
    'Puerto Rico': 'San Juan',
    'Rhode Island': 'Providence',
    'South Carolina': 'Columbia',
    'South Dakota': 'Pierre',
    'Tennessee': 'Nashville',
    'Texas': 'Austin',
    'Utah': 'Salt Lake City',
    'Vermont': 'Montpelier',
    'Virginia': 'Richmond',
    'Washington': 'Olympia',
    'West Virginia': 'Charleston',
    'Wisconsin': 'Madison',
    'Wyoming': 'Cheyenne'  
}

"""
The set of US states and territories
"""
states_and_territories = \
    set(capitals.keys()).union( 
    {
    #"American Samoa", 
    #"Guam", 
    #"Northern Mariana Islands", 
    #"Virgin Islands",
    "District of Columbia"
    } )

"""
A mapping of states to their populations, as per US 2020 Census. Taken from,
https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States_by_population
"""
populations = {
    "California" : 39538223,
    "Texas" : 29145505,
    "Florida" : 21538187,
    "New York" : 20201249,
    "Pennsylvania" : 13002700,
    "Illinois" : 12812508,
    "Ohio" : 11799448,
    "Georgia" : 10711908,
    "North Carolina" : 10439388,
    "Michigan" : 10077331,
    "New Jersey" : 9288994,
    "Virginia" : 8631393,
    "Washington" : 7705281,
    "Arizona" : 7151502,
    "Massachusetts" : 7029917,
    "Tennessee" : 6910840,
    "Indiana" : 6785528,
    "Maryland" : 6177224,
    "Missouri" : 6154913,
    "Wisconsin" : 5893718,
    "Colorado" : 5773714,
    "Minnesota" : 5706494,
    "South Carolina" : 5118425,
    "Alabama" : 5024279,
    "Louisiana" : 4657757,
    "Kentucky" : 4505836,
    "Oregon" : 4237256,
    "Oklahoma" : 3959353,
    "Connecticut" : 3605944,
    "Puerto Rico" : 3285874,
    "Utah" : 3271616,
    "Iowa" : 3190369,
    "Nevada" : 3104614,
    "Arkansas" : 3011524,
    "Mississippi" : 2961279,
    "Kansas" : 2937880,
    "New Mexico" : 2117522,
    "Nebraska" : 1961504,
    "Idaho" : 1839106,
    "West Virginia" : 1793716,
    "Hawaii" : 1455271,
    "New Hampshire" : 1377529,
    "Maine" : 1362359,
    "Rhode Island" : 1097379,
    "Montana" : 1084225,
    "Delaware" : 989948,
    "South Dakota" : 886667,
    "North Dakota" : 779094,
    "Alaska" : 733391,
    "District of Columbia" : 689545,
    "Vermont" : 643077,
    "Wyoming" : 576851
    #"Guam" : 168485,
    #"Virgin Islands" : 106235,
    #"Northern Mariana Islands" : 51433,
    #"American Samoa" : 49437,
}

"""
Months And Their Number Of Days
"""
month_days = {
    1:31,
    2:28,
    3:31,
    4:30,
    5:31,
    6:30,
    7:31,
    8:31,
    9:30,
    10:31,
    11:30,
    12:31
}

def to_string(L):
    """
    Given a list of the form [day, month, year] we return
    a string of the form 'day-month-year'.
    
    Input(s):
    *(list): List of our date's day, month, and year
    
    Output(s):
    *(str): A string of date in the proper format (with leading zeros)
    """
    
    first, second, third = str(L[0]).zfill(2), str(L[1]).zfill(2), str(L[2])
    return f'{first}-{second}-{third}'

def dates_list(startdate, enddate):
    """
    Given the first and last date we wish to see,
    we return a list of all the dates in between (inclusive).
    
    Input(s):
    *startdate(str): The first date we are interested in.
    
    *enddate(str): The last date we are interested in.
    
    Output(s):
    *output(list): A list of dates that have occurred between
    startdate and enddate (inclusive).
    """
    
    output = [startdate]
    if startdate == enddate: return output
    
    current = list(map(int, startdate.split('-')))
    end = list(map(int, enddate.split('-')))
    while current != end:
        
        if current[:2] == [12, 31]:
            current[:2] = [1, 1]
            current[2] += 1
            
        elif current[1] == month_days[current[0]]:
            current[0] += 1
            current[1] = 1
            
        else: current[1] += 1
            
        output.append( to_string(current) )
        
    return output

def readfile(directory, mapping, metric):
    """
    Given the directory of files, the mapping of 
    states/territories to a value we desire, and 
    the metric we are interested in, we update our 
    mapping by setting the value of a state/territory
    to the appropriate value.
    
    Input(s):
    *directory(str): The directory of the csv of interest.
    
    *mapping(dictionary, str to list): A mapping of US states/
    territories to a sequence of float avlues
    
    Output(s):
    *None
    """
    
    with open(directory) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader: 
            
            if row['Province_State'] in mapping:
                
                if row[metric] == '':
                    mapping[ row['Province_State'] ].append(0)
                else:
                    mapping[ row['Province_State'] ].append( float(row[metric]) )

def states_time_series(areas, metric, start, end):
    """
    Given the areas of interest in the US
    and a metric of interest, we create
    a mapping from the areas to their 
    time_series data from a metric. 
    
    Input(s):
    *areas(set): The set of US states/territories
    we are would like time-series data for.
    
    *metric(str): The COVID case metric we are 
    interested in.
    
    Output(s):
    *output(dict): A mapping from each area in
    areas to its time-series data.
    """
    
    output = {area:[] for area in areas}
    
    dates = dates_list(start, end)
    
    for date in dates:
        directory = f'./JHU DATA/daily_reports_us/{date}.csv'
        readfile(directory, output, metric)
        
    return output

def get_data(metric, start, end):
    """
    Given a metric of interest, we create
    a mapping from each US state and territory
    to their time-series data under a certain
    metric. 
    
    Input(s):
    *metric(str): The COVID case metric we are 
    interested in.
    
    Output(s):
    *X(np.array): A mapping from each area in
    areas to its time-series data.
    
    *indices(dict): A mapping of rows indices
    to their respective US state/territory.
    """
    
    mapping = {area:[] for area in states_and_territories }
    
    dates = dates_list(start, end)
    
    for date in dates:
        directory = f'./JHU DATA/daily_reports_us/{date}.csv'
        readfile(directory, mapping, metric)
    
    X, indices = [], {}
    index = 0
    for area in mapping: 
        X.append( mapping[area] )
        indices[index] = area
        index += 1
        
        
    return np.array(X), indices

def label_indices(start, end):
    """
    Given a start and end date of our 
    time series, we return the labels 
    we desire for our plots.
    
    Input(s):
    *start(str): The date we wish to
    start our time-series data.
    
    *end(str): The date we wish to
    end our time-series data.
    
    Output(s):
    *indices(list of ints): The indices
    that correspond to the start of a
    month.
    """
    
    dates = dates_list(start, end)
    indices = [0]
    current = 0
    #We track the current month we are in.
    previous = int(dates[0][:2])
    
    for date in dates[1:]:
        
        current += 1
        
        #If we are in a new month.
        if date[3:5] == '01':
            indices.append( current )

    return indices


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False



def get_census_data(metrics, zips = None, counties = None, states = None, verbose=False):
    """
    Given a list of zips/counties/states possible including DC, 
    we return a data frame containing data from metric
    Input(s):
    *metric(list of strs): The data in which we are interested, demographics-wise.
    These are the column names from come from the SECOND row of the census.csv file
    At most one of the following arguments should be provided:
    *zips(list of strs): The list containing the zip codes
    *counties(list of strs): The list containing the counties
    *states(list of strs): The list containing the states
   
    Output(s):
    *dataframe(states/counties/zips and indices): The dataframe containing the 
    data with columns being
    """
    df = pd.read_csv('census/census.csv', skiprows=(0,))
    df['id'] = df['id'].apply(lambda x: str(x[9:]))
    f = lambda x: 0 if x is None else 1
    if f(zips) + f(counties) + f(states) > 1:
        raise AttributeError("Expected at most one of zips/counties and states to not be None")
        
        
    # extract zip codes
    elif (zips is not None) or (f(zips) + f(counties) + f(states) == 0):
        if zips is not None:
            df = df[df['id'].isin(zips)]
        d = {'id' : np.array(df['id'])}
        for index in metrics:
           d[index] = df[index].values
        df = pd.DataFrame(data = d, columns = d.keys())
        
    # extract from counties
    elif counties is not None:
        d = {'id' : np.array(counties)}
        for index in metrics:
            d[index] = np.array([0.0] * len(counties))
        county_zips = {}
        for county in counties:
            county_zips[county] = []
        for name in states_and_dc:
            with open('us-data/' + name.replace(' ', '_') + '.geo.json') as file:
                s = json.load(file)
                zips = list(map(lambda x: (x['id'], x['properties']['county']), s['features']))
            for zp in zips:
                if zp[1] in county_zips.keys():
                    county_zips[zp[1]].append(zp[0])
        for county_idx in range(len(counties)):
            county = counties[county_idx]
            zips = county_zips[county]
            cnt = df[df['id'].isin(zips)]
            for zp in zips:
                for index in metrics:
                    if len(cnt[cnt['id'] == zp]) > 0:
                        d[index][county_idx] += 1.0 * cnt[cnt['id'] == zp][index].values[0]
        df = pd.DataFrame(data = d, columns = d.keys())
        
    # extract from the states
    elif states is not None:
        
        if verbose: print(f"loading states: {states}")
        
        d = {'id' : np.array(states)}
        
        for index in metrics:
            d[index] = np.array([0.0] * len(states))
            
        for state_idx in range(len(states)):
            state = states[state_idx]
            
            # check that the file exists and if not, print out the message, but don't throw an error
            fname = 'us-data/' + state.replace(' ', '_') + '.geo.json'
            if not os.path.isfile(fname):
                print(f"{fname} could not be found. It will be skipped. Make sure you have the data for it")
                continue
            
            # get the zip codes within this state
            with open(fname) as file:
                s = json.load(file)
                zips = list(map(lambda x: x['id'], s['features']))
                
            #
            cnt = df[df['id'].isin(zips)]
            for zp in zips:
                for index in metrics:
                    if verbose: print(f"metric: {index}")                  
                        
                    if len(cnt[cnt['id'] == zp]) > 0 and isfloat(cnt[cnt['id'] == zp][index].values[0]):
                        d[index][state_idx] += float(cnt[cnt['id'] == zp][index].values[0])
        df = pd.DataFrame(data = d, columns = d.keys())
    return df