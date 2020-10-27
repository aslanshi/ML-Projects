#!/usr/bin/env python
# coding: utf-8

# ## Overview:
# 
# Organize the notebook so that user could easily use by command lines.
# 
# created on: May 9, 2020
# 
# created by: Nanchun (Aslan) Shi

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter
from gurobipy import GRB, Model


# ### Time assignment

# In[2]:


def time_assignment(classFile, timeslotFile, preferenceFile):
    
    df = pd.read_csv(classFile)
    df = df.iloc[:,np.r_[3:6,9:11,18:24,30,35]]
    df = df[df.first_days == 'MW'].reset_index(drop=True).copy()
    df['time_block'] = list(zip(df.first_days, df.first_begin_time, df.first_end_time))
    
    time_blks = df.time_block.value_counts().index.sort_values()
    
    lookup = pd.read_excel(timeslotFile, index_col = 0).loc[['Mon','Wed'],:]
    lookup.columns = [str(time) for time in lookup.columns]
    
    time_blks_dict_window_3 = {}
    time_blks_dict_window_4 = {}
    for tb in time_blks:

        mon_start, wed_start = lookup.loc[:,tb[1]]
        duration = pd.to_datetime(tb[2],format='%H:%M:%S') - pd.to_datetime(tb[1],format='%H:%M:%S')
        window = round(duration.seconds/1800)

        if window == 3:
            time_blks_dict_window_3[tb] = [list(range(mon_start,mon_start+window)), 
                                           list(range(wed_start,wed_start+window))]
        else:
            time_blks_dict_window_4[tb] = [list(range(mon_start,mon_start+window)), 
                                           list(range(wed_start,wed_start+window))]
            
    pref = pd.read_csv(preferenceFile)
    
    window_dict = {3: time_blks_dict_window_3, 4: time_blks_dict_window_4}
    
    ## original time blocks should be good enough no need to create new ones
    ## after all we don't want something like 11:00:00 - 1:50:00
    ## for reproducability
    np.random.seed(1)

    tb_counter = Counter()

    for tb in time_blks:
        tb_counter[tb] += 0

    choice = []
    sections = set()

    ## iterate through all the courses in the dataframe
    for idx, cols in df.iterrows():

        instructor = cols['first_instructor_uid']
        ## get the preference scores for this instructor
        temp = pref[pref.id == instructor]
        course = cols['course']
        ## create a unique key by combining course and instructor id
        section = course + ' ' + str(instructor)
        ## calculate duration of this course
        duration = pd.to_datetime(cols['first_end_time'],format='%H:%M:%S') - pd.to_datetime(cols['first_begin_time'],format='%H:%M:%S')
        ## number of half-hour period used by this course
        window = round(duration.seconds/1800)
        ## get the corresponding time block dictionary
        dic = window_dict[window]
        ## assume an instructor could teach different sections but not different courses
        ## the following codes are checking if this course taught by this instructor appears before
        ## in other words, check if this is a course with multiple sections
        ## if it is, will assign the next time block for this course
        if section in sections:
            try:
                pre_idx = df[(df.course == course) & (df.first_instructor_uid == instructor)].loc[:idx-1,:].index[-1]
                pre_tb = choice[pre_idx]
                tb_idx = list(dic.keys()).index(pre_tb)
                cur_tb = list(dic.keys())[tb_idx+1]
                choice.append(cur_tb)
                tb_counter[cur_tb] += 1

            ## this exception happends when the last section of the same course is the end of the day
            ## so I will assign a random non-overlapping time blokc for it
            except:
                pre_idxs = list(df[(df.course == course) & (df.first_instructor_uid == instructor)]                                                                                            .loc[:idx-1,:].index)
                pre_tbs = set([choice[i] for i in pre_idxs])
                cand = list(set(dic.keys()) - pre_tbs)
                np.random.shuffle(cand)
                choice.append(cand[0])
                tb_counter[cand[0]] += 1
            continue
        sections.add(section)

        ## if this course does not have multiple sections or it is the first one appearing
        ## then I will assign time block for it based on highest preference scores
        score_dict = {}
        for tb in dic.keys():
            ## get Mon and Wed scores
            mon_score = temp[[str(num) for num in dic[tb][0]]].sum(axis=1)
            wed_score = temp[[str(num) for num in dic[tb][1]]].sum(axis=1)
            ## add them up
            total = mon_score + wed_score
            ## tracking scores for this course among all time blocks
            score_dict[tb] = total.values[0]
        ## select the time blocks with maximum score
        sorted_tbs = [key for key,value in score_dict.items() if value == max(score_dict.values())]

        ## randomly select when tie
        np.random.shuffle(sorted_tbs)

        ## The way to avoid out-of-room situation is to set 14 as maximum for each time block
        ## so that adjcent time blocks in total will not have more that 28, which is the total_number of rooms we have
        ## for each of the preferred time blocks, check if the time blocks are below limit
        ## if there is one that is not over the limit, assign it to the current course
        for i in range(len(sorted_tbs)):
            if tb_counter[sorted_tbs[i]] <= 13:
                choice.append(sorted_tbs[i])
                tb_counter[sorted_tbs[i]] += 1
                break

        ## this IF happends when all the preferred time blocks by this course are over the limit
        ## so I will assign them a random timeblock that the instructor could word on, so total score should >= window
        if len(choice) < idx + 1:
            l = list(dic.keys())
            np.random.shuffle(l)
            for tb in l:
                if score_dict[tb] > window:
                    choice.append(tb)
                    tb_counter[tb] += 1
                    break
        ## if there is no such block, I will assign them the block with highest scores, even it could have 0 somewhere          
        if len(choice) < idx + 1:  
            l = list(dic.keys())
            scores = pd.Series([score_dict[tb] for tb in l])
            cand_tb = l[scores.idxmax()]
            choice.append(cand_tb)
            tb_counter[cand_tb] += 1
            
    df['assigned_time_block'] = choice
    
    return df, time_blks


# ### Room Optimization

# In[3]:


def room_assignment(roomFile, df, time_blks):

    ## for reproducability
    np.random.seed(1)

    R = set(df.first_room.unique()) ##28 rooms
    room = pd.read_csv(roomFile,index_col=0).iloc[:,0]
    assignment_dict = {}
    courses = set()
    previous_rooms = set()

    ## remeber, time blocks are listed in the way that only adjecent pairs are conflicting with each other
    for tb in time_blks:

        section = set()
        pre_assigned_rooms = set()

        temp = df[df.assigned_time_block == tb].copy()
        temp['id'] = temp.course + temp.section

        ## if same course and same instructor, assign same room
        for c,s,ins in zip(temp.course,temp.section,temp.first_instructor_uid):
            c_ins = c+str(ins)
            if c_ins in courses:
                key = [key for key in assignment_dict.keys() if c in key][0]
                r = assignment_dict[key]
                section.add(c+s)
                pre_assigned_rooms.add(r)
                assignment_dict[c+s] = r
        ## keep tracking unique course + instructor keys
        for c,ins in zip(temp.course, temp.first_instructor_uid):
            courses.add(c+str(ins))

        ## the following grouobi optimization is maximizing room efficiency and minimize number of extra seats
        ## too few allowance could lead to unfeasible solution
        course = temp[['id','seats_offered']].set_index('id')

        mod = Model()

        I = set(course.index) - section
        ## if all courses in this time block got assigned above, no need to proceed
        if len(I) == 0:
            continue
        ## rooms in the previous time block could not be used since the time are overlapping
        ## also exclude those assigend above
        J = R - previous_rooms - pre_assigned_rooms

        x = mod.addVars(I, J, vtype = GRB.BINARY)
        y = mod.addVars(I, J, lb = 0, ub = 10, vtype = GRB.INTEGER)

        RE = sum(x[i,j]*course.loc[i,:]/room.loc[j] for i in I for j in J)
        extra = sum(y[i,j] for i in I for j in J)
        mod.setObjective(RE - extra, sense = GRB.MAXIMIZE)

        for i in I:
            mod.addConstr(sum(x[i,j] for j in J) == 1)
            for j in J:
                mod.addConstr(x[i,j]*course.loc[i,:] <= y[i,j] + room.loc[j])
        for j in J:
            mod.addConstr(sum(x[i,j] for i in I) <= 1)

        mod.setParam('outputflag',False)
        mod.optimize()

        ## rooms used in optimization
        used_rooms = set()
        for i in I:
            for j in J:
                if x[i,j].x:
                    used_rooms.add(j)
                    assignment_dict[i] = j

        ## normally, there would not be overlaps in a particular time block
        ## it could happend when two courses inherit from their previous sections, but in different time blocks
        ## so when that happens will randomly assign a room that could hold the class
        ## all rooms except rooms of previous time block and pre-assigned rooms
        all_rooms = J
        ## avaliable rooms
        ava_rooms = all_rooms - used_rooms
        ## subset of assignment for courses in the current time block
        subset_dict = {k:v for k,v in assignment_dict.items() if k in course.index}
        cum_count = []
        ## iterate through current assigments and check if there are rooms used by multiple courses
        for k,v in subset_dict.items():
            size = course.loc[k,:].values[0]
            if v in cum_count:
                room_sizes = {r:room.loc[r] for r in ava_rooms}
                cand_rooms = [r for r,s in room_sizes.items() if s in range(size-10,size+30)]
                ## if no feasible rooms, extend the range; rare cases
                if len(cand_rooms) == 0:
                    cand_rooms = [r for r,s in room_sizes.items() if s in range(size-60,size+60)]
                rand_room = np.random.choice(cand_rooms)
                assignment_dict[k] = rand_room
                ava_rooms.discard(rand_room)
            try:
                ## if re-assignment happened, append the new room so could be used for the next check
                cum_count.append(rand_room)
            except NameError:
                ## if no conflicts, append the original room and move to the next iteration
                cum_count.append(v)

        ## after all updates made
        ## get all rooms used for this time block, so that I could exclude them for the next time slot
        new_subset_dict = {k:v for k,v in assignment_dict.items() if k in course.index}
        previous_rooms = set([v for v in new_subset_dict.values()])
        
    df['id'] = df.course + df.section
    df['assigned_room'] = df.id.map(assignment_dict)
    
    return df


# In[4]:


def optimize(classFile, timeslotFile, preferenceFile, roomFile, outputFile):
    
    df, time_blks = time_assignment(classFile, timeslotFile, preferenceFile)
    df = room_assignment(roomFile, df, time_blks)
    
    df.to_excel(outputFile)


# In[5]:


if __name__=='__main__':
    import sys, os
    if len(sys.argv)!=6:
        print('Correct syntax: python AssignmentProgram.py classFile timeslotFile preferenceFile roomFile outputFile')
    else:
        classFile=sys.argv[1]
        timeslotFile=sys.argv[2]
        preferenceFile=sys.argv[3]
        roomFile=sys.argv[4]
        outputFile=sys.argv[5]
        if os.path.exists(classFile) and os.path.exists(timeslotFile) and os.path.exists(preferenceFile) and os.path.exists(roomFile):
            optimize(classFile, timeslotFile, preferenceFile, roomFile, outputFile)
            print(f'Successfully optimized. Results in "{outputFile}"')
        else:
            print(f'Some files not found!')

