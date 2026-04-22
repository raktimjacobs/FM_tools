import pandas as pd
from floodmodeller_api import DAT

#Using flood modeller api to extract sections data
dat = DAT("EX4.DAT")
sections = dat.sections

#extracting the data from first section
list_sections1 = list(sections)#list of sections in the DAT file
len_list = len(list_sections1)  #stores length of the sections list
sec1 = dat.sections[list_sections1[0]]  #cross-section data of one river section
sec1_data = sec1.active_data
sec1_data['Name'] = sec1.name
sec_data = sec1_data

#appending the rest of section data
for sec in range(0,len_list):
    sec1 = dat.sections[list_sections1[sec]]  #
    if sec1.unit == 'RIVER':
        df = sec1.active_data # saves the active data from section in a dataframe
        df['Name'] = sec1.name
        sec_data = pd.concat([sec_data,df])

#Save to csv
sec_data.to_csv('dat_2_csv.csv', index = False)