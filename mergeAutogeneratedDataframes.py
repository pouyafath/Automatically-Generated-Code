import numpy as np
import pandas as pd
import json
import os

sourceDfPath = "mainDatasets/humanDataframes"

reposDFs = os.listdir(sourceDfPath)



with open('seenAutoRepos.json', 'r') as f:
    seenAutoRepos = json.load(f)

c = 0

flag = 0
# del df,df1
dfs = []
listOfSeenRepos = set()
for repoDF in reposDFs:
    print(repoDF.split('_')[0])
    listOfSeenRepos.add(repoDF.split('_')[0])
    if repoDF.split('_')[0] in seenAutoRepos:
        sourceRepoDF = sourceDfPath + '/' + repoDF
        if flag == 0:
            df = pd.read_csv(sourceRepoDF)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Drop rows with NaN
            df.dropna(inplace=True)
            flag = 1
            continue
        if flag == 1:
            df1 = pd.read_csv(sourceRepoDF)
            df1.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Drop rows with NaN
            df1.dropna(inplace=True)
            flag = 2
            continue
        if flag == 2:
            df = pd.concat([df,df1], ignore_index=True)
            df1 = pd.read_csv(sourceRepoDF)
            df1.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Drop rows with NaN
            df1.dropna(inplace=True)

        listOfSeenRepos.add(repoDF.split('_')[0])
        
        
df = pd.concat([df,df1], ignore_index=True)
df = df.iloc[:,1:]
del df1 
print(len(df))

df.to_csv('mainDatasets/merged_human_labeled_dataset_new.csv')

# with open('seenAutoRepos_match_with_auto.json', 'w') as f:
#     json.dump(list(listOfSeenRepos), f)

# print(len(listOfSeenRepos))

# df = pd.read_csv('merged_auto_labeled_dataset2.csv')
# print(len(df))

# listOfRealRepos = os.listdir('ApacheRepos')

# with open('seenAutoRepos.json', 'r') as f:
#     seenAutoRepos = json.load(f)

# print(seenAutoRepos)
# print()


# i = 0
# for repo in listOfRealRepos:
#     i += 1
#     if repo in seenAutoRepos:
#         print(i)
    
# df1 = pd.read_csv('/home/shabnam/pouya/mainDatasets/merged_auto_labeled_dataset.csv')
# df2 = pd.read_csv('/home/shabnam/pouya/mainDatasets/merged_human_labeled_dataset.csv')
# df2 = df2.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

# df = pd.concat([df1,df2.iloc[:4155]], ignore_index=True)
# df = df.iloc[:,1:]


# df.to_csv('/home/shabnam/pouya/mainDatasets/merged_dataset.csv')
# print(len(df))