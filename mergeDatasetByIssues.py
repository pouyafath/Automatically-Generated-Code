import numpy as np
import pandas as pd
import os
import json
import time



df = pd.read_csv('/home/shabnam/pouya/mainDatasets/merged_dataset.csv')

new_dataset = {'path':[], 'label':[], 'bug_issue':[], 'hours_to_resolution':[], 'avg_hours_to_resolution':[], 'num_of_modified_files_issues':[], 'number_of_issue_report':[], 'bug_commit':[], 'number_of_commit_report':[], 'complexity':[], 'sum_added_lines':[], 'sum_deleted_lines':[], 'sum_num_lines':[], 'num_of_developers':[]}
# note: bug_commit when the commit has buggy status equal 1 

issue_path = 'bugIssuesReport'
issue_repos = os.listdir(issue_path)
commit_path = 'mainDatasets/apacheBugCommits'
commit_repos = os.listdir(commit_path) # ['accumulo', 'spark'] < 5

index = 0
for path,label in df.loc[:,['path','label']].values: # 8000 path
    repo = path.split('/')[1] 
    filename = '/' + path.split('/')[-1]
    parentFolder_filename = '/'+ path.split('/')[-2] + '/' + path.split('/')[-1]
    filename = parentFolder_filename


    new_dataset['path'].append(path)
    new_dataset['label'].append(label)
    new_dataset['bug_issue'].append(0)
    new_dataset['hours_to_resolution'].append(0)
    new_dataset['bug_commit'].append(0)
    new_dataset['complexity'].append(0)
    new_dataset['number_of_commit_report'].append(0)
    new_dataset['num_of_developers'].append(0)
    new_dataset['avg_hours_to_resolution'].append(0)
    new_dataset['num_of_modified_files_issues'].append(0)



    number_of_issue_report = 0
    number_of_commit_report = 0
    number_of_modified_files = 0
    sum_added_lines = 0
    sum_deleted_lines = 0
    sum_number_of_lines = 0
    temp_path = path.replace('ApacheRepos/'+repo+'/','')

    # info from issues
    if repo in issue_repos:
        with open(issue_path+f'/{repo}/bug_reports_with_commits.json','r') as f:
            issues = json.load(f)
            find = 0
            for key,value in issues.items():
                finded_by_filename = False
                for tpi in value['modified_files']:
                    if filename in tpi: # X.java in /src/ali/X.java
                        finded_by_filename = True
                        break
                if temp_path in value['modified_files'] or finded_by_filename:
                    number_of_issue_report += 1
                    number_of_modified_files += len(value['modified_files'])
                    new_dataset['hours_to_resolution'][index] += float(value['hours_to_resolution'])
                    if not find:
                        find = 1
                        new_dataset['bug_issue'][index] = 1
            # if not find:
            #     new_dataset['bug_issue'].append(0)
            #     new_dataset['hours_to_resolution'].append(0)
            new_dataset['number_of_issue_report'].append(number_of_issue_report)
            try:
                new_dataset['avg_hours_to_resolution'][index] = new_dataset['hours_to_resolution'][index] / number_of_modified_files
            except:
                new_dataset['avg_hours_to_resolution'][index] = 0
            new_dataset['num_of_modified_files_issues'][index] = number_of_modified_files

    # info from commit
    if repo in commit_repos:
        with open(commit_path+f'/{repo}/bugs_from_commits.json','r') as f:
            commit = json.load(f)
            developers_set = set()
            for key,value in commit.items():
                finded = 0                
                for mfDict in value['modified_files']: 
                    finded_by_filename = False 
                    for tp in [mfDict['new_path'], mfDict['old_path'], mfDict['file_name']]:
                        if tp:
                            if tp.endswith(filename):
                                finded_by_filename = True
                                break
                    if temp_path in [mfDict['new_path'], mfDict['old_path']] or finded_by_filename:
                        finded = 1
                
                if finded:
                    if value['is_buggy']:
                        new_dataset['bug_commit'][index] += 1
                    sum_added_lines += mfDict['added_lines']
                    sum_deleted_lines += mfDict['deleted_lines']
                    if mfDict['nloc']:
                        sum_number_of_lines += mfDict['nloc'] 
                    number_of_commit_report += 1
                    # print(number_of_commit_report)
                    if mfDict['complexity']:
                        new_dataset['complexity'][index] += mfDict['complexity']
                    if value['committer']:
                        developers_set.add(value['committer'])
                # if not find:
                #     new_dataset['complexity'].append(-1)
            new_dataset['number_of_commit_report'][index] = number_of_commit_report
            new_dataset['num_of_developers'][index] = len(developers_set)
    
    new_dataset['sum_added_lines'].append(sum_added_lines)
    new_dataset['sum_deleted_lines'].append(sum_deleted_lines)
    new_dataset['sum_num_lines'].append(sum_number_of_lines)
    index += 1
    print(index)


df = pd.DataFrame(new_dataset)
print(df)
df.to_csv('info_from_issue_and_commit_based_on_parent_folder_search_2.csv')

# print(np.sum(df.loc[:,'bug_issue'].value))
        

                

    
