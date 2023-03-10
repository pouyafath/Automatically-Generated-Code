import pydriller
from pydriller import Repository
import json
import pandas
import os
from tqdm import tqdm as tq
import statistics
import csv
from pickle import FALSE
from datetime import datetime



bug_related_msg = ['bug','bugs','fix','fixes','patch','fault','corrected','tweaked','problem','problems','issue']
autoGeneratedKeywords = ['this class was autogenerated', 'autogenerated', 'THIS FILE IS AUTOMATICALLY GENERATED', 'This code was generated by a tool', 'This source code was auto-generated by', 'automatically generated', 'This source code was auto-generated by wsdl', 'his class was automatically generated by a Snowball to Java compiler', 'This source code was auto-generated by MonoXSD','Generated.*by.*JFlex','generated by JFlex','The following code was generated by JFlex', 'generated by CUP', 'The following code was generated by CUP', 'Generated by Doxygen', 'generated by JavaCC', 'Generated by JavaCC: ignore naming convention violation', 'DO NOT EDIT THIS FILE - it is machine generated', 'Generated by Maven', 'Autogenerated by Thrift', 'Generated By:javaCC: Do not edit this line', 'This file was generated by SableCC', 'GeneratedOrderBy ANTLR']


# exception_messages = []
def bugs_from_commits_in_repo(repo_name):
    print('---------------------')
    print(repo_name)
    commits_info = {}
    bug_count=0
    
    
    number_of_exceptions = 0
    total_commits = 0
    
    files_changes_count=[]
    real_files_changes_count=[]
    
    added_any_type_lines_per_repo=[]
    deleted_any_type_lines_per_repo=[]
    

    for commit in Repository(repo_name, since=datetime(2017, 1, 1, 0, 0, 1)).traverse_commits():  #local
        total_commits += 1
        if total_commits % 100 == 0:
            print(total_commits)
         
        if commit.in_main_branch == False:
            continue
        is_bug = False
        is_java = False
        for bug_msg in bug_related_msg:
            if bug_msg in commit.msg:
                is_bug = True
                break
        single_commmit = {}
        changed_files = []
        try:                        
            # print(commit.modified_files)
            for file in commit.modified_files: 
                if ".java" in file.filename:
                    print(file.filename)
                    print(repo,'---->', str(commit.committer_date))

                    path = os.path.join(repo_name,file.new_path)
                    if os.path.exists(path):
                        print('Ignore file bacause it is not exist')
                        continue
                    # label= 0
                    # try:
                    #     path = os.path.join(repo_name,file.new_path)
                    #     with open(path, 'r') as openfile:
                    #         content = openfile.read()
                    #         label=0
                    #         for keyword in autoGeneratedKeywords:
                    #             if keyword in content:
                    #                 label= 1
                    #                 print(path)
                    #                 break
                    #         del content
                    # except:
                    #     pass
                    # print(file.new_path)
                    file_obj = {}
                    is_java = True
                    file_obj['file_name'] = file.filename 
                    file_obj['old_path'] = file.old_path
                    file_obj['new_path'] = file.new_path
                    file_obj['nloc'] = file.nloc
                    file_obj['added_lines'] = file.added_lines
                    file_obj['deleted_lines'] = file.deleted_lines
                    # file_obj['is_autogenerated'] = label
                    file_obj['complexity'] = file.complexity
                    changed_files.append(file_obj)
                    
                    # print(repo,'---->', str(commit.committer_date))
                    single_commmit['committer_date'] = str(commit.committer_date) 
                    single_commmit['author_date'] = str(commit.author_date)
                    single_commmit['commit_message'] = commit.msg
                    single_commmit['in_main_branch'] = commit.in_main_branch
                    single_commmit['committer'] = commit.committer.email
                    single_commmit["modified_files"]=changed_files
                    single_commmit['is_buggy'] = is_bug
                    commits_info[commit.hash] = single_commmit

                
        except Exception as e:
            number_of_exceptions += 1
            
    print('total commits for '+ repo_name,':',total_commits)
    return commits_info


def save_to_file(root, repo_name, file_name, json_obj):
    folder_path = os.path.join(root,repo_name)
    file_path = os.path.join(folder_path,file_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(file_path, 'w') as f:
        json.dump(json_obj, f, indent=4)


def load_from_file(root, repo_name, file_name):
    folder_path = os.path.join(root,repo_name)
    file_path = os.path.join(folder_path,file_name)
    with open(file_path, 'r') as f:
        return json.load(f)

commits_infos = []

basePath = '/home/shabnam/pouya/'
apacheDir = 'mainDatasets/autogeneratedDataframes'
listOfSeenRepos = set()
for repoDF in os.listdir(basePath+apacheDir):
    listOfSeenRepos.add(repoDF.split('_')[0])

# repos = os.listdir(basePath+apacheDir)
# selected_repos = ['tapestry4', 'apr', 'etch', 'stdcxx', 'xalan-j', 'zookeeper', 'sling-old-svn-mirror', 'apr-iconv', 'tapestry3']
# selected_repos = ['accumulo']
for repo in list(listOfSeenRepos):
    repoPath = os.path.join(basePath,'ApacheRepos',repo)
    print(repoPath)
    commit_info = bugs_from_commits_in_repo(repoPath)
    # # commits_infos.append(commit_info)
    save_to_file(basePath+"/mainDatasets/apacheBugCommits/",repo,"bugs_from_commits.json",commit_info)

    # if commit_info:
    #     save_to_file(basePath+"/mainDatasets/apacheBugCommits/",repo,"bugs_from_commits.json",commit_info)
    # else:
    #     print('the reposictorty does not have any commits so has not be saved')