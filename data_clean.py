import pandas as pd

identity_data = pd.read_csv('/Users/chesionyaya/PycharmProjects/classification_task/src/data/jigsaw-unintended-bias-in-toxicity-classification/identity_individual_annotations.csv')
all_data = pd.read_csv('/Users/chesionyaya/PycharmProjects/classification_task/src/data/jigsaw-unintended-bias-in-toxicity-classification/all_data.csv')
cleaned_identity_data = identity_data[identity_data['id'].isin(all_data['id'])]
cleaned_identity_data_reused = cleaned_identity_data.copy()


print(cleaned_identity_data.columns)

###take the comments with male
cleaned_identity_data['gender'] = (cleaned_identity_data['gender'] == 'male').astype(int)
male_identity_mean= cleaned_identity_data.groupby('id')['gender'].mean()
male_ids = male_identity_mean[male_identity_mean>0.5].index
male_all_data = all_data[all_data['id'].isin(male_ids)].dropna()
male_series_name = ['id','comment_text','toxicity']
male_comments = male_all_data[male_series_name]
print(len(male_comments))
print(len(male_ids))

###take the comments with women
cleaned_identity_data_reused['gender'] = (cleaned_identity_data_reused['gender'] == 'female').astype(int)
female_identity_mean = cleaned_identity_data_reused.groupby('id')['gender'].mean()
female_ids = female_identity_mean[female_identity_mean>0.5].index
female_all_data = all_data[all_data['id'].isin(female_ids)].dropna()
female_series_name = ['id','comment_text','toxicity']
female_comments = female_all_data[female_series_name]
print(len(female_comments))