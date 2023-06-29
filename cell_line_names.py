import pandas as pd

df=pd.read_csv('common_celllines(1).csv')
#info_df=pd.read_csv('sample_info.csv')[['DepMap_ID','cell_line_name']]

ids=df.DepMap_ID.drop_duplicates()
ids.to_csv('common_cell_line_ids.csv')