import os
import json
import pandas as pd
import time
import glob
# raw_dir = "/mnt/ve_share2/zy/Argoverse_2_Motion_Forecasting_Dataset/raw/train/eb4b97cf-b611-4f1c-bdc9-4d4ba961c166/scenario_eb4b97cf-b611-4f1c-bdc9-4d4ba961c166.parquet"
# processed_dir='/mnt/ve_share2/zy/Argoverse_2_Motion_Forecasting_Dataset/processed/train'


# # 方法一：使用os.listdir和os.path.isfile
# start_time_1 = time.time()
# processed_file_names = [name for name in os.listdir(processed_dir) if
#                         os.path.isfile(os.path.join(processed_dir, name)) and
#                         name.endswith(('pkl', 'pickle'))]

# end_time_1 = time.time()
# print(processed_file_names[:10])
# # 方法二：使用os.scandir和os.path.isfile
# start_time_2 = time.time()
# processed_file_names_2 = [entry.name for entry in os.scandir(processed_dir) if
#                           entry.is_file() and entry.name.endswith(('pkl', 'pickle'))]
# end_time_2 = time.time()
# print(processed_file_names_2[:10])
# # 方法三：使用glob.glob
# start_time_3 = time.time()
# processed_file_names_3 = glob.glob(os.path.join(processed_dir, '*.pkl')) + glob.glob(os.path.join(processed_dir, '*.pickle'))
# end_time_3 = time.time()
# print(processed_file_names_3[:10])



# # 输出时间差
# print("方法一的执行时间：", end_time_1 - start_time_1)
# print("方法二的执行时间：", end_time_2 - start_time_2)
# print("方法三：",end_time_3-start_time_3)

# df = pd.read_parquet(raw_dir)
# historical_df = df[df['timestep'] < 50]
# print(historical_df.keys())
# print(historical_df[:110])

# with open(raw_dir, "rb") as f:
#     data= json.load(f)
# print(data)
# exit()
# if os.path.isdir(raw_dir):
#     print("start read raw train")
#     for name in os.listdir(raw_dir):
#         print(name)
#         exit()
    # raw_file_names = [name for name in os.listdir(raw_dir) if
    #                         os.path.isdir(os.path.join(raw_dir, name))]
raw='/mnt/ve_share2/zy/Argoverse_2_Motion_Forecasting_Dataset/raw/val/00010486-9a07-48ae-b493-cf4545855937/scenario_00010486-9a07-48ae-b493-cf4545855937.parquet'
sub='/mnt/ve_share2/zy/QCNet/submission/submission.parquet'
df = pd.read_parquet(sub)
print(df)
