import google.generativeai as genai
import pandas as pd
import glob
import os

from Delay import completed_df,current_progress_df
from ReportDetails import imp_details_df,overall_df

genai.configure(api_key='AIzaSyA82nnh81Dh9IYtbr62MkvbVEk3o1j53mc')
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model=genai.GenerativeModel(model_name='gemini-pro',generation_config=generation_config)
def extract_multiple_dfs(dfs: list) -> list:
    parts = [f"--- START OF DATAFRAMES ---"]
    for i, df in enumerate(dfs):
        parts.append(f"--- START OF DATAFRAME {i+1} ---")
        for index, row in df.iterrows():
            parts.append(" | ".join(map(str, row.values)))
    return parts


path="C:\\Users\\t226722\\DashBoard Presentation\\Data"
xlsx_files = glob.glob(os.path.join(path, "*.xlsx"))
df_list = [pd.read_excel(file) for file in xlsx_files]
df =pd.concat(df_list, ignore_index=True)


#OVERALL SUMMARY
df1=[imp_details_df,overall_df,completed_df]
convo = model.start_chat(history=[
    {
        "role": "user",
        "parts": extract_multiple_dfs(df1)
    },
])
convo.send_message("Write a overall summary report of overall performance in the project ? The output should be less than 30 words")
report1=convo.last.text

#AREAS NEED TO BE IMPROVED
df2=[completed_df]
convo = model.start_chat(history=[
    {
        "role": "user",
        "parts": extract_multiple_dfs(df2)
    },
])
convo.send_message("What can be done to improve the project's overall performance ? Suggest solutions based on the delay and comments. The output should be less than 30 words.")
report2=convo.last.text

#CURRENT PROGRESS
df3=[current_progress_df]
convo = model.start_chat(history=[
    {
        "role": "user",
        "parts": extract_multiple_dfs(df3)
    },
])
convo.send_message("How is the overall progress of the project positive or negative ? Write a small explanation to justify the progress. The output should be less than 30 words.")
report3=convo.last.text