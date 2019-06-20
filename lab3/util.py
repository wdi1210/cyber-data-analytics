import pandas as pd
import random


def modifyData(df):
    df['Date'] = df['Date_1'] + ' ' + df['Date_2']
    df.set_index('Date', inplace=True)
    df.drop(['Date_1', 'Date_2', 'direction'], axis = 1, inplace=True)
    

def reservoirSample(input_df, result_df, rezSize):
    if len(result_df) < rezSize:
        diff = rezSize - len(result_df)
        diff = min(diff, len(input_df))
        results_df = pd.concat([input_df.head(diff), result_df])
    else:
        for i in range(rezSize, len(input_df)):
            j = random.randint(1, rezSize)
            if j < rezSize:
                result_df.iloc[j] = input_df.iloc[i]


