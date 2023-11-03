import pandas as pd
from tqdm import tqdm

# def generate_features(df):
#     features = ['seconds_in_bucket', 'imbalance_buy_sell_flag',
#                'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
#                'reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
#                'imb_s1', 'imb_s2'
#                ]
    
#     df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
#     df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')
    
#     prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    
#     print("Generating 1st set of features...")
#     for i,a in tqdm(enumerate(prices), total=len(prices), desc='1st loop'):
#         for j,b in enumerate(prices):
#             if i > j:
#                 df[f'{a}_{b}_imb'] = df.eval(f'({a}-{b})/({a}+{b})')
#                 features.append(f'{a}_{b}_imb')
#                 print(f"Processed {a} and {b}")

#     print("Generating 2nd set of features...")
#     for i,a in tqdm(enumerate(prices), total=len(prices), desc='2nd loop'):
#         for j,b in tqdm(enumerate(prices), total=len(prices), desc=f'2nd nested loop for {a}', leave=False):
#             for k,c in tqdm(enumerate(prices), total=len(prices), desc=f'3rd nested loop for {b}', leave=False):
#                 if i > j and j > k:
#                     max_ = df[[a, b, c]].max(axis=1)
#                     min_ = df[[a, b, c]].min(axis=1)
#                     mid_ = df[[a, b, c]].sum(axis=1) - min_ - max_

#                     df[f'{a}_{b}_{c}_imb2'] = (max_ - mid_) / (mid_ - min_)
#                     features.append(f'{a}_{b}_{c}_imb2')
#                     print(f"Processed {a}, {b}, and {c}")
    
#     return df[features]

# TRAINING = True
# if TRAINING:
#     df_train = pd.read_csv('optiver-trading-at-the-close/train.csv')
#     df_ = generate_features(df_train)
#     df_.to_pickle('output_dataframe.pkl')
#     df_.to_csv('output_dataframe.csv', index=False)

df = pd.read_pickle('output_dataframe.pkl')
display( df.head() )