def generate_features(df):
    features = ['seconds_in_bucket'
                , 'imbalance_buy_sell_flag'
                , 'imbalance_size'
                , 'matched_size'
                , 'bid_size'
                , 'ask_size'
                , 'reference_price'
                ,'far_price'
                , 'near_price'
                , 'ask_price'
                , 'bid_price'
                , 'wap'
                , 'imb_s1'
                , 'imb_s2']
# use the formula instead of the column function
    df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')
    prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']

    print("Generating 1st set of features...")
    for p1, p2 in tqdm(combinations(prices, 2)):
        df[f'{p1}_{p2}_imb'] = df.eval(f'({p1} - {p2})/({p1} + {p2})')
        features.append(f'{p1}_{p2}_imb')
    

    print("Generating 2st set of features...")
    for p1, p2, p3 in tqdm(combinations(prices, 3)):
        max_p = df[[p1, p2, p3]].max(axis = 1)
        min_p = df[[p1, p2, p3]].min(axis = 1)
        mid_p = df[[p1, p2, p3]].sum(axis = 1) - max_p - min_p
        df[f'{p1}_{p2}_{p3}_imb'] = (max_p - mid_p) / (mid_p - min_p) #normalized p from the 3
        features.append(f'{p1}_{p2}_{p3}_imb')

    return df    
    
    
df = generate_features(df_train)
df.columns