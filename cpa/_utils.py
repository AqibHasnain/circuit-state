from sklearn.preprocessing import MinMaxScaler

def MinMaxScale(arr,feature_range=(0,1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(arr)
    return scaler.transform(arr)

