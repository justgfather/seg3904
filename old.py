import pandas as pd
import numpy as np
from timeit import default_timer as timer

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

from datetime import time
from sklearn.preprocessing import LabelEncoder

SRCADDR = '147.32.84.180'
DSTADDR = '60.190.223.75'
SRCADDR_TEST = '147.32.84.160'
DSTADDR_TEST = '217.163.21.37'
# WINDOWSIZE = 10000
WINDOWSIZE = 10

def convertColumnToInt32(dataFrame, columnName):
    """
        This function changes column values to int32

    """

    dataFrame[columnName] = dataFrame[columnName].astype(np.int32)
    return dataFrame


def deleteNullRow(dataFrame, columnName):
    """
        This function is used to delete a row
        which contains a null value(NaN)

    """

    dataFrame.dropna(subset=[columnName], inplace=True, axis=0)
    dataFrame.reset_index(drop=True, inplace=True)


def labelEncoder32(dataFrame, ipAddr, newColumnName, columnName):
    '''
        This function is used to convert column values (strings)
        into numbers (integer)

    '''

    labelEncoder = LabelEncoder()
    labelEncoder.fit(dataFrame[columnName])

    ipAddr_dis = list(labelEncoder.transform([ipAddr]))[0]  # integer representation of ipAddr

    dataFrame[newColumnName] = labelEncoder.transform(dataFrame[columnName])
    convertColumnToInt32(dataFrame, newColumnName)

    return ipAddr_dis


def generateFeatures(dataFrame, srcAddr, dstAddr, numberOfFlows, flag):
    '''
        This function is used to create
        features

    '''

    if (flag):  # Connection-based features

        tempDataFrame_1 = createDataFrameWhereAlt(dataFrame, srcAddr, 'Src IP', numberOfFlows)
        generateSrcAddrFeaturesConnectionBased(tempDataFrame_1, srcAddr, numberOfFlows)
        print('temp1')
        # print(tempDataFrame_1.dtypes)
        # print(tempDataFrame_1.shape)
        # print(tempDataFrame_1.heap(2))

        tempDataFrame_2 = createDataFrameWhereAlt(tempDataFrame_1, dstAddr, 'Dst IP', numberOfFlows)
        print('temp2')
        tempDataFrame_3 = tempDataFrame_2.copy(deep=True)
        print('temp3')
        generateDstAddrFeaturesConnectionBased(tempDataFrame_3, dstAddr, numberOfFlows)


    # print(tempDataFrame_3.dtypes)
    # print(tempDataFrame_3.shape)
    # print(tempDataFrame_3.head(2))

    else:

        tempDataFrame_1 = createDataFrameWhereAlt(dataFrame, srcAddr, 'Src IP', numberOfFlows)
        print('temptime')
        generateSrcAddrFeaturesTimeBased(tempDataFrame_1, srcAddr, numberOfFlows)

        tempDataFrame_1.to_csv('datatraining2_withTime_sourceOnly.csv')

        tempDataFrame_2 = createDataFrameWhereAlt(tempDataFrame_1, dstAddr, 'Dst IP', numberOfFlows)
        print('temp2_time')
        tempDataFrame_3 = tempDataFrame_2.copy(deep=True)
        print('temp3_time')
        generateDstAddrFeaturesTimeBased(tempDataFrame_3,dstAddr,numberOfFlows)

    return tempDataFrame_3


def createDataFrameWhereAlt(dataFrame, ipAddr, columnName, numberOfFlows):
    '''
        This function is an alternative to createDataFrameWhere
    '''

    if (ipAddr not in dataFrame[columnName].values):
        return None

    # start = time.time()

    numberOfRows = dataFrame.shape[0]

    result = None

    listOfDataFrames = []

    flag = True

    index_i = 0

    index_j = numberOfFlows

    while (flag):

        tempDataFrame = dataFrame[index_i:index_j]

        if (ipAddr in tempDataFrame[columnName].values):

            listOfDataFrames.append(tempDataFrame)
            # print("AHA")

            if (len(listOfDataFrames) == 2):
                result = pd.concat(listOfDataFrames)
                result = result[~result.index.duplicated(keep='first')]

                # result.drop_duplicates(inplace=True) #less performant than result[~result.index.duplicated(keep='first')]
                # result.reset_index(drop=True, inplace=True)   #No need because drop_duplicates removes incorrect indexes

                listOfDataFrames.clear()
                listOfDataFrames.append(result)

        index_i = index_i + 1
        index_j = index_j + 1

        if (index_j >= numberOfRows):
            # print("End condition reached")

            # print(len(listOfDataFrames))
            # print(listOfDataFrames[0].shape)

            flag = False

            # end = time.time()

            # print('Time in hours (createDataFrameWhere) : ', (end - start) * 0.000277778)

            return listOfDataFrames[0]


def createDataFrameWhere(dataFrame, ipAddr, columnName, numberOfFlows):
    '''
        This function is used to create a dataframe where the given
        ipAddr appears at least one time within the last <numberOfFlows>

    '''

    if (ipAddr not in dataFrame[columnName].values):
        return None

    # start = time.time()

    numberOfRows = dataFrame.shape[0]

    listOfDataFrames = []

    flag = True

    index_i = 0

    index_j = numberOfFlows

    while (flag):

        tempDataFrame = dataFrame[index_i:index_j]

        if (ipAddr in tempDataFrame[columnName].values):

            listOfDataFrames.append(tempDataFrame)

            if (len(listOfDataFrames) == 2):
                result = pd.concat(listOfDataFrames).drop_duplicates().reset_index(drop=True)

                listOfDataFrames.clear()

                listOfDataFrames.append(result)

        index_i = index_i + 1
        index_j = index_j + 1

        if (index_j == numberOfRows):
            # print("End condition reached")

            # print(len(listOfDataFrames))
            # print(listOfDataFrames[0].shape)

            flag = False

            # end = time.time()

            #  print('Time in hours (createDataFrameWhere) : ', (end - start) * 0.000277778)

            return listOfDataFrames[0]


def generateSrcAddrFeaturesConnectionBased(dataFrame, srcAddr, windowSize):
    '''

        this function is used to generate connection-based features using
        the given source ip address and window size


    '''

    srcAddr_dis = labelEncoder32(dataFrame, srcAddr, 'SrcAddr_Dis', 'Src IP')

    # print("DIS (SrcAddr) : ", srcAddr_dis)

    dataFrame['A_TotBytes_S'] = dataFrame['Tot Bytes'].rolling(windowSize).mean()  # Average TotBytes
    dataFrame['A_SrcBytes_S'] = dataFrame['TotLen Fwd Pkts'].rolling(windowSize).mean()  # Average SrcBytes
    dataFrame['A_TotPkts_S'] = dataFrame['Tot Pkts'].rolling(windowSize).mean()  # Average TotPkts

    dataFrame['Dct_Sport_S'] = dataFrame['Src Port'].rolling(windowSize).apply(lambda x: len(np.unique(x)),
                                                                               raw=False)  # Disctinct Source ports
    dataFrame['Dct_Dport_S'] = dataFrame['Dst Port'].rolling(windowSize).apply(lambda x: len(np.unique(x)),
                                                                               raw=False)  # Disctinct Destination ports

    dataFrame['Dct_SrcAddr_S'] = dataFrame['SrcAddr_Dis'].rolling(windowSize).apply(lambda x: len(np.unique(x)),
                                                                                    raw=False)  # Disctinct SrcAddr

    dataFrame['Nbr_App_S'] = dataFrame['SrcAddr_Dis'].rolling((windowSize // 10)).apply(
        lambda x: np.count_nonzero(np.where(x == srcAddr_dis)),
        raw=False)  # number of apperance of SrcAddr in (windowSize/10) netflows

    deleteNullRow(dataFrame, 'A_TotBytes_S')

    # print(dataFrame.shape[0])

    return dataFrame


def generateDstAddrFeaturesConnectionBased(dataFrame, dstAddr, windowSize):
    '''

        this function is used to generate connection-based features using
        the given destination ip address and window size


    '''

    dstAddr_dis = labelEncoder32(dataFrame, dstAddr, 'DstAddr_Dis', 'Dst IP')

    # print("DIS (DstAddr) : ", dstAddr_dis)

    dataFrame['A_TotBytes_D'] = dataFrame['Tot Bytes'].rolling(windowSize).mean()
    dataFrame['A_SrcBytes_D'] = dataFrame['TotLen Fwd Pkts'].rolling(windowSize).mean()
    dataFrame['A_TotPkts_D'] = dataFrame['Tot Pkts'].rolling(windowSize).mean()

    dataFrame['Dct_Sport_D'] = dataFrame['Src Port'].rolling(windowSize).apply(lambda x: len(np.unique(x)), raw=False)
    dataFrame['Dct_Dport_D'] = dataFrame['Dst Port'].rolling(windowSize).apply(lambda x: len(np.unique(x)), raw=False)

    dataFrame['Dct_DstAddr_D'] = dataFrame['DstAddr_Dis'].rolling(windowSize).apply(lambda x: len(np.unique(x)),
                                                                                    raw=False)

    dataFrame['Nbr_App_D'] = dataFrame['DstAddr_Dis'].rolling((windowSize // 10)).apply(
        lambda x: np.count_nonzero(np.where(x == dstAddr_dis)), raw=False)

    deleteNullRow(dataFrame, 'A_TotBytes_D')

    # print(dataFrame.shape[0])

    return dataFrame


def generateSrcAddrFeaturesTimeBased(dataFrame, srcAddr, time):
    '''

        this function is used to generate time-based features using
        the given source ip address and time


    '''

    time = time * 60  # convert to seconds
    time = str(time) + 's'

    dataFrame['timeStampIndex'] = pd.to_datetime(
        dataFrame['Timestamp'])  # used to create the rolling window based on minutes

    dataFrame.set_index('timeStampIndex', inplace=True)

    dataFrame = dataFrame.sort_index()

    srcAddr_dis = labelEncoder32(dataFrame, srcAddr, 'SrcAddr_Dis', 'Src IP')

    # print("DIS (SrcAddr) : ", srcAddr_dis)

    dataFrame['A_TotBytes_S_t'] = dataFrame['Tot Bytes'].rolling(time).mean()
    dataFrame['A_SrcBytes_S_t'] = dataFrame['TotLen Fwd Pkts'].rolling(time).mean()
    dataFrame['A_TotPkts_S_t'] = dataFrame['Tot Pkts'].rolling(time).mean()

    dataFrame['Dct_Sport_S_t'] = dataFrame['Src Port'].rolling(time).apply(lambda x: len(np.unique(x)), raw=False)
    dataFrame['Distinct_Dport (DstAddr)_t'] = dataFrame['Dst Port'].rolling(time).apply(lambda x: len(np.unique(x)),
                                                                                      raw=False)  # Not listed in the specification document

    dataFrame['Dct_SrcAddr_S_t'] = dataFrame['SrcAddr_Dis'].rolling(time).apply(lambda x: len(np.unique(x)), raw=False)

    dataFrame['Nbr_App_S_t'] = dataFrame['SrcAddr_Dis'].rolling(time).apply(
        lambda x: np.count_nonzero(np.where(x == srcAddr_dis)), raw=False)

    deleteNullRow(dataFrame, 'A_TotBytes_S_t')

    # print(dataFrame.shape[0])

    return dataFrame


def generateDstAddrFeaturesTimeBased(dataFrame, dstAddr, time):
    '''

        this function is used to generate time-based features using
        the given destination ip address and time


    '''

    time = time * 60  # convert to seconds
    time = str(time) + 's'

    dataFrame['timeStampIndex'] = pd.to_datetime(
        dataFrame['Timestamp'])  # used to create the rolling window based on minutes

    dataFrame.set_index('timeStampIndex', inplace=True)

    dataFrame = dataFrame.sort_index()

    dstAddr_dis = labelEncoder32(dataFrame, dstAddr, 'DstAddr_Dis', 'Dst IP')

    dataFrame['A_TotBytes_D_t'] = dataFrame['Tot Bytes'].rolling(time).mean()
    dataFrame['A_SrcBytes_D_t'] = dataFrame['TotLen Fwd Pkts'].rolling(time).mean()
    dataFrame['A_TotPkts_D_t'] = dataFrame['Tot Pkts'].rolling(time).mean()

    dataFrame['Dct_Sport_D_t'] = dataFrame['Src Port'].rolling(time).apply(lambda x: len(np.unique(x)), raw=False)
    dataFrame['Distinct_Dport (DstAddr)_t'] = dataFrame['Dst Port'].rolling(time).apply(lambda x: len(np.unique(x)),
                                                                                      raw=False)  # Not listed in the specification document

    dataFrame['Dct_DstAddr_t'] = dataFrame['DstAddr_Dis'].rolling(time).apply(lambda x: len(np.unique(x)), raw=False)

    dataFrame['Nbr_App_D_t'] = dataFrame['DstAddr_Dis'].rolling(time).apply(
        lambda x: np.count_nonzero(np.where(x == dstAddr_dis)), raw=False)

    deleteNullRow(dataFrame, 'A_TotBytes_D_t')

    # print(dataFrame.shape[0])

    return dataFrame


##########################################################################################################################
def set_labels(dataset):
    malicious_ips = ['192.168.2.112', '198.164.30.2', '192.168.2.113', '192.168.2.112', '147.32.84.180',
                     '147.32.84.140', '10.0.2.15',
                     '172.16.253.130', '172.16.253.240', '192.168.3.35', '172.29.0.116', '192.168.248.165',
                     '131.202.243.84',
                     '192.168.2.110', '192.168.1.103', '192.168.2.109', '147.32.84.170', '147.32.84.130',
                     '192.168.106.141',
                     '172.16.253.131', '74.78.117.238', '192.168.3.25', '172.29.0.109', '10.37.130.4', '192.168.5.122',
                     '192.168.4.118',
                     '192.168.4.120', '192.168.2.105', '147.32.84.150', '147.32.84.160', '192.168.106.131',
                     '172.16.253.129',
                     '158.65.110.24', '192.168.3.65', '172.16.253.132']

    # iterate through dataframe and set value
    for row in dataset.itertuples():
        if ((dataset.at[row.Index, 'Src IP'] in malicious_ips) or (
                dataset.at[row.Index, 'Dst IP'] in malicious_ips)):
            dataset.at[row.Index, 'Label'] = 1
        else:
            dataset.at[row.Index, 'Label'] = 0


def convert_column_to_datetime(dataFrame, columnName):
    dataFrame[columnName] = pd.to_datetime(dataFrame[columnName])
    return dataFrame


def delete_columns(dataFrame, column_list_to_keep):
    for column in dataFrame.columns:
        if column not in column_list_to_keep:
            dataFrame.drop(column, axis=1, inplace=True)


def convert_column_to_int(dataFrame, columnName):
    dataFrame[columnName] = dataFrame[columnName].astype(np.int32)
    return dataFrame


def delete_null_row(dataFrame, columnName):
    dataFrame.dropna(subset=[columnName], inplace=True, axis=0)
    dataFrame.reset_index(drop=True, inplace=True)


def delete_row_where(dataframe, column_name, value):
    dataframe.drop(dataframe[dataframe[column_name].str.contains(value) == True].index, inplace=True, axis=0)
    dataframe.reset_index(drop=True, inplace=True)


def total_packets(row_serie):
    return row_serie[0] + row_serie[1]


def total_bytes(row_serie):
    return row_serie[0] + row_serie[1]


def create_model(dataset_training, dataset_testing):
    list_of_features_to_drop = [
        'Flow ID',
        'Timestamp',
        'Flow Duration',
        'Protocol',
        'Src IP',
        'Src Port',
        'Dst IP',
        'Dst Port',
        'Label'
    ]

    y_training = dataset_training['Label']
    X_training = dataset_training.drop(list_of_features_to_drop, axis=1)

    print(len(X_training.columns))

    y_testing = dataset_testing['Label']
    X_testing = dataset_testing.drop(list_of_features_to_drop, axis=1)

    # Normalize data
    std = StandardScaler()
    x_data_training = std.fit_transform(X_training)

    # Create Model
    model = Sequential()
    # Input Layer Layer and first hidden layer
    # to_drop 45 et 77
    model.add(Dense(20, activation='sigmoid', input_shape=(94,)))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=["accuracy"])

    model.fit(x_data_training, y_training, epochs=200, batch_size=10,
              validation_data=(X_testing, y_testing), verbose=2)

    predictions = model.predict(X_training, batch_size=10, verbose=0)

    print(predictions)


def preprocess_data(dataFrame):
    # delete_columns(dataFrame, columns_list_to_keep)
    convert_column_to_int(dataFrame, 'Src Port')
    convert_column_to_int(dataFrame, 'Dst Port')
    convert_column_to_datetime(dataFrame, 'Timestamp')

    # Create total packets column
    dataFrame['Tot Pkts'] = dataFrame[['Tot Fwd Pkts', 'Tot Bwd Pkts']].apply(total_packets, axis=1)

    # Create total bytes column
    dataFrame['Tot Bytes'] = dataFrame[['TotLen Fwd Pkts', 'TotLen Bwd Pkts']].apply(total_bytes, axis=1)

    set_labels(dataFrame)
    return dataFrame


def main():
    # dataset_training = pd.read_csv('./ISCX_Botnet-Training.pcap_Flow_ubuntu.csv', encoding='utf-8', low_memory=False)
    # dataset_testing = pd.read_csv('./ISCX_Botnet-Testing.pcap_Flow.csv', encoding='utf-8', low_memory=False)

    dataset_training = pd.read_csv('datatraining.csv', encoding='utf-8', low_memory=False)
    dataset_testing = pd.read_csv('datatesting.csv', encoding='utf-8', low_memory=False)
    # column_list_to_keep = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts']

    # preprocess_data(dataset_training)
    preprocess_data(dataset_testing)
    # temp_training = generateFeatures(dataset_training, SRCADDR, DSTADDR, WINDOWSIZE, False)
    # temp_training.to_csv('datatraining2_withTime.csv')
    temp_testing = generateFeatures(dataset_testing, SRCADDR_TEST, DSTADDR_TEST, WINDOWSIZE, False)
    temp_testing.to_csv('datatesting2_withTime.csv')

    # dataset_training.drop(dataset_training.loc[dataset_training['Flow Byts/s'] == "Infinity"].index, inplace=True)
    # dataset_training.drop(dataset_training.loc[dataset_training['Flow Pkts/s'] == "Infinity"].index, inplace=True)

    # dataset_testing.drop(dataset_testing.loc[dataset_testing['Flow Byts/s'] == "Infinity"].index, inplace=True)
    # dataset_testing.drop(dataset_testing.loc[dataset_testing['Flow Pkts/s'] == "Infinity"].index, inplace=True)


    # create_model(dataset_training, dataset_testing)


if __name__ == "__main__":
    main()
