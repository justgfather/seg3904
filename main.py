import keras
from sys import argv
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt



def rename_dataframe_column(dataframe_rename_columns):
    for column in dataframe_rename_columns.columns:
        new_column_name = column.replace(" ", "_").replace('/', '_').lower()
        # new_column_name = column.replace("/", "_")
        dataframe_rename_columns.rename(index=str, columns={column: new_column_name}, inplace=True)


def clean_dataset(dataset_to_clean):
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
    for row in dataset_to_clean.itertuples():
        if ((dataset_to_clean.at[row.Index, 'src_ip'] in malicious_ips) or (
                dataset_to_clean.at[row.Index, 'dst_ip'] in malicious_ips)):
            dataset_to_clean.at[row.Index, 'label'] = 1
        else:
            dataset_to_clean.at[row.Index, 'label'] = 0

    # Drop row with infinity string value
    dataset_to_clean.drop(dataset_to_clean.loc[dataset_to_clean['flow_byts_s'] == "Infinity"].index, inplace=True)
    dataset_to_clean.drop(dataset_to_clean.loc[dataset_to_clean['flow_pkts_s'] == "Infinity"].index, inplace=True)

    # Pour drop colonne completement
    #to_drop = ['flow_id', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'timestamp']

    to_drop_feature_selection = ['flow_id', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'timestamp']

    # inplace = true, fait en sorte que c est drop directement dans l objet
    dataset_to_clean.drop(to_drop_feature_selection, axis=1, inplace=True)

    # dataset.dropna(subset = ['Src IP','Src Port','Dst IP','Dst Port'])
    dataset_to_clean.dropna(how="any", axis=0, inplace=True)

    # cast flow pkts and
    dataset_to_clean[["flow_byts_s", "flow_pkts_s"]] = \
        dataset_to_clean[["flow_byts_s", "flow_pkts_s"]].apply(pd.to_numeric)

def clean_dataset2(dataset_to_clean):
    to_drop_feature_selection = ['StartTime', 'Proto', 'SrcAddr', 'Sport', 'DstAddr', 'Dport','unix_time']
    dataset_to_clean.drop(to_drop_feature_selection, axis=1, inplace=True)

def main():

    # dataset_training = pd.read_csv('./ISCX_Botnet-Training.pcap_Flow_ubuntu.csv', encoding='utf-8', low_memory=False)
    # dataset_testing = pd.read_csv('./ISCX_Botnet-Testing.pcap_Flow.csv', encoding='utf-8', low_memory=False)

    dataset_training = pd.read_csv('unb-Training.csv', encoding='utf-8', low_memory=False)
    dataset_testing = pd.read_csv('unb-testing.csv', encoding='utf-8', low_memory=False)

    clean_dataset2(dataset_testing)
    clean_dataset2(dataset_training)

    # Rename columns
    # rename_dataframe_column(dataset_training)
    # rename_dataframe_column(dataset_testing)

    # Clean dataset
    # clean_dataset(dataset_training)
    # clean_dataset(dataset_testing)

    # Feature selection
    # corr = dataset_training.corr()
    # corr.style.background_gradient(cmap='coolwarm')

    # Split data
    x_data_training = dataset_training.drop('Label', axis=1)
    labels_training = dataset_training['Label']
    x_data_testing = dataset_testing.drop('Label', axis=1)
    labels_testing = dataset_testing['Label']

    # Normalize data
    std = StandardScaler()
    x_data_training = std.fit_transform(x_data_training)
    x_data_testing = std.fit_transform(x_data_testing)

    # Create Model
    model = keras.Sequential()
    # Input Layer Layer and first hidden layer
    model.add(keras.layers.Dense(9, activation='relu', input_shape=(18,), kernel_regularizer= regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.4))
    # Output Layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=["accuracy"])

    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # callbacks=[es_callback]
    model.fit(x_data_training, labels_training, epochs= int(argv[1]), batch_size= int(argv[2]), shuffle= True,
                             validation_data=(x_data_testing, labels_testing), verbose=2)

    predictions = model.predict(x_data_testing, batch_size=int(argv[3]), verbose=0)

    print(predictions)

    # Save model
    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    del model


if __name__ == "__main__":
    main()

