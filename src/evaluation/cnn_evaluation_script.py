import numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
import sys, time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, GlobalAveragePooling1D, Conv1DTranspose, LSTM, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

pd.set_option('display.precision', 4)

from tensorflow.keras import backend as K

th_vae_datasets = ['b4l5', 'b4l10', 'b24l50', 'b8l20', 'b16l5', 'b20l3', 'b20l5', 'b24l15']# , 'b8l15'] #9
th_gan_datasets = ['b6e100', 'b8e2000', 'b12e5000', 'b16e2000', 'b20e100', 'b32e500']#, 'b32e100', 'b8e1000'] # 8

el_vae_datasets = ['b4l3', 'b20l3', 'b20l20', 'b24l3', 'b32l3', 'b32l5'] #7 'b8l10'
el_gan_datasets = ['b4e100', 'b20e2000', 'b10e100', 'b16e500', 'b24e1000', 'b32e1000'] # 7 'b20e500'

def r_squared(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - ss_res/(ss_tot + K.epsilon()))

base_data_train, base_data_test = np.load('../../data/training_data/training_data_1month.npy', allow_pickle=True)

tf.random.set_seed(42)

from sklearn.preprocessing import MinMaxScaler

base_data_train, base_data_test = np.load('../../data/training_data/training_data_1month.npy', allow_pickle=True)
scalers = {var_name: MinMaxScaler(feature_range=(-1,1)) for var_name in ['G.air.T', 'G.E_th_I', 'G.sky.T', 'G.E_el_I']}

air_var, sky_var, el_var, th_var = base_data_train[:,:,0], base_data_train[:,:,1], base_data_train[:,:,2], base_data_train[:,:,3]
air_var_test, sky_var_test, el_var_test, th_var_test = base_data_test[:,:,0], base_data_test[:,:,1], base_data_test[:,:,2], base_data_test[:,:,3]

scaled_data_train = np.stack((scalers['G.air.T'].fit_transform(air_var),
                             scalers['G.sky.T'].fit_transform(sky_var),
                             scalers['G.E_el_I'].fit_transform(el_var),
                             scalers['G.E_th_I'].fit_transform(th_var)), axis=-1)
scaled_data_test = np.stack((scalers['G.air.T'].fit_transform(air_var),
                             scalers['G.sky.T'].fit_transform(sky_var),
                             scalers['G.E_el_I'].fit_transform(el_var),
                             scalers['G.E_th_I'].fit_transform(th_var)), axis=-1)
print(scaled_data_train.shape, scaled_data_test.shape)

def split_base_data(base_data_train, base_data_test, th_indices=[0, 3], el_indices=[1, 2]):
    th_base_data_train, th_base_data_test = base_data_train[:, :, th_indices], base_data_test[:, :, th_indices]
    el_base_data_train, el_base_data_test = base_data_train[:, :, el_indices], base_data_test[:, :, el_indices]
    return th_base_data_train, th_base_data_test, el_base_data_train, el_base_data_test

th_base_data_train, th_base_data_test, el_base_data_train, el_base_data_test = split_base_data(base_data_train, base_data_test)
print(th_base_data_train.shape, th_base_data_test.shape, el_base_data_train.shape, el_base_data_test.shape)

def create_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=24, activation='relu', input_shape=(input_shape)),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        Conv1D(filters=64, kernel_size=24, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(730, activation='linear')
    ])
    return model

def test_cnn(training_data, th_or_el):
    X_train = training_data[:,:,0].reshape(-1,730,1)
    y_train = training_data[:,:,1]

    if th_or_el == 0:
        X_test = scaled_data_test[:,:,0].reshape(-1, 730, 1)  
        y_test = scaled_data_test[:,:,3]
    else:
        X_test = scaled_data_test[:,:,1].reshape(-1, 730, 1)  
        y_test = scaled_data_test[:,:,2]

    X_train, X_train_val, y_train, y_train_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
                                                                  
    model = create_cnn((X_train.shape[1], X_train.shape[2]))
    model.compile(optimizer=Adam(), loss='mse', metrics=['mse', 'mae'])

    early_stopping = EarlyStopping(monitor='mse', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='mse', factor=0.5, patience=5, verbose=1)
    model.fit(X_train, y_train, epochs=100, batch_size=16, callbacks=[early_stopping, reduce_lr], verbose=0, validation_data=(X_train_val, y_train_val))

    loss, mse, mae = model.evaluate(X_test, y_test)
    r2 = r_squared(tf.convert_to_tensor(y_test, dtype=tf.float32), tf.convert_to_tensor(model.predict(X_test), dtype=tf.float32))
    
    return {'mse':mse, 'mae':mae, 'r2':r2.numpy()}


num_reps = 100

def test_cnn_wrapper(data, th_or_el=0):
    mse, mae, r2 = [], [], []
    
    # Run each CNN training 10 times to ensure results are significant and not outliers
    for i in range(num_reps):
        print(f'RUN: {i}')
        results = test_cnn(np.random.permutation(data), th_or_el) # permuting the data for each run just to ensure full shuffling
        mse.append(results['mse'])
        mae.append(results['mae'])
        r2.append(results['r2'])

    return {'mse':mse, 'mae':mae, 'r2':r2}
        



batches=[2,4,6,8,10,12,16,20,24,32]
epochs=[100,500,1000,2000,5000]

def load_data(base_path, context, dataset_names):
    datasets = {}
    for name in dataset_names:
        file_path = f'{base_path}/{context}_{name}_generated_samples.npy'
        datasets[name] = np.load(file_path, allow_pickle=True)
    return datasets

base_path_vae = '../../data/vae_synthetic_data/'
base_path_gan = '../../data/gan_synthetic_data/'
th_context = 'th_v_air'
el_context = 'el_v_sky'

th_vae_data = load_data(base_path_vae, th_context, th_vae_datasets)
th_gan_data = load_data(base_path_gan, th_context, th_gan_datasets)

el_vae_data = load_data(base_path_vae, el_context, el_vae_datasets)
el_gan_data = load_data(base_path_gan, el_context, el_gan_datasets)

def evaluate_and_save_cnn(data_dicts, model_type, column_index, base_path='../../data/evaluation_results/'):
    for dataset_name, dataset in data_dicts.items():
        print(dataset_name)
        cnn_test_results = test_cnn_wrapper(dataset[0:216, :, :], column_index)

        result_df = pd.DataFrame(cnn_test_results)
        result_file_name = f'{base_path}{model_type}_{"th" if column_index==0 else "el"}_{dataset_name}.csv'
        result_df.to_csv(result_file_name)
        print(f'CNN results for {dataset_name}: {result_df}')
        print(f'Saved CNN evaluation results for {dataset_name} to: {result_file_name}')

def evaluate_and_save_cnn_blended(data_dicts, model_type, column_index, base_path='../../data/evaluation_results/'):
    for dataset_name, dataset in data_dicts.items():
        scaled_data = scaled_data_train[:,:,[0,3]] if column_index==0 else scaled_data_train[:,:,[1,2]]
        print(dataset_name)
        cnn_test_results = test_cnn_wrapper(np.concatenate((scaled_data, dataset[0:216,:,:]), axis=0), column_index)

        result_df = pd.DataFrame(cnn_test_results)
        result_file_name = f'{base_path}{model_type}_blended_{"th" if column_index==0 else "el"}_{dataset_name}.csv'
        result_df.to_csv(result_file_name)
        print(f'CNN results for {dataset_name}: {result_df}')
        print(f'Saved CNN evaluation results for {dataset_name} to: {result_file_name}')

if __name__=="__main__":
    
    #evaluate_and_save_cnn(th_gan_data, 'gan', 0)
    evaluate_and_save_cnn(el_gan_data, 'gan', 1)
    evaluate_and_save_cnn_blended(el_gan_data, 'gan', 1)

    '''x = input("select test (0,1,2,3,4,5): ")
    match x:
        case 0:
            #evaluate_and_save_cnn(th_vae_data, 'vae', 0)
            evaluate_and_save_cnn(th_gan_data, 'gan', 0)
        case 1:
            #evaluate_and_save_cnn_blended(th_vae_data, 'vae', 0)
            evaluate_and_save_cnn_blended(th_gan_data, 'gan', 0)
        case 2:
            #evaluate_and_save_cnn(el_vae_data, 'vae', 1)
            evaluate_and_save_cnn(el_gan_data, 'gan', 1)
        case 3:
            #evaluate_and_save_cnn_blended(el_vae_data, 'vae', 1)
            evaluate_and_save_cnn_blended(el_gan_data, 'gan', 1)
        case 4:
            evaluate_and_save_cnn(el_vae_data, 'vae', 1)
            #evaluate_and_save_cnn(el_gan_data, 'gan', 1)
        case 5:
            evaluate_and_save_cnn_blended(el_vae_data, 'vae', 1)
            evaluate_and_save_cnn_blended(el_gan_data, 'gan', 1)
        case _:
            print("CASE 0")
            evaluate_and_save_cnn(th_vae_data, 'vae', 0)
            evaluate_and_save_cnn_blended(th_vae_data, 'vae', 0)'''
