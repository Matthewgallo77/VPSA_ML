def trainLSTM_Model(df):
    
    x=[]
    y=[]
    cycle_length = 10
    for i in range(0, len(df)-cycle_length):
        x.append(df.iloc[i:i+cycle_length].drop(targetPurity, axis=1).values) # CAPTURE 10 DATA POINTS ON EACH LOOP
        y.append(df.iloc[i+cycle_length][targetPurity]) # CAPTURE 10 DATA POINTS ON EACH LOOP (Purity only)

    print(len(x[0]))
    x = np.array(x)
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # SET RANDOM STATE FOR REPRODUCIBLE RESULTS
    '''
    units = 64neurons (power of 2 usually)
    input_shape = how many "points" to learn from. if t_equil = 3 hours and we are collecting data every minute (3*60/1)=180points
    5 is the number of input parameters
    Dense is what you are predicting (purity)
    '''
    model = tf.keras.Sequential([tf.keras.layers.LSTM(64, input_shape=(cycle_length, 5)), tf.keras.layers.Dense(1)])
    model.compile(optimizer='adam', loss='mse')

    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    return model