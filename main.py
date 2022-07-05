import pickle

if __name__ == "__main__":
    filename = 'final_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    # loaded_model.predict(X_test)
