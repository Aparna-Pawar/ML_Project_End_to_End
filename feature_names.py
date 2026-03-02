import pickle
preprocessor = pickle.load(open('artifacts/preprocessor.pkl', 'rb'))

# Dig into the steps. If step 1 is a ColumnTransformer:
column_transformer = preprocessor.steps[1][1] 

for name, transformer, columns in column_transformer.transformers_:
    print(f"Type: {name} | Columns: {columns}")