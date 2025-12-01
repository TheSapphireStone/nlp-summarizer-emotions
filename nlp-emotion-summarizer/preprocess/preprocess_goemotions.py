import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    # Assuming columns: "text" and "labels" (list of emotions)
    df['labels'] = df['labels'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['labels'])
    return df['text'].tolist(), y, mlb
