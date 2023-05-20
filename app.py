from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

def prediction(lst):
    filename = 'model/predictor.pickle'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
        pred_value = model.predict(lst)
        return pred_value    

@app.route('/', methods=['POST', 'GET'])
def index():
    pred=0
    if request.method == 'POST':
        mobiles = request.form['mobiles']
        cluster = request.form['cluster']
        print(mobiles)
        print(cluster)
        
        cluster_list = ['Peradeniya', 'Ampara', 'Kandy', 'Vavuniya', 'Trincomalee', 'Rathnapura',
                        'Colombo', 'Kurunegala', 'Anurdhapura', 'Karapitiya', 'Kaluthara', 'Jaffna', 'Hambanthota',
                        'CNTH', 'CIM', 'Chilaw', 'Batticaloa', 'Badulla', 'Kamburugamuwa', 'Gampaha',
                        'Kegalle', 'NuwaraEliya', 'Monaragala', 'Matara', 'Polonnaruwa']

        feature_list = [int(mobiles)]
        feature_list.extend([1 if c == cluster else 0 for c in cluster_list])

        # Check the number of features expected by the model
        expected_features = 29
        if len(feature_list) < expected_features:
            feature_list.extend([0] * (expected_features - len(feature_list)))

        print(feature_list)

        feature_array = np.array(feature_list).reshape(1, -1)
        pred = prediction(feature_array)

        print(pred)
  
    return render_template("index.html",pred = pred)

if __name__ == '__main__':
    app.run(debug=True)
