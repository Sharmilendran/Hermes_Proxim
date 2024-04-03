import json

import flask
from flask import Flask, jsonify, request
from flask_cors import CORS

import final
import predict
import response

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['GET'])
def GetPrediction():
    classifications = predict.classify()
    print(classifications)
    proximity = final.soundSorterMatcher()
    print(proximity)
    # proximity = json.dumps(proximity.__dict__)
    # res = response.PredictResponse(classifications, proximity)
    res = {"classifications": classifications, "proximity": proximity}
    print(res)
    return res


if __name__ == '__main__':
    app.run(debug=True)