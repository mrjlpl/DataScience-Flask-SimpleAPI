from flask import Flask
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("property_v0.model")

def fe(df):
    df["area_num"] = df["area"].map(lambda x: x.split("m")[0].replace(" ", "").replace(",", ".")).astype("float")
    df["area_num_log"] = np.log(df["area_num"])
    area_num_99 = np.percentile(df["area_num"], 99)
    df["area_norm"] = df["area_num"].map(lambda x: x if x <= area_num_99 else area_num_99)
    df["area_per_room"] = df["area_norm"] / df["rooms"]

    for i in range(5):
        df["loc{}".format(i)] = df["location"].map(lambda x: x[i] if len(x) > i else "")

    df['loc01'] = df['loc0'] + df['loc1']
    df['loc012'] = df['loc0'] + df['loc1'] + df['loc2']
    df['loc12'] = df['loc1'] + df['loc2']

    citi_areas = ['PoznańWinogrady', 'WarszawaŚródmieście', 'KrakówKrowodrza', 'WarszawaBiałołęka',
                  'KrakówStare Miasto', 'kołobrzeskiKołobrzeg', 'WarszawaMokotów', 'ŁódźŚródmieście', 'PoznańGrunwald',
                  'WrocławKrzyki']
    for item in citi_areas:
        df[item] = df["loc12"] == item

    big_cities = {'Poznań', 'Sopot', 'Wrocław', 'Kraków', 'Gdańsk', 'Gdynia', 'Opole', 'Katowice', 'Częstochowa',
                  'Szczecin', 'Kalisz', 'Łódź', 'Olsztyn', 'Warszawa'}
    for city in big_cities:
        df[city] = df["loc1"] == city
        df["big_city"] = df["loc1"].map(lambda x: x in big_cities)

    return df


@app.route('/predict/<string:area>/<int:rooms>/<string:location>')
def get_forecast(area, rooms, location):
    df = pd.DataFrame([{
        "area": "12",
        "rooms": 1,
        "location": [x.strip() for x in location.split(",")]
    }])

    df_fe = fe(df)
    feats = ['area_num', 'area_num_log', 'area_per_room', 'rooms', 'big_city', 'PoznańWinogrady', 'WarszawaŚródmieście',
             'KrakówKrowodrza', 'WarszawaBiałołęka', 'KrakówStare Miasto', 'kołobrzeskiKołobrzeg', 'WarszawaMokotów',
             'ŁódźŚródmieście', 'PoznańGrunwald', 'WrocławKrzyki', 'Częstochowa', 'Szczecin', 'Kraków', 'Gdańsk',
             'Poznań', 'Gdynia', 'Opole', 'Olsztyn', 'Sopot', 'Wrocław', 'Łódź', 'Warszawa', 'Kalisz', 'Katowice']
    X = df_fe[feats].values
    y_pred = model.predict(X)

    return ("area={}, rooms={}, location={}, y_pred={}".format(area, rooms, location, y_pred))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8051):q!
