import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

_CORR_PLOTTED = False

#obradjivanje nan podataka u zavisnosti koja je kolona u pitanju
def _fill_nan(ds, column_name):
    n = len(ds[column_name])

    for i in range(n):
        if pd.isna(ds[column_name][i]):
            match column_name:
                case "instant":
                    ds[column_name][i] = i + 1
                case "dteday":
                    if ds[column_name][i-1] == ds[column_name][i+1]:    #ovakav pristup nije zasticen od greske kad je prva ili poslednja vrednost nan
                        ds[column_name][i] = ds[column_name][i+1]
                    else:
                        if ds["hr"][i] == 23:
                            ds[column_name][i] = ds[column_name][i-1]
                        else:
                            ds[column_name][i] = ds[column_name][i+1]
                case "season" | "holiday":
                    if ds[column_name][i - 1] == ds[column_name][i + 1]:
                        ds[column_name][i] = ds[column_name][i + 1]
                    else:
                        ds.drop(i, inplace=True)
                case "yr":
                    if pd.to_datetime(ds["dteday"][i]).year == 2011:    #python magic
                        ds[column_name][i] = 0
                    else:
                        ds[column_name][i] = 1
                case "mnth":
                    ds[column_name][i] = pd.to_datetime(ds["dteday"][i]).month
                case "hr":
                    ds[column_name][i] = (ds["hr"][i-1] + 1) % 24
                case "weekday":
                    if ds[column_name][i - 1] == ds[column_name][i + 1]:
                        ds[column_name][i] = ds[column_name][i + 1]
                    else:
                        if ds["hr"][i] == 23:
                            ds[column_name][i] = ds[column_name][i - 1] % 7
                        else:
                            ds[column_name][i] = ds[column_name][i + 1] % 7
                case "workingday":
                    match ds["weekday"][i]:
                        case 6 | 0:
                            ds[column_name][i] = 0
                        case _:
                            ds[column_name][i] = 1
                case "temp" | "atemp" | "hum" | "windspeed":
                    ds[column_name][i] = ds[column_name].values.mean()
                case "weathersit" | "casual" | "registered" | "cnt":
                    ds.drop(i, inplace=True)
                case _:
                    print("Invalid column name")

#uklanjanje anomalija koje nemaju fizickog smisla ili se ne slazu sa opsegom ostalih vrednosti
def _handle_anomalies_nat(ds, column_name):
    ds_clean = ds
    match column_name:
        case "season":
            ds_clean = ds_clean[(ds_clean[column_name] >= 1) & (ds_clean[column_name] <= 4)]
        case "yr" | "holiday" | "workingday":
            ds_clean = ds_clean[(ds_clean[column_name] >= 0) & (ds_clean[column_name] <= 1)]
        case "mnth":
            ds_clean = ds_clean[(ds_clean[column_name] >= 1) & (ds_clean[column_name] <= 12)]
        case "hr":
            ds_clean = ds_clean[(ds_clean[column_name] >= 0) & (ds_clean[column_name] <= 23)]
        case "weekday":
            ds_clean = ds_clean[(ds_clean[column_name] >= 0) & (ds_clean[column_name] <= 6)]
        case "weathersit":
            ds_clean = ds_clean[(ds_clean[column_name] >= 1) & (ds_clean[column_name] <= 4)]
        case _:
            print("Invalid column name")

    return ds_clean

#uklanjanje anomalija koristeci Z-score
def _handle_anomalies_z(ds, column_name):
    z_score = (ds[column_name] - ds[column_name].mean()) / ds[column_name].std(ddof=0)
    ds_clean = ds[z_score.abs() <= 3]

    return ds_clean

#pretprocesiranje celog dataseta prvo, pa tek onda kad se podeli na X i y
#prvo nad celim zato sto neke od stavki pretprocesiranja menjaju broj redova ds-a,
#pa da se ne bi razlikovala velicina izmedju X i y, radimo nad celim ds-om
def _eda_ds(ds):
    #uklanjanje duplikata redova
    ds.drop_duplicates(inplace=True)

    #otkrivanje i obrada nedostajucih redova
    #posto nedostaje manje od 1% ukupnog dataseta, mozemo zanemariti ove nedostatke posto nece mnogo uticati na kvalitet modela
    #mogu da se insertuju nedostajuci redovi, ulazne vrednosti se interpoliraju a izlazne se ostave da bi se koristile za predict, ovde nije potrebno

    #otkrivanje i obrada nan vrednosti
    for column_name in ds.columns:
        if ds[column_name].isna().any():
            _fill_nan(ds, column_name)

    #kodiranje (treba samo za dteday)
    #konvertuje svaki datum u int broj dana od neke reference (npr. 2011-01-01 → 734138)
    ds["dteday"] = pd.to_datetime(ds["dteday"]).map(lambda x: x.toordinal())
    #print(ds)

    #uklanjanje anomalija
    column_names_nat = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
    column_names_z = ["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
    ds_clean = ds
    for column_name in column_names_nat:
        ds_clean = _handle_anomalies_nat(ds, column_name)
    for column_name in column_names_z:
        ds_clean = _handle_anomalies_z(ds, column_name)

    return ds_clean

def _eda_X(X):
    # skaliranje podataka (podaci, pogotovo oni bitni, su vec skalirani)

    # detektovanje i obrada korelacija
    global _CORR_PLOTTED        #pomocna globalna promenljiva kako se ne bi ista heatmapa prikazivala dva puta

    if not _CORR_PLOTTED:
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = X.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)    #maska za samo donji trougao
        sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap="coolwarm", ax=ax)
        plt.title("Correlation Matrix")
        # plt.show()
        # plt.pause(5)
        _CORR_PLOTTED = True

    #iz matrice korelacija vidi se da su temp i atemp u visokoj korelaciji 0.99
    #isto se vidi i za season i mnth 0.83
    #tako da mozemo da drop-ujemo npr. atemp i season
    X.drop(columns=["season", "atemp"], inplace=True)

#eda za model koji predvidja samo cnt a zanemaruje casual i registered
def cnt_model():
    ds_cnt = pd.read_csv('hour.csv')
    #print(list(ds_cnt.columns))
    #print(ds_cnt)

    #obrada celog dataset-a
    ds_cnt = _eda_ds(ds_cnt)
    #print(ds_cnt)

    #izbacivanje parametara koji ocigledno nisu znacajni za formiranje izlaza
    ds_cnt.drop(columns=['instant', 'dteday', 'yr', 'casual', 'registered'], inplace=True)
    #print(ds_cnt)

    #podela na ulazne i izlazne podatke
    y = ds_cnt["cnt"]
    #print(y)
    ds_cnt.drop(columns=['cnt'], inplace=True)
    #print(ds_cnt)

    X = ds_cnt
    #print(X)

    #obrada nakon podele
    _eda_X(X)
    #print(X)

    return X, y

#model koji predvidja casual i registered
def cas_reg_model():
    ds_cas_reg = pd.read_csv('hour.csv')
    # print(list(ds_cas_reg.columns))
    # print(ds_cas_reg)

    # obrada celog dataset-a
    ds_cas_reg = _eda_ds(ds_cas_reg)
    # print(ds_cas_reg)

    # izbacivanje parametara koji ocigledno nisu znacajni za formiranje izlaza
    ds_cas_reg.drop(columns=['instant', 'dteday', 'yr', 'cnt'], inplace=True)
    # print(ds_cnt)

    # podela na ulazne i izlazne podatke
    y = ds_cas_reg[["casual", "registered"]]
    #print(y)
    ds_cas_reg.drop(columns=['casual', 'registered'], inplace=True)
    # print(ds_cas_reg)

    X = ds_cas_reg
    # print(X)

    # obrada nakon podele
    _eda_X(X)
    #print(X)

    return X, y