from matplotlib import pyplot as plt
from Projekat.EDA import cnt_model, cas_reg_model
from Projekat.LR_Ridge_Lasso import linear_regression, ridge, lasso
from Projekat.DT_Ensemble import decision_tree, random_forest, gradient_boosting
from sklearn.model_selection import train_test_split
import time

start = time.time()

def measure_time(func, args):
    start = time.time()
    func(*args)
    end = time.time()
    duration = end - start

    if duration < 60:
        print(f"Trajanje: {duration:.4f} sek")
    else:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        print(f"Trajanje: {minutes} min {seconds} sek")

def single_output():
    X, y = cnt_model()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=208)

    args = [X_train, X_test, y_train, y_test]

    print("\n---------------------------------------Single output (cnt)---------------------------------------\n")

    # modeli linearne regresije, ridge i lasso
    measure_time(linear_regression, args)   #0.03 sec
    measure_time(ridge, args)               #0.08 sec
    measure_time(lasso, args)               #0.42 dec

    figs = []
    args.append(figs)

    # modeli stabla odluke i ansambl metode
    measure_time(decision_tree, args)       #1 min 57 sec
    measure_time(random_forest, args)       #7 min 42 sec
    measure_time(gradient_boosting, args)   #7 min 6 sec

    for fig in figs:
        plt.figure(fig.number)

def multi_output():
    X, y = cas_reg_model()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=208)

    args = [X_train, X_test, y_train, y_test]

    print("\n---------------------------------------Multi-output (casual & registered)---------------------------------------\n")

    # modeli linearne regresije, ridge i lasso
    measure_time(linear_regression, args)   #0.07 sec
    measure_time(ridge, args)               #0.18 sec
    measure_time(lasso, args)               #0.87 sec

    figs = []
    args.append(figs)

    # modeli stabla odluke i ansambl metode
    measure_time(decision_tree, args)       #2 min 7 sec
    measure_time(random_forest, args)       #8 min 10 sec
    measure_time(gradient_boosting, args)   #14 min 13 sec

    for i in range(1, len(figs)):
        fig = figs[i]
        plt.figure(fig.number)

#za parametar prediction_target prosledjujemo:
#0 - ako zelimo da model predvidja cnt i zanemaruje casual i registered
#1 - ako zelimo da model predvidja casual i registered
#2 - uradi oba da imamo da poredimo na jednom mestu

def run(prediction_target=0):
    match prediction_target:
        case 0:
            single_output()
        case 1:
            multi_output()
        case 2:
            single_output()
            multi_output()
        case _:
            print("Invalid number for prediction_target!")

run(2)

end = time.time()

duration = end - start      #41 min 20 sec

if duration < 60:
    print(f"\nTrajanje celog programa: {duration:.4f} sek\n")
else:
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"\nTrajanje celog programa: {minutes} min {seconds} sek\n")

plt.show()