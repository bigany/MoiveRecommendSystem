from rs import *
import random

if __name__ == '__main__':

    path = './ml-latest-small/'
    timer = Timer()

    # Initializing data and then Construct the RS System.
    dataset = initial_data(path)
    timer.print("Data Initialized Successfully.")

    RS = RecommendSystem(*dataset)
    timer.print("Recommend System Configured Successfully.")

    # Print the recommendation for random user.
    user = random.randint(1, RS.userNo)
    RS.print_reclist(user)

    # Appraise the system by calculate mean r.
    r_hat = RS.calculate_Rank()
    timer.print(" Mean r value of the System is {:f}.\n".format(r_hat))

    # Calculate TPR and FPR for drawing the ROC curve.
    [TPR, FPR] = RS.calculate_ROC()
    timer.print("Calculate TPR and FPR Done.")

    # Draw ROC curve.
    auc_curve(TPR, FPR)
