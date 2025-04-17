"""
Application for Anomaly Detection Model

This file takes a dataset (a set of invoices) in json format and outputs which 
invoices are anomalous. The json may be the output from Itemize's API.

This file imports all the scripts needed. Make sure all scripts are in the same folder.

Date: 4/11/2025
"""


def main():
    # Step 1: Prompt for file path
    json_path = input("Paste the file path for your JSON file: ").strip()

    # Step 2: Run preprocessing
    print("Running preprocessing...")
    preprocessed_df = preprocess(json_path)

    # Step 3: Run random forest model
    print("Running Random Forest...")
    rf_output_df = run_random_forest(preprocessed_df)

    # Step 4: Run XGBoost model
    print("Running XGBoost...")
    anomalies_df = run_xgboost(rf_output_df)

    # Step 5: Print output
    print("Anomalous invoices detected:")
    print(anomalies_df)




if __name__ == "__main__":
    main()