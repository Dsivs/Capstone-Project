"""
Application for Anomaly Detection Model

This module takes a dataset (a set of invoices) in json format and outputs which 
invoices are anomalous. The json may be the output from Itemize's API.

This module imports all the scripts needed. Make sure all scripts are in the same folder.

Date: 4/11/2025
"""


def main():
    # Step 1: Prompt for file path
    json_path = input("Paste the file path for your JSON file: ").strip()

    # Step 2: Run preprocessing
    print("Running preprocessing...")
    preprocessed_df = preprocess(json_path)

    # Step 3: Run random forest model
    print("Running Ensemble Model...")
    rf_output_df = ensemble_model(preprocessed_df)

    # Step 5: Print output
    print("Anomalous invoices detected:")
    print(anomalies_df)



if __name__ == "__main__":
    main()