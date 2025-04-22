"""
Application for Anomaly Detection Model

This module takes a dataset (a set of invoices) in json format and outputs which
invoices are anomalous. The json may be the output from Itemize's API.

This module imports all the scripts needed. Make sure all scripts are in the same folder.

Date: 4/11/2025
"""

import os

import ensemble_parallel


def main() -> None:
    print("Current working directory:")
    print(f"{os.getcwd()}")

    default_model_path = "ensemble_model.pkl"
    model = None

    if os.path.exists(default_model_path):
        while True:
            choice = (
                input(
                    f"\nFound default model at '{default_model_path}'. "
                    "Load this model? (y/n): "
                )
                .strip()
                .lower()
            )
            if choice == "y":
                model = ensemble_parallel.load_model(default_model_path)
                break
            elif choice == "n":
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    while model is None:
        custom_path = input(
            "\nEnter path to model file (or type 'q' to quit): "
        ).strip()
        if custom_path.lower() == "q":
            print("Exit.")
            return
        elif os.path.exists(custom_path):
            model = ensemble_parallel.load_model(custom_path)
        else:
            print("File not found. Please try again.")

    while True:
        json_path = input(
            "\nEnter the path to the JSON file you want to predict on "
            "(or 'q' to quit): "
        ).strip()
        if json_path.lower() == "q":
            print("Exit.")
            return
        if os.path.exists(json_path):
            break
        else:
            print("File not found. Please try again.")
    print("Running prediction...")

    result = model.predict_from_json(json_path)
    anomalous_invoice = result[result["is_anomalous"] == 1]
    print("Anomalous Invoices Detected:")
    print(anomalous_invoice.to_string(index=False))
    print(f"Anomalous invoice count: {len(anomalous_invoice)}.")

    while True:
        save = input("\nSave the results as 'prediction.csv'? (y/n): ").strip().lower()
        if save == "y":
            result.to_csv("prediction.csv", index=False)
            print("Results saved to 'prediction.csv'.")
            break
        elif save == "n":
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


if __name__ == "__main__":
    ensemble_parallel.train_and_save_model("synthetic_invoices.json")
    main()
