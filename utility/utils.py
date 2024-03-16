import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from utility.data_gen import simplify_ground_truth


def correct_metrices(predictions):
    """predictions: path to the predictions file or the dataframe itself
    predictions should have the following columns: ID, Class"""
    predictions_df = None
    if isinstance(predictions, str):
        predictions_df = pd.read_csv(predictions)
    else:
        predictions_df = predictions

    true_df = pd.read_csv(r"data\ISIC2018_Task3_Test_GroundTruth.csv")

    # merging the two dataframes
    final_df = pd.merge(
        predictions_df,
        true_df,
        left_on="ID",
        right_on="image_id",
        how="left",
    )
    final_df = final_df.drop("image_id", axis=1).rename(
        columns={"dx": "True_Disease", "id": "True_Class"}
    )

    # checking nan values
    # print(final_df.isna().sum())

    # Extract the predicted and true class labels
    predicted_labels = final_df["Class"]
    true_labels = final_df["True_Class"]

    # Calculate the classification report
    report = classification_report(true_labels, predicted_labels, output_dict=True)

    # Print the accuracy and recall score for each class
    for class_label, metrics in report.items():
        if class_label.isdigit():
            accuracy = round(metrics["precision"], 3)
            recall = round(metrics["recall"], 3)
            disease_name = true_df[true_df["id"] == int(class_label)]["dx"].values[0]
            string = f"Class {class_label} '['{disease_name}']'"
            string = string.ljust(19)
            print(f"{string}: Accuracy = {accuracy}, Recall = {recall}")

    # Calculate the macro average accuracy and recall score
    macro_avg_accuracy = round(report["macro avg"]["precision"], 3)
    macro_avg_recall = round(report["macro avg"]["recall"], 3)
    print(
        f"Macro Average: Accuracy = {macro_avg_accuracy}, Recall = {macro_avg_recall}"
    )

    return report


def get_weigths(DataClass):
    labels = np.array(
        [
            l["class_id"]
            for l in simplify_ground_truth(DataClass.gt_file, DataClass.mapping_file)
        ]
    )
    # Find unique elements and their frequencies
    unique_elements, counts = np.unique(labels, return_counts=True)

    # Calculate inverse frequencies
    inverse_frequencies = sum(counts) / counts
    # print(unique_elements, inverse_frequencies)
    return inverse_frequencies
