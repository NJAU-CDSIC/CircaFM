import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import json
import numpy
import numpy as np

class Metric:

    @staticmethod
    def calculate_map(labels, scores):
        """
        Calculate the mean Average Precision (mAP) for multi-class classification.
        
        Args:
            labels: Ground truth labels.
            scores: Predicted scores/probabilities.
        """
        map_score = metrics.average_precision_score(labels, scores, average='macro')
        print(f"mAP: {map_score:.4f}")

    @staticmethod
    def get_classification_report(labels, predictions):
        """
        Generate a classification report (precision, recall, F1-score) for each class.
        
        Args:
            labels: Ground truth labels.
            predictions: Predicted class labels.
        
        Returns:
            str: Classification report.
        """
        report = metrics.classification_report(labels, predictions, digits=4)
        return report

    @staticmethod
    def save_metrics(result, addr, filename, epoch, c_time, mode='a'):
        """
        Save metrics to a JSON file. Supports append or write mode.
        
        Args:
            result (dict): Metrics dictionary to save.
            addr (str): Base directory path.
            filename (str): Name of the JSON file.
            epoch (int): Current epoch number.
            c_time (str): Subdirectory name (e.g., timestamp).
            mode (str): 'a' for append (adds to existing list), 'w' for overwrite.
        """
        result['epoch'] = epoch
        saveAddr = os.path.join(addr, c_time, filename)
        
        # Ensure the directory exists
        directory = os.path.dirname(saveAddr)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if mode.lower() == "a":
            if not os.path.exists(saveAddr) or os.path.getsize(saveAddr) == 0:
                # File does not exist or is empty -> create as JSON array
                with open(saveAddr, 'w') as f:
                    json.dump([result], f, indent=4)
            else:
                # Remove 'targets' key if present
                del result['targets']
                with open(saveAddr, 'r+') as file:
                    file.seek(0, 2)  # Move to end of file
                    position = file.tell()
                    while position >= 0:
                        file.seek(position)
                        if file.read(1) == '\n':
                            break
                        position -= 1
                    file.seek(position)
                    file.write(',\n' + json.dumps([result], indent=4)[2:])
        elif mode.lower() == 'w':
            with open(saveAddr, 'w') as f:
                json.dump(result, f, indent=4)
        
        print(f"Epoch {epoch} results saved to {saveAddr}")