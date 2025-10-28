import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

class CustomBins:
    nbins = 10
    bins = [(-np.inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
            (6, 7), (7, 8), (8, 14), (14, np.inf)]

    @staticmethod
    def get_bin_custom(x, nbins):
        x_days = x / 24.0

        for i in range(nbins):
            bin_range = CustomBins.bins[i]
            if bin_range[0] <= x_days < bin_range[1]:
                return i
        return nbins - 1  


def calculate_metrics(y_true_list, y_pred_list):
    y_true = np.array(y_true_list).flatten()
    y_pred = np.array(y_pred_list).flatten()

    y_pred = np.maximum(0, y_pred)

    metrics = {}

    metrics['mad'] = mean_absolute_error(y_true, y_pred)

    metrics['mse'] = mean_squared_error(y_true, y_pred)

    mask = y_true != 0
    if np.any(mask):
        metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics['mape'] = np.nan  

    
    y_true_bins = np.array([CustomBins.get_bin_custom(x, CustomBins.nbins) for x in y_true])
    y_pred_bins = np.array([CustomBins.get_bin_custom(x, CustomBins.nbins) for x in y_pred])

    metrics['kappa'] = cohen_kappa_score(y_true_bins, y_pred_bins, weights='linear')

    # metrics['y_true_bins'] = y_true_bins
    # metrics['y_pred_bins'] = y_pred_bins

    return metrics

