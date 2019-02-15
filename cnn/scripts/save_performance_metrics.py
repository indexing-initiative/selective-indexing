from .helper import load_delimited_data, save_delimited_data
import sys


def run(dereferenced_predictions_filepath, encoding, delimiter, group_indices, metrics_filepath, threshold):
    predictions = load_delimited_data(dereferenced_predictions_filepath, encoding, delimiter)
    group_counts = _collect_counts(predictions, group_indices, delimiter, threshold)
    group_metrics = _compute_metrics(group_counts)
    _save_metrics(metrics_filepath, encoding, delimiter, group_indices, group_metrics)


def _collect_counts(predictions, group_indices, delimiter, threshold):
    group_counts = {}
    for prediction in predictions:
        if len(group_indices) == 1 and group_indices[0] == -1:
            group = 'all'
        else:
            group = delimiter.join([prediction[group_index] for group_index in group_indices])
   
        if group not in group_counts:
            group_counts[group] = { 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        counts = group_counts[group]
            
        act  = prediction[5].strip().lower() == 'true'
        score = float(prediction[6].strip())
        pred = score >= threshold
        
        if pred and act:
            counts['tp'] += 1
        elif pred and not act:
            counts['fp'] += 1
        elif not pred and act:
            counts['fn'] += 1
        elif not pred and not act:
            counts['tn'] += 1

    return group_counts


def _compute_metrics(group_counts):
    eps = sys.float_info.epsilon
    group_metrics = {}
    for group, counts in group_counts.items():
        metrics = dict(counts)
        group_metrics[group] = metrics
        tp, fp, fn, tn = counts['tp'], counts['fp'], counts['fn'], counts['tn']
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        fscore = 2*precision*recall / (precision + recall + eps)
        count = tp + fp + fn + tn
        indexed_count = tp + fn 
        not_indexed_count = tn + fp
        indexed_fraction = indexed_count / count
        not_indexed_fraction = not_indexed_count / count
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['fscore'] = fscore
        metrics['count'] = count
        metrics['indexed_count'] = indexed_count
        metrics['not_indexed_count'] = not_indexed_count
        metrics['indexed_fraction'] = indexed_fraction
        metrics['not_indexed_fraction'] = not_indexed_fraction
    return group_metrics


def _save_metrics(filepath, encoding, delimiter, group_indices, group_metrics):
    sorted_items = sorted(group_metrics.items(), key=lambda x: x[0])
    to_save = []
    headings = ['Group Index {}'.format(index) for index in group_indices]
    headings.extend(['Count', 'Indexed Count', 'Non Indexed Count', 'Indexed Fraction', 'Non Indexed Fraction', 
                    'tp', 'fp', 'fn', 'tn', 
                    'Precision', 'Recall', 'F1 Score',])
    to_save.append(headings)
    for group, metrics in sorted_items:
        to_save.append((group, 
                        metrics['count'], metrics['indexed_count'], metrics['not_indexed_count'], metrics['indexed_fraction'], metrics['not_indexed_fraction'],
                        metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn'],
                        metrics['precision'], metrics['recall'], metrics['fscore'],
                        ))
    save_delimited_data(filepath, encoding, delimiter, to_save)

        
if __name__ == '__main__':
    DEREFERENCED_PREDICTIONS_FILEPATH = '/****/dereferenced_predictions.csv'
    METRICS_FILEPATH = '/****/metrics-by-journal-group-pub-year.csv'
    ENCODING = 'utf8'
    DELIMITER = ','
    GROUP_INDEX = [4, 1]
    run(DEREFERENCED_PREDICTIONS_FILEPATH, ENCODING, DELIMITER, GROUP_INDEX, METRICS_FILEPATH, 0.5)
