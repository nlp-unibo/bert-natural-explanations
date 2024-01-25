from cinnamon_core.utility.json_utility import load_json
from cinnamon_core.utility.pickle_utility import save_pickle


def from_doc_id_to_name(doc_ids, doc_names):
    return [doc_names[idx] for idx in doc_ids]


if __name__ == '__main__':
    folds_json = load_json('folds.json')
    with open('folds.txt', 'r') as f:
        doc_names = f.readlines()
        doc_names = [name.strip() for name in doc_names]

    folds = []
    for fold_idx in range(10):
        fold_idx_str = f'fold_{fold_idx}'
        fold_data = folds_json[fold_idx_str]
        train_indexes = from_doc_id_to_name(doc_ids=fold_data['train'], doc_names=doc_names)
        val_indexes = from_doc_id_to_name(doc_ids=fold_data['validation'], doc_names=doc_names)
        test_indexes = from_doc_id_to_name(doc_ids=fold_data['test'], doc_names=doc_names)
        folds.append((train_indexes, val_indexes, test_indexes))

    save_pickle('../prebuilt_folds/tos_folds.pkl', folds)
