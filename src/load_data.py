from datasets import load_dataset 
import json 
import config 

def prepare_dataset():
    full_data = load_dataset('ag_news', split='train')
    full_data = full_data.rename_column('label', 'labels')
    
    split_df = full_data.train_test_split(test_size=0.2, 
                                            stratify_by_column='labels',
                                            seed=515,
                                            )
    return split_df['train'], split_df['test']


if __name__ == "__main__":

    train_df, val_df = prepare_dataset()
    label_distribution ={
        'train' : train_df.with_format('pandas')['labels'].value_counts(normalize=True).to_dict(), 
        'validation' : val_df.with_format('pandas')['labels'].value_counts(normalize=True).to_dict()
    } 
    
    with open(config.ROOT/'data_split_result.json', 'w') as fp:
        json.dump(label_distribution, fp)
