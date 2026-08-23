from pathlib import Path
import yaml,pandas as pd, numpy as np
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer

def main(config='configs/nlp_train.yaml'):
 c=yaml.safe_load(open(config)); labels=[x.strip() for x in c['labels'].split(',')]; df=pd.read_csv(c['data_file']); assert 'text' in df.columns
 for x in labels: assert x in df.columns, f'Missing label column {x}'
 ds=Dataset.from_pandas(df[['text']+labels]); tok=AutoTokenizer.from_pretrained(c['base_model'])
 def enc(b): return tok(b['text'],truncation=True,max_length=c['max_length'])
 ds=ds.map(enc,batched=True)
 def make_labels(ex): ex['labels']=[float(ex[x]) for x in labels]; return ex
 ds=ds.map(make_labels); ds=ds.remove_columns(['text']+labels)
 m=AutoModelForSequenceClassification.from_pretrained(c['base_model'],num_labels=len(labels),problem_type='multi_label_classification',id2label=dict(enumerate(labels)),label2id={x:i for i,x in enumerate(labels)})
 args=TrainingArguments(output_dir=c['output_dir'],num_train_epochs=c['epochs'],per_device_train_batch_size=c['batch_size'],learning_rate=c['learning_rate'],report_to='mlflow',save_strategy='epoch')
 Trainer(model=m,args=args,train_dataset=ds,processing_class=tok).train(); Path(c['output_dir']).mkdir(parents=True,exist_ok=True); m.save_pretrained(c['output_dir']); tok.save_pretrained(c['output_dir'])
if __name__=='__main__':main()
