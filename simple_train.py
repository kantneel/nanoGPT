
from model import GPT, GPTConfig
from simple_trainer import Trainer
from dataset.parallel_copy import CopyDataset
from dataset.sort import SortDataset
from torch.utils.data import DataLoader
import torch

def get_model_and_config(train_dataset):
    nano_gpt_config = GPTConfig(
        n_layer=3,
        n_head=3,
        n_embd=48,
        dropout=0.0,
        bias=False,
    )
    model_config = nano_gpt_config
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model = GPT(model_config)
    return model, model_config

def get_trainer(model, train_dataset):
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4
    train_config.max_iters = 2000
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, train_dataset)
    
    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    
    trainer.set_callback('on_batch_end', batch_end_callback)
    return trainer


def train_and_evaluate(train_dataset, test_dataset):
    model, model_config = get_model_and_config(train_dataset)
    trainer = get_trainer(model, train_dataset)
    
    # Run training
    trainer.run()
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluation function (copied from the notebook)
    def eval_split(trainer, split, max_batches):
        dataset = {'train': train_dataset, 'test': test_dataset}[split]
        n = train_dataset.length  # naughty direct access shrug
        results = []
        mistakes_printed_already = 0
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            
            end_x = n + int(hasattr(dataset, 'sep_id'))
            inp = x[:, :end_x]
            sol = y[:, -n:]
            
            cat = model.generate(inp, n, do_sample=False)
            sol_candidate = cat[:, end_x:]
            correct = (sol == sol_candidate).all(1).cpu()
            
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 3:
                    mistakes_printed_already += 1
                    print(f"GPT predicts output for {inp[i].tolist()} is {sol_candidate[i].tolist()} but label is {sol[i].tolist()}")
                    
            if max_batches is not None and b+1 >= max_batches:
                break
            
        rt = torch.tensor(results, dtype=torch.float)
        print(f"{split} final score: {rt.sum()}/{len(results)} = {100*rt.mean():.2f}% correct")
        return rt.sum()
    
    # Run evaluation
    with torch.no_grad():
        train_score = eval_split(trainer, 'train', max_batches=50)
        test_score = eval_split(trainer, 'test', max_batches=50)
    
    return model, trainer, train_score, test_score

# Usage:
train_dataset = CopyDataset('train')
test_dataset = CopyDataset('test')

model, trainer, train_score, test_score = train_and_evaluate(train_dataset, test_dataset)

train_dataset = SortDataset('train')
test_dataset = SortDataset('test')

model, trainer, train_score, test_score = train_and_evaluate(train_dataset, test_dataset)