from typing import Callable

import click
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tokenizer import Tokenizer
from dataset import SeqPairDataset
from model import EncoderDecoder
from utils import print_header, plot_comparison

#Set seed for reproducibility
torch.manual_seed(0)

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    model.train()
    model.to(device)
    total_loss = 0.0

    for encoder_input_ids, decoder_input_ids, label_ids in tqdm(dataloader, desc='Training'):
        encoder_input_ids = encoder_input_ids.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        label_ids = label_ids.to(device)

        logits = model(encoder_input_ids, decoder_input_ids)
        flattened_logits = logits.flatten(end_dim=1)
        flattened_labels = label_ids.flatten()

        loss = loss_fn(flattened_logits, flattened_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def test_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    model.eval()
    model.to(device)
    total_loss = 0.0

    with torch.no_grad():
        for encoder_input_ids, decoder_input_ids, label_ids in tqdm(dataloader, desc='Evaluating'):
            encoder_input_ids = encoder_input_ids.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            label_ids = label_ids.to(device)

            logits = model(encoder_input_ids, decoder_input_ids)
            flattened_logits = logits.flatten(end_dim=1)
            flattened_labels = label_ids.flatten()

            loss = loss_fn(flattened_logits, flattened_labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def compute_bleu(
        model: EncoderDecoder,
        dataloader: DataLoader,
        tok: Tokenizer,
        max_len: int = 50,
        strategy: str = "greedy",
        beam_width: int = 5,
        out_file: str = 'results/bleu_output.txt',
        exp_2 = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ) -> float: 
    """
    Computes BLEU score on output of model's generate() function using
    NLTK sentence_bleu and SmoothingFunction().method4
    """
    smoothing = SmoothingFunction()

    model.eval()
    model.to(device)

    total_bleu, num_samples = (0.0, 0)
    total_time, total_seq_len = (0.0, 0)

    with torch.no_grad():
        for encoder_input_ids, decoder_input_ids, label_ids in tqdm(dataloader, desc='Evaluating'):
            encoder_input_ids = encoder_input_ids.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            label_ids = label_ids.to(device)

            start = time.process_time()
            preds = model.generate(
                        src_ids=encoder_input_ids,
                        bos_id=tok.bos_id,
                        eos_id=tok.eos_id,
                        max_len=max_len,
                        strategy=strategy,
                        beam_width=beam_width
                    )
            end = time.process_time()
            total_time += end - start
            
            specials = {tok.bos_id, tok.eos_id, tok.pad_id}
            weights = (1.0, 0, 0, 0)
            for pred, ref_ids in zip(preds, label_ids.tolist()): 
                #Remove special tokens
                pred_ids = [tok_id for tok_id in pred if tok_id not in specials]
                #Decode prediction
                pred = tok.decode(pred_ids)
                total_seq_len += len(pred)

                #Remove special tokens from ground truth
                ref_ids = [tok_id for tok_id in ref_ids if tok_id not in specials]
                #Decode ground truth
                ref = tok.decode(ref_ids)
                if(num_samples % 1250 == 0 and out_file is not None):
                    with open(out_file, 'a', encoding='utf8') as f:
                        f.write(f'REF : {" ".join(ref)}\n')
                        f.write(f'PRED: {" ".join(pred)}\n\n')

                bleu = sentence_bleu([ref], pred, weights=weights, smoothing_function=smoothing.method4)
                total_bleu += bleu
                num_samples += 1

    if(exp_2):
        print(f'Time per sample: {total_time / num_samples:.04f} seconds', flush=True)
        print(f'Average sequence length: {total_seq_len / num_samples:.02f}', flush=True)
        print(f'BLEU score: {total_bleu / num_samples}', flush=True)
    return total_bleu / num_samples


def grid_search(
        train: DataLoader,
        dev: DataLoader, 
        tokenizer: Tokenizer,
        models_dir: str,
):
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    dropout = 0.1
    max_len = 50
    lr = 1e-3
    epochs = 3

    with open(f'{models_dir}/grid_search.csv', 'w', encoding='utf8') as f: 
        f.write('Num Layers,Model Dim,Num Heads,FF Dim,BLEU Score\n')

    for num_layers in [1]:
        for model_dim in [128, 256]:
            for num_heads in [2, 4, 8]:
                for ff_dim in [256, 512]:
                    model_fn = f'{models_dir}/model.{num_layers}.{model_dim}.{num_heads}.{ff_dim}.pt'
                    print(f'Training for {model_fn}')

                    model = EncoderDecoder(
                        src_vocab_size=len(tokenizer.src_vocab),
                        tgt_vocab_size=len(tokenizer.tgt_vocab),
                        d_model=model_dim,
                        num_dec_layers=num_layers,
                        num_enc_layers=num_layers,
                        num_heads=num_heads,
                        d_ff=ff_dim,
                        max_len=max_len, 
                        dropout=dropout,
                        pad_idx=tokenizer.pad_id
                    )
                    optim = torch.optim.Adam(model.parameters(), lr=lr)

                    for _ in range(epochs):
                        train_epoch(model=model, dataloader=train, loss_fn=loss_fn, optimizer=optim)
                        test_epoch(model=model, dataloader=dev, loss_fn=loss_fn)
                        
                    torch.save(model.state_dict(), model_fn)

                    bleu = compute_bleu(model=model, dataloader=dev, tok=tokenizer, out_file=None)
                    with open(f'{models_dir}/grid_search.csv', 'a', encoding='utf8') as f: 
                        f.write(f'{num_layers},{model_dim},{num_heads},{ff_dim},{bleu}\n')


def experiment_1(
        hyperparams: dict, 
        train: DataLoader, 
        dev: DataLoader, 
        test: DataLoader, 
        tokenizer: Tokenizer,
        results_dir: str,   
    ):
    print_header('Experiment 1: Learnable vs Sinusoidal Positional Encoding')

    #Create baseline
    baseline = EncoderDecoder(
        src_vocab_size=len(tokenizer.src_vocab),
        tgt_vocab_size=len(tokenizer.tgt_vocab),
        num_dec_layers=hyperparams['num_layers'],
        num_enc_layers=hyperparams['num_layers'],
        num_heads=hyperparams['num_heads'],
        d_ff=hyperparams['ff_dim'],
        d_model=hyperparams['model_dim'],
        max_len=hyperparams['max_len'],
        dropout=hyperparams['dropout'],
        pad_idx=tokenizer.pad_id
    )

    #Create learnable pos encoding model
    alternative = EncoderDecoder(
        src_vocab_size=len(tokenizer.src_vocab),
        tgt_vocab_size=len(tokenizer.tgt_vocab),
        num_dec_layers=hyperparams['num_layers'],
        num_enc_layers=hyperparams['num_layers'],
        num_heads=hyperparams['num_heads'],
        d_ff=hyperparams['ff_dim'],
        d_model=hyperparams['model_dim'],
        max_len=hyperparams['max_len'],
        dropout=hyperparams['dropout'],
        pad_idx=tokenizer.pad_id,
        exp_1=True
    )

    #Print num of parameters
    baseline_params = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    alternative_params = sum(p.numel() for p in alternative.parameters() if p.requires_grad)
    print(f'Baseline Parameters: {baseline_params}')
    print(f'Alternative Parameters: {alternative_params}')

    base_train_losses, base_val_losses = ([], [])
    alt_train_losses, alt_val_losses = ([], [])
    #Train and evaluate both
    for _ in range(hyperparams['num_epochs']):
        base_train_loss = train_epoch(model=baseline, dataloader=train,
                                      optimizer=torch.optim.Adam(baseline.parameters(), lr=1e-3),
                                      loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id))
        base_test_loss = test_epoch(model=baseline, dataloader=dev,
                                    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id))
        base_train_losses.append(base_train_loss)
        base_val_losses.append(base_test_loss)
        print(f'Baseline Train Loss: {base_train_loss:.04f}, Val Loss: {base_test_loss:.04f}')

        alt_train_loss = train_epoch(model=alternative, dataloader=train,
                                     optimizer=torch.optim.Adam(alternative.parameters(), lr=1e-3),
                                     loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id))
        alt_test_loss = test_epoch(model=alternative, dataloader=dev,
                                   loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id))
        alt_train_losses.append(alt_train_loss)
        alt_val_losses.append(alt_test_loss)
        print(f'Alternative Train Loss: {alt_train_loss:.04f}, Val Loss: {alt_test_loss:.04f}')

    #Save results plot
    plot_comparison(
        losses=[base_train_losses, alt_train_losses],
        title='Training Loss: Learnable vs Sinusoidal Positional Encoding',
        legend=['Sinusoidal', 'Learnable'],
        fn=f'{results_dir}/exp1_train.png'
    )

    plot_comparison(
        losses=[base_val_losses, alt_val_losses],
        title='Validation Loss: Learnable vs Sinusoidal Positional Encoding',
        legend=['Sinusoidal', 'Learnable'],
        fn=f'{results_dir}/exp1_val.png'
    )

    #Compute BLEU scores
    baseline_bleu = compute_bleu(model=baseline, dataloader=test, tok=tokenizer, out_file=f'{results_dir}/exp1_baseline.txt')
    alternative_bleu = compute_bleu(model=alternative, dataloader=test, tok=tokenizer, out_file=f'{results_dir}/exp1_alternative.txt')
    print(f'Baseline BLEU: {baseline_bleu:.04f}')
    print(f'Alternative BLEU: {alternative_bleu:.04f}')


def experiment_2(
        hyperparams: dict,
        dataloader: DataLoader,
        tokenizer: Tokenizer,
        results_dir: str,
        pretrained: bool = False,
        models_dir: str = None,
):
    beam_sizes = [3, 5, 10]

    model = EncoderDecoder(
        src_vocab_size=len(tokenizer.src_vocab),
        tgt_vocab_size=len(tokenizer.tgt_vocab),
        num_dec_layers=hyperparams['num_layers'],
        num_enc_layers=hyperparams['num_layers'],
        num_heads=hyperparams['num_heads'],
        d_ff=hyperparams['ff_dim'],
        d_model=hyperparams['model_dim'],
        max_len=hyperparams['max_len'],
        dropout=hyperparams['dropout'],
        pad_idx=tokenizer.pad_id
    )
    #Load in pretrained model 
    if pretrained:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(torch.load(f'{models_dir}/model.{hyperparams["num_layers"]}.{hyperparams["model_dim"]}.{hyperparams["num_heads"]}.{hyperparams["ff_dim"]}.pt', weights_only=True, map_location=device))

    else:
        for _ in range(hyperparams['num_epochs']):
            train_epoch(model=model, dataloader=dataloader,
                        loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id),
                        optimizer=torch.optim.Adam(model.parameters(), lr=hyperparams['lr']))
            
    print_header('Greedy')
    compute_bleu(model=model, dataloader=dataloader, tok=tokenizer,
                 strategy='greedy', exp_2=True, out_file=f'{results_dir}/exp2_greedy.txt')

    for size in beam_sizes[1:]:
        print_header(f'Beam Search with beam width {size}')
        compute_bleu(model=model, dataloader=dataloader, tok=tokenizer,
                     strategy='beam_search', beam_width=size, exp_2=True, out_file=f'{results_dir}/exp2_beam_{size}.txt')


def experiment_3(
        hyperparams: dict,
        train: DataLoader,
        dev: DataLoader,
        test: DataLoader,
        tokenizer: Tokenizer,
        results_dir: str,
        pretrained: bool = False,
        models_dir: str = None,
):
    print_header('Experiment 3a: Number of Attention Heads')
    num_heads = [2,4,8]
    train_losses_3a, val_losses_3a = ([], [])  

    for heads in num_heads:
        train_losses, val_losses = ([], [])
        model = EncoderDecoder(
            src_vocab_size=len(tokenizer.src_vocab),
            tgt_vocab_size=len(tokenizer.tgt_vocab),
            num_dec_layers=hyperparams['num_layers'],
            num_enc_layers=hyperparams['num_layers'],
            num_heads=heads,
            d_ff=hyperparams['ff_dim'],
            d_model=hyperparams['model_dim'],
            max_len=50,
            dropout=0.1,
            pad_idx=tokenizer.pad_id
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        if pretrained:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.load_state_dict(torch.load(f'{models_dir}/model.{hyperparams["num_layers"]}.{hyperparams["model_dim"]}.{heads}.{hyperparams["ff_dim"]}.pt', weights_only=True, map_location=device))
        else: 
            for _ in range(hyperparams['num_epochs']):
                train_losses.append(train_epoch(model=model, dataloader=train, loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id), optimizer=optim))
                val_losses.append(test_epoch(model=model, dataloader=dev, loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)))
                print(f'Heads: {heads}, Train Loss: {train_losses[-1]:.04f}, Val Loss: {val_losses[-1]:.04f}')

        train_losses_3a.append(train_losses)
        val_losses_3a.append(val_losses)

        bleu = compute_bleu(model=model, dataloader=test, tok=tokenizer, out_file=f'{results_dir}/exp3a_heads_{heads}.txt')
        print(f'BLEU score with {heads} heads: {bleu:.04f}')  

    plot_comparison(
        losses=train_losses_3a,
        title='Training Loss: Number of Attention Heads',
        legend=[f'{h} heads' for h in num_heads],
        fn=f'{results_dir}/exp3a_train.png'
    )

    plot_comparison(
        losses=val_losses_3a,
        title='Validation Loss: Number of Attention Heads',
        legend=[f'{h} heads' for h in num_heads],
        fn=f'{results_dir}/exp3a_val.png'
    )

    print_header('Experiment 3b: Encoder/Decoder Depth')
    num_layers = [1,2,4]
    train_losses_3b, val_losses_3b = ([], [])

    for layers in num_layers:
        train_losses, val_losses = ([], [])
        model = EncoderDecoder(
            src_vocab_size=len(tokenizer.src_vocab),
            tgt_vocab_size=len(tokenizer.tgt_vocab),
            num_dec_layers=layers,
            num_enc_layers=layers,
            num_heads=hyperparams['num_heads'],
            d_ff=hyperparams['ff_dim'],
            d_model=hyperparams['model_dim'],
            max_len=50,
            dropout=0.1,
            pad_idx=tokenizer.pad_id
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        if pretrained:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.load_state_dict(torch.load(f'{models_dir}/model.{layers}.{hyperparams["model_dim"]}.{hyperparams["num_heads"]}.{hyperparams["ff_dim"]}.pt', weights_only=True, map_location=device))
        else: 
            for _ in range(hyperparams['num_epochs']):
                train_losses.append(train_epoch(model=model, dataloader=train, loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id), optimizer=optim))
                val_losses.append(test_epoch(model=model, dataloader=dev, loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)))
                print(f'Layers: {layers}, Train Loss: {train_losses[-1]:.04f}, Val Loss: {val_losses[-1]:.04f}')

            train_losses_3b.append(train_losses)
            val_losses_3b.append(val_losses)

        bleu = compute_bleu(model=model, dataloader=test, tok=tokenizer, out_file=f'{results_dir}/exp3b_layers_{layers}.txt')
        print(f'BLEU score with {layers} layers: {bleu:.04f}')

    plot_comparison(
        losses=train_losses_3b,
        title='Training Loss: Encoder/Decoder Depth',
        legend=[f'{l} layers' for l in num_layers],
        fn=f'{results_dir}/exp3b_train.png'
    )

    plot_comparison(
        losses=val_losses_3b,
        title='Validation Loss: Encoder/Decoder Depth',
        legend=[f'{l} layers' for l in num_layers],
        fn=f'{results_dir}/exp3b_val.png'
    )
    

@click.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.argument('dev_file', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--grid-search', is_flag=True, help='Run grid search')
@click.option('--models-dir', type=click.Path(exists=True), help='Directory to save/load models')
@click.option('--experiment', default='all', type=click.Choice(['1', '2', '3', 'all']), help='Run a specific experiment or all')
def main(
    train_file: str,
    dev_file: str,
    test_file: str,
    grid_search: bool,
    results_dir: str,
    models_dir: str,
    experiment: str,
):
    hyperparams = {
        'num_layers': 4,
        'model_dim': 256,
        'num_heads': 8,
        'ff_dim': 256,
        'lr': 1e-3,
        'dropout': 0.1,
        'num_epochs': 3,
        'batch_size': 32, 
        'max_len': 50,
    }

    print("Tokenizing")
    tok = Tokenizer()
    tok.from_file(train_file)

    print("Making datasets")
    max_src_len = hyperparams['max_len']
    max_tgt_len = hyperparams['max_len']

    train = SeqPairDataset(train_file, tok, max_src_len, max_tgt_len)
    dev = SeqPairDataset(dev_file, tok, max_src_len, max_tgt_len)
    test = SeqPairDataset(test_file, tok, max_src_len, max_tgt_len)

    batch_size = hyperparams['batch_size']

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev, batch_size=batch_size)
    test_dataloader = DataLoader(test, batch_size=batch_size)    

    if grid_search:
        print("Starting grid search")
        grid_search(
            train=train_dataloader,
            dev=dev_dataloader,
            tokenizer=tok,
            models_dir=models_dir,
        )

    print("Running experiments")
    if experiment in ['1', 'all']:
        experiment_1(
            hyperparams=hyperparams,
            train=train_dataloader,
            dev=dev_dataloader,
            test=test_dataloader,
            tokenizer=tok,
            results_dir=results_dir,
        )

    if experiment in ['2', 'all']:
        experiment_2(
            hyperparams=hyperparams,
            dataloader=test_dataloader,
            tokenizer=tok,
            results_dir=results_dir,

        )
        
    if experiment in ['3', 'all']:  
        experiment_3(
            hyperparams=hyperparams,
            train=train_dataloader,
            dev=dev_dataloader,
            test=test_dataloader,
            tokenizer=tok,
            results_dir=results_dir,
        )
    print("Finished")

if __name__ == "__main__":
    main()
