"""Run dataset generation, training, and evaluation from one command."""
from __future__ import annotations
import argparse
from part2_build_graph import save_dataset
from part5_train import train
from part6_test import run_test

def main():
    parser=argparse.ArgumentParser(description="Run the complete BEM-attention experiment.")
    parser.add_argument("--dataset-root",default="dataset_rect_bem_gnn")
    parser.add_argument("--train-dir",default="artifacts_train")
    parser.add_argument("--test-dir",default="artifacts_test")
    parser.add_argument("--epochs",type=int,default=1500)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--hidden-channels",type=int,default=128)
    parser.add_argument("--latent-channels",type=int,default=32)
    parser.add_argument("--decoder-channels",type=int,default=128)
    parser.add_argument("--dropout",type=float,default=0.05)
    parser.add_argument("--beta-kl",type=float,default=1e-5)
    parser.add_argument("--skip-data",action="store_true")
    parser.add_argument("--skip-train",action="store_true")
    parser.add_argument("--skip-test",action="store_true")
    a=parser.parse_args()
    if not a.skip_data: save_dataset(a.dataset_root)
    if not a.skip_train:
        train(dataset_root=a.dataset_root,save_dir=a.train_dir,epochs=a.epochs,lr=a.lr,
              hidden_channels=a.hidden_channels,latent_channels=a.latent_channels,
              decoder_channels=a.decoder_channels,dropout=a.dropout,beta_kl=a.beta_kl)
    if not a.skip_test: run_test(dataset_root=a.dataset_root,model_dir=a.train_dir,out_dir=a.test_dir)
if __name__=="__main__": main()
