from __future__ import annotations
import argparse, json, os
from augment import augment_pipeline
from train_gen import train_generator
from train_disc import train_discriminator
from eval import evaluate
from hitl import sample_simple_for_hitl, ingest_simple_csv

def main():
    ap = argparse.ArgumentParser("MXene-GAN CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap_aug = sub.add_parser("augment", help="Build JSONL dataset and vocab")
    ap_aug.add_argument("--data", required=True, help="data.csv path")
    ap_aug.add_argument("--out_dir", required=True, help="output directory")
    ap_aug.add_argument("--alpha", type=float, default=0.6)
    ap_aug.add_argument("--beta", type=float, default=0.3)
    ap_aug.add_argument("--gamma", type=float, default=0.1)

    ap_tg = sub.add_parser("train-gen", help="Train generator")
    ap_tg.add_argument("--config", required=True)

    ap_td = sub.add_parser("train-disc", help="Train discriminator")
    ap_td.add_argument("--config", required=True)

    ap_ev = sub.add_parser("eval", help="Evaluate generator + discriminator")
    ap_ev.add_argument("--config", required=True)
    ap_ev.add_argument("--gen_config", required=True)
    ap_ev.add_argument("--disc_config", required=True)
    
    ap_prop = sub.add_parser("hitl.sample", help="Propose a human-review batch")
    ap_prop.add_argument("--gen_ckpt", required=True)
    ap_prop.add_argument("--disc_ckpt", default="")
    ap_prop.add_argument("--out_csv", required=True)
    ap_prop.add_argument("--num_samples", type=int, default=2000)
    ap_prop.add_argument("--temperature", type=float, default=1)
    ap_prop.add_argument("--top_p", type=float, default=0.9)

    ap_ing = sub.add_parser("hitl.ingest", help="Ingest reviewed CSV into JSONLs")
    ap_ing.add_argument("--review_csv", required=True)
    ap_ing.add_argument("--out_dir", required=True)

    args = ap.parse_args()
    if args.cmd == "augment":
        augment_pipeline(args.data, args.out_dir, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    elif args.cmd == "train-gen":
        cfg = json.load(open(args.config))
        os.makedirs(cfg["run_dir"], exist_ok=True)
        train_generator(cfg)
    elif args.cmd == "train-disc":
        cfg = json.load(open(args.config))
        os.makedirs(cfg["run_dir"], exist_ok=True)
        train_discriminator(cfg)
    elif args.cmd == "eval":
        cfg = json.load(open(args.config))
        gen_cfg = json.load(open(args.gen_config))
        disc_cfg = json.load(open(args.disc_config))
        os.makedirs(cfg["run_dir"], exist_ok=True)
        evaluate(cfg, gen_cfg, disc_cfg)
    elif args.cmd == "hitl.sample":
        sample_simple_for_hitl(args.gen_ckpt, args.out_csv, args.num_samples, args.top_p, args.temperature, disc_ckpt=args.disc_ckpt)
    elif args.cmd == "hitl.ingest":
        ingest_simple_csv(args.review_csv, args.out_dir)

if __name__ == "__main__":
    main()
