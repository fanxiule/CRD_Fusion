from trainer import CRDFusionTrainer
from train_options import TrainOptions

options = TrainOptions()
opts = options.parse()

if __name__ == "__main__":
    trainer = CRDFusionTrainer(opts)
    trainer.train()
