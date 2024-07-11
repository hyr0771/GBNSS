from utils import tab_printer
from param_parser import parameter_parser
from model import GBNSS_Topological_Trainer, GBNSS_Biological_Trainer

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    if args.label == "go":
        trainer = GBNSS_Biological_Trainer(args)
    else: 
        trainer = GBNSS_Topological_Trainer(args)

    trainer.fit()

    trainer.save()
    trainer.score()

if __name__ == "__main__":
    main()