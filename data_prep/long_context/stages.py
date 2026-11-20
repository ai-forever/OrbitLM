"""Maps the --stage argument of the tools to the stage definitions in stage_16k.py and stage_32k.py."""
import stage_16k
import stage_32k

STAGES = {stage_16k.NAME: stage_16k, stage_32k.NAME: stage_32k}


def add_stage_argument(parser):
    parser.add_argument("--stage", required=True, choices=sorted(STAGES),
                        help="which stage's block length, root and stream definitions to use")


def get(name):
    return STAGES[name]
