import numpy as np
import ast
import pandas as pd
import argparse
from tqdm import tqdm
from nli.utils import populate_question_with_entailment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_retrieval_csv",
        type=str,
        default="data/nli_example/bamboogle_no_retrieval.csv",
    )
    parser.add_argument(
        "--with_retrieval_csv",
        type=str,
        default="data/nli_example/bamboogle_with_retrieval.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    return parser.parse_args()


def run_nli(args):
    """run nli model to filter irrelevant retrieved context"""

    # parse args to vars
    t, no_retrieval_csv, with_retrieval_csv = (
        args.threshold,
        args.no_retrieval_csv,
        args.with_retrieval_csv,
    )

    # read data
    no_retrieval = pd.read_csv(no_retrieval_csv).to_dict("rows")
    with_retrieval = pd.read_csv(with_retrieval_csv).to_dict("rows")

    # preprocess files
    for d in [no_retrieval, with_retrieval]:
        for x in d:
            if x["acc@1"] is not None:
                x["acc@1"] = float(x["acc@1"]) if x["acc@1"] != "FALSE" else 0
            x["question"] = ast.literal_eval(x["question"])
            x["gpt_answers"] = ast.literal_eval(x["gpt_answers"])
    no_retrieval_questions = {x["question"]["question"] for x in no_retrieval}
    with_retrieval = [
        x for x in with_retrieval if x["question"]["question"] in no_retrieval_questions
    ]
    no_retrieval.sort(key=lambda x: x["question"]["question"])
    with_retrieval.sort(key=lambda x: x["question"]["question"])

    # populate question
    for _, q in tqdm(enumerate(with_retrieval)):
        populate_question_with_entailment(q)

    # calculate res and print results
    res = np.average(
        [
            no_retrieval[i]["acc@1"]
            if "nli_true_prob" not in x
            or float(min([x["nli_true_prob"]] + x["sub_questions_nli_true_prob"])) <= t
            else x["acc@1"]
            for i, x in enumerate(with_retrieval)
        ]
    )
    res_no_nli = np.average([x["acc@1"] for x in with_retrieval])
    res_no_retrieval = np.average([x["acc@1"] for x in no_retrieval])
    print(
        f"Threshold: {t}, Res: {res*100:.1f}, Res no retrieval: {res_no_retrieval*100:.1f}, Res no NLI: {res_no_nli*100:.1f}"
    )


if __name__ == "__main__":
    """ """
    args = parse_args()
    run_nli(args)
