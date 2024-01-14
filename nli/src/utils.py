import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# nli model
device = "cuda"
nli_model = AutoModelForSequenceClassification.from_pretrained(
    "facebook/bart-large-mnli"
)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
nli_model.to(device)
nli_model.eval()


def populate_question_with_entailment(x, batch_size=8):
    """
    run the nli model and fill the nli_true_prob and sub_questions_nli_true_prob fields
    """
    premise = x["question"]["decompositions"][0].split("Question:")[0]
    if len(x["gpt_answers"]):
        hypothesis = (
            (
                x["question"]["question"].strip()
                if x["question"]["question"].strip()[-1] == "?"
                else x["question"]["question"].strip() + "?"
            )
            + " The answer is: "
            + x["gpt_answers"][0]
        )

        with torch.no_grad():  # Disable autograd to save memory
            # run through model pre-trained on MNLI
            toks = tokenizer.encode(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation_strategy="only_first",
            ).to(device)
            logits = nli_model(toks)[0]

            # we throw away "neutral" (dim 1) and take the probability of
            # "entailment" (2) as the probability of the label being true
            entail_contradiction_logits = logits[:, [0, 2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:, 1]
            x["nli_true_prob"] = float(prob_label_is_true)
            x["nli_probs"] = [float(x) for x in probs[0]]
            x["sub_questions_nli_true_prob"] = []

        torch.cuda.empty_cache()  # Clear GPU memory after processing

        # Batch processing for sub_questions
        sub_questions = [
            sub_question
            for sub_question in x["question"]["decompsition_steps"][0]
            if sub_question["question"] is not None
        ]
        for i in range(0, len(sub_questions), batch_size):
            batch = sub_questions[i : i + batch_size]
            hypotheses = [
                sub_question["question"] + " " + sub_question["answer"]
                for sub_question in batch
            ]

            with torch.no_grad():  # Disable autograd to save memory
                # run through model pre-trained on MNLI
                batch_encoded = tokenizer.batch_encode_plus(
                    [(premise, hypothesis) for hypothesis in hypotheses],
                    return_tensors="pt",
                    padding=True,
                    truncation_strategy="only_first",
                ).to(device)
                logits = nli_model(**batch_encoded)[0]

                # we throw away "neutral" (dim 1) and take the probability of
                # "entailment" (2) as the probability of the label being true
                entail_contradiction_logits = logits[:, [0, 2]]
                probs = entail_contradiction_logits.softmax(dim=1)
                prob_label_is_true = probs[:, 1]
                for j, sub_question in enumerate(batch):
                    x["sub_questions_nli_true_prob"].append(prob_label_is_true[j])

            torch.cuda.empty_cache()  # Clear GPU memory after processing
