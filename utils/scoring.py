from pycocoevalcap.eval import Bleu, Meteor, Rouge, Cider, Spice
from pycocoevalcap.eval import PTBTokenizer

def generate_scores(gts, res):
    tokenizer = PTBTokenizer()

    gts  = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    output = {}
    img_output = {}

    for scorer, method in scorers:
        print('computing {} score...'.format(scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) != list:
            method = [method]
            score = [score]
            scores = [scores]

        for sc, scs, m in zip(score, scores, method):
            print("%s: %0.3f" % (m, sc))
            output[m] = sc
            for img_id, score in zip(gts.keys(), scs):
                if type(score) is dict:
                    score = score['All']['f']

                if img_id not in img_output:
                    img_output[img_id] = {}
                img_output[img_id][m] = score

    return output, img_output
