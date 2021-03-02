
class Evaluate:
    def get_score(self, doc_cats, query_cat):
        score = 0
        for doc_cat in doc_cats:
            score = score + sum(el in query_cat.split(".") for el in doc_cat.split("."))

        return score