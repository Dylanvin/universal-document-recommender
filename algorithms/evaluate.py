class Evaluate:
    def get_score(self, doc_cats, query_cat):
        query_cat_ls = query_cat.split(".")
        score = 0
        for doc_cat in doc_cats:
             score = score + sum(el in query_cat_ls for el in doc_cat.split("."))

        return round(score/len(query_cat_ls), 5)