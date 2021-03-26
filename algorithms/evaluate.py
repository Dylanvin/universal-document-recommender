class Evaluate:
    def get_score(self, doc_cats, query_cat):
        """
        Validation method. Splits the query document and given most similar document categories by their "."s. For
        each segment of the query document which is identical in the other documents 1 is added to the score. The score
        is then divided by the query category segment length.

        :param list[str] doc_cats:
        :param str query_cat:
        :return int: score
        """
        query_cat_ls = query_cat.split(".")
        score = 0
        for doc_cat in doc_cats:
            score = score + sum(el in query_cat_ls for el in doc_cat.split("."))

        return round(score/len(query_cat_ls), 5)
