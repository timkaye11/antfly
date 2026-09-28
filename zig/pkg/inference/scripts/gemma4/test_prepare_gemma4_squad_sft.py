import copy
import unittest

from prepare_gemma4_squad_sft import order_key, select


def fixture():
    articles = []
    for index in range(20):
        context = f"Article {index} contains the answer red fox and more context."
        articles.append(
            {
                "title": f"article-{index}",
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "id": f"q-{index}",
                                "question": "Which animal?",
                                "answers": [
                                    {
                                        "text": "red fox",
                                        "answer_start": context.index("red fox"),
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    return {"version": "1.1", "data": articles}


def selection(data):
    return select(data, lambda text: len(text.split()), train_count=8, eval_count=4)


class SquadPreparationTests(unittest.TestCase):
    def test_disjoint_articles_and_passages_with_multitoken_supervision(self):
        result = selection(fixture())
        for field in ("id", "article", "context_sha256"):
            self.assertFalse(
                {r[field] for r in result["train"]} & {r[field] for r in result["eval"]}
            )
        self.assertEqual(len(result["train"]), 8)
        self.assertEqual(len(result["eval"]), 4)
        for rows in result.values():
            for row in rows:
                self.assertEqual(row["answer_tokens"], 2)
                self.assertEqual(row["record"]["messages"][-1]["content"], "red fox")

    def test_selection_independent_of_input_order(self):
        data = fixture()
        shuffled = copy.deepcopy(data)
        shuffled["data"].reverse()
        self.assertEqual(selection(data), selection(shuffled))

    def test_corrupt_answer_span_rejected(self):
        data = fixture()
        data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"] = 0
        with self.assertRaisesRegex(ValueError, "declared source span"):
            selection(data)

    def test_duplicate_question_rejected_even_in_another_article(self):
        data = fixture()
        data["data"][1]["paragraphs"][0]["qas"][0]["id"] = "q-0"
        with self.assertRaisesRegex(ValueError, "duplicate question"):
            selection(data)

    def test_duplicate_passage_across_articles_rejected(self):
        data = fixture()
        ordered = sorted(data["data"], key=lambda a: order_key(a["title"]))
        ordered[-1]["paragraphs"][0]["context"] = ordered[0]["paragraphs"][0]["context"]
        ordered[-1]["paragraphs"][0]["qas"][0]["answers"] = ordered[0]["paragraphs"][0][
            "qas"
        ][0]["answers"]
        with self.assertRaisesRegex(ValueError, "duplicate passage"):
            selection(data)

    def test_insufficient_multitoken_examples_rejected(self):
        data = fixture()
        for article in data["data"]:
            answer = article["paragraphs"][0]["qas"][0]["answers"][0]
            answer["text"] = "red"
        with self.assertRaisesRegex(ValueError, "insufficient eligible"):
            selection(data)

    def test_one_question_per_passage(self):
        data = fixture()
        for article in data["data"]:
            paragraph = article["paragraphs"][0]
            qa = copy.deepcopy(paragraph["qas"][0])
            qa["id"] += "-duplicate-passage"
            paragraph["qas"].append(qa)
        result = selection(data)
        for rows in result.values():
            self.assertEqual(len(rows), len({r["context_sha256"] for r in rows}))


if __name__ == "__main__":
    unittest.main()
