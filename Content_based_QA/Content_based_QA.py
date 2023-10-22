# JiangHao
from haystack.utils import clean_wiki_text, convert_files_to_docs, \
    fetch_archive_from_http, print_answers
from haystack.nodes import FARMReader, TransformersReader, DensePassageRetriever
from haystack.nodes import TfidfRetriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
import os


class Conten_based_QA:
    def __init__(self):
        document_store = InMemoryDocumentStore()
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # 预处理数据
        docs = convert_files_to_docs(
            dir_path=os.path.join(cur_dir, 'data'),
            clean_func=clean_wiki_text,
            split_paragraphs=False)
        document_store.write_documents(docs)

        # reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2",
        #                     use_gpu=False)
        reader = FARMReader(
            model_name_or_path=os.path.join(cur_dir, 'my_model'), use_gpu=False,
            num_processes=0)

        # retriever = DensePassageRetriever(
        #     document_store=document_store,
        #     query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        #     passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        #     max_seq_len_query=64,
        #     max_seq_len_passage=256,
        #     batch_size=16,
        #     use_gpu=True,
        #     embed_title=True,
        #     use_fast_tokenizers=True,
        # )
        # document_store.update_embeddings(retriever)
        retriever = TfidfRetriever(document_store=document_store)

        self.pipe = ExtractiveQAPipeline(reader, retriever)

        print("Content Based QA model init finished ......")

    def get_answer(self, question):
        prediction = self.pipe.run(
            query=question,
            params={"Retriever": {"top_k": 2}, "Reader": {"top_k": 5}}
        )

        return prediction['answers'][0].answer, prediction['no_ans_gap']

if __name__ == '__main__':
    Conten_based_QA = Conten_based_QA()
    while True:
        question = input("input question:")
        answer, no_answer = Conten_based_QA.get_answer(question)

        print(answer)
        print(no_answer)
